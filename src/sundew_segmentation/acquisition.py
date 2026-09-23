"""Licensed iNaturalist acquisition for the sundew segmentation dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.error import URLError
from urllib.request import Request, urlopen
import json
import re
import time

from PIL import Image


INAT_API = "https://api.inaturalist.org/v1/observations"

# Every licence this project is willing to touch, and where it is defined.
# NoDerivatives is absent on purpose: masks and crops are derivative works, so
# ND photographs could be trained on but never illustrated or shared.
# ShareAlike is absent on purpose too, and that one is a judgement call rather
# than a plain reading -- see docs/licence-policy.md.
LICENSE_URLS = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
}

# The default, and the policy for the segmentation dataset: no NonCommercial.
# The species corpus passes a wider set explicitly; nothing widens by accident.
ALLOWED_LICENSES = {k: LICENSE_URLS[k] for k in ("cc0", "cc-by")}
NONCOMMERCIAL_LICENSES = dict(LICENSE_URLS)

USER_AGENT = "SundewSegmentation/0.1 (personal noncommercial research)"


@dataclass(frozen=True)
class Candidate:
    observation_id: int
    photo_id: int
    image_url: str
    source_page: str
    license_code: str
    license_url: str
    attribution: str
    creator: str
    taxon_id: int | None
    taxon_name: str
    common_name: str | None
    observer_login: str
    observed_on: str | None
    source_width: int | None
    source_height: int | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def original_photo_url(url: str) -> str:
    """Convert an iNaturalist thumbnail URL to its original-size variant."""
    parts = urlsplit(url)
    path = re.sub(
        r"/(square|thumb|small|medium|large)\.",
        "/original.",
        parts.path,
        count=1,
    )
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def build_observation_url(page: int, per_page: int = 200) -> str:
    params = {
        "taxon_id": "51935",
        "photos": "true",
        "quality_grade": "research",
        "captive": "false",
        "photo_license": "cc0,cc-by",
        # iNaturalist caches a random ordering long enough for consistent paging.
        # This avoids a collection dominated by the newest season or region.
        "order_by": "random",
        "per_page": str(per_page),
        "page": str(page),
    }
    return f"{INAT_API}?{urlencode(params)}"


def fetch_json(url: str, timeout: float = 45.0) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def extract_candidates(
    observations: Iterable[Mapping[str, Any]],
    allowed_licenses: Mapping[str, str] = ALLOWED_LICENSES,
) -> list[Candidate]:
    """Return one eligible photograph per observation without location fields.

    ``allowed_licenses`` defaults to the narrow CC0/CC-BY policy, so a caller
    that widens the API query without widening this too gets nothing back
    rather than silently acquiring photographs it did not mean to.
    """
    candidates: list[Candidate] = []
    seen_photos: set[int] = set()
    for observation in observations:
        if observation.get("quality_grade") != "research" or observation.get("captive") is True:
            continue

        taxon = observation.get("taxon") or {}
        user = observation.get("user") or {}
        observation_id = observation.get("id")
        if not isinstance(observation_id, int):
            continue

        eligible_photos = []
        for photo in observation.get("photos") or []:
            license_code = str(photo.get("license_code") or "").lower()
            photo_id = photo.get("id")
            photo_url = photo.get("url")
            if license_code not in allowed_licenses:
                continue
            if not isinstance(photo_id, int) or not isinstance(photo_url, str):
                continue
            dimensions = photo.get("original_dimensions") or {}
            width = dimensions.get("width")
            height = dimensions.get("height")
            if isinstance(width, int) and isinstance(height, int) and min(width, height) < 600:
                continue
            eligible_photos.append(photo)

        if not eligible_photos:
            continue

        # One photo per observation limits burst and individual-plant leakage.
        # iNaturalist sends original_dimensions with null width/height for some
        # photos, so .get(key, 0) returns None rather than the default: the key
        # is present. The filter above keeps those deliberately -- an unknown
        # dimension is not grounds to reject -- so they must sort as area 0
        # instead of raising, and lose to any photo whose size is known.
        photo = max(
            eligible_photos,
            key=lambda item: (
                ((item.get("original_dimensions") or {}).get("width") or 0)
                * ((item.get("original_dimensions") or {}).get("height") or 0)
            ),
        )
        photo_id = int(photo["id"])
        if photo_id in seen_photos:
            continue
        seen_photos.add(photo_id)

        dimensions = photo.get("original_dimensions") or {}
        creator = str(photo.get("attribution_name") or user.get("name") or user.get("login") or "")
        candidates.append(
            Candidate(
                observation_id=observation_id,
                photo_id=photo_id,
                image_url=original_photo_url(str(photo["url"])),
                source_page=f"https://www.inaturalist.org/observations/{observation_id}",
                license_code=str(photo["license_code"]).lower(),
                license_url=allowed_licenses[str(photo["license_code"]).lower()],
                attribution=str(photo.get("attribution") or ""),
                creator=creator,
                taxon_id=taxon.get("id") if isinstance(taxon.get("id"), int) else None,
                taxon_name=str(taxon.get("name") or observation.get("species_guess") or "Drosera"),
                common_name=taxon.get("preferred_common_name"),
                observer_login=str(user.get("login") or "unknown"),
                observed_on=observation.get("observed_on"),
                source_width=dimensions.get("width") if isinstance(dimensions.get("width"), int) else None,
                source_height=dimensions.get("height") if isinstance(dimensions.get("height"), int) else None,
            )
        )
    return candidates


def select_diverse(
    candidates: Iterable[Candidate],
    limit: int,
    max_per_species: int = 40,
    max_per_observer: int = 15,
    seed: int = 20260916,
) -> list[Candidate]:
    """Deterministically shuffle candidates and enforce simple diversity caps."""
    pool = list(candidates)
    Random(seed).shuffle(pool)
    species_counts: Counter[str] = Counter()
    observer_counts: Counter[str] = Counter()
    selected: list[Candidate] = []
    for candidate in pool:
        if species_counts[candidate.taxon_name] >= max_per_species:
            continue
        if observer_counts[candidate.observer_login] >= max_per_observer:
            continue
        selected.append(candidate)
        species_counts[candidate.taxon_name] += 1
        observer_counts[candidate.observer_login] += 1
        if len(selected) >= limit:
            break
    return selected


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _download_file(url: str, destination: Path, timeout: float = 90.0) -> tuple[str, int]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
    digest = sha256()
    byte_count = 0
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            byte_count += len(chunk)
            if byte_count > 30 * 1024 * 1024:
                raise ValueError("image exceeds the 30 MB safety limit")
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest(), byte_count


def _download_with_retries(
    url: str,
    destination: Path,
    attempts: int = 3,
) -> tuple[str, int]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _download_file(url, destination)
        except (TimeoutError, URLError) as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(float(attempt))
    assert last_error is not None
    raise last_error


def _image_details(path: Path) -> tuple[int, int, str, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or "unknown"
        grayscale = image.convert("L").resize((9, 8))
        pixels = list(grayscale.getdata())
        bits = [pixels[row * 9 + col] > pixels[row * 9 + col + 1] for row in range(8) for col in range(8)]
        dhash = sum((1 << index) for index, bit in enumerate(bits) if bit)
    return width, height, image_format, f"{dhash:016x}"


def download_candidates(
    candidates: Iterable[Candidate],
    output_dir: Path,
    target_count: int,
    delay_seconds: float = 0.05,
) -> dict[str, int]:
    """Download candidates, validate images, and append provenance records."""
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "metadata.jsonl"
    rejected_path = output_dir / "rejected.jsonl"

    existing = read_manifest(manifest_path)
    known_photo_ids = {int(row["photo_id"]) for row in existing}
    known_hashes = {str(row["sha256"]) for row in existing}
    accepted = len(existing)
    rejected = 0
    consecutive_network_failures = 0

    for candidate in candidates:
        if accepted >= target_count:
            break
        if candidate.photo_id in known_photo_ids:
            continue

        final_path = image_dir / f"inat_{candidate.photo_id}.jpg"
        temp_path = image_dir / f"inat_{candidate.photo_id}.part"
        try:
            file_hash, byte_count = _download_with_retries(candidate.image_url, temp_path)
            width, height, image_format, dhash = _image_details(temp_path)
            if min(width, height) < 600:
                raise ValueError(f"decoded image too small: {width}x{height}")
            if file_hash in known_hashes:
                raise ValueError("exact duplicate content")
            temp_path.replace(final_path)

            row = asdict(candidate)
            row.update(
                {
                    "local_path": final_path.as_posix(),
                    "sha256": file_hash,
                    "dhash64": dhash,
                    "bytes": byte_count,
                    "decoded_width": width,
                    "decoded_height": height,
                    "image_format": image_format,
                    "acquired_at": utc_now(),
                    "curation_status": "pending",
                    "change_notice": "Downloaded without intentional modification; binary annotation pending.",
                }
            )
            append_jsonl(manifest_path, row)
            known_photo_ids.add(candidate.photo_id)
            known_hashes.add(file_hash)
            accepted += 1
            consecutive_network_failures = 0
        except Exception as error:
            temp_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            append_jsonl(
                rejected_path,
                {
                    "photo_id": candidate.photo_id,
                    "observation_id": candidate.observation_id,
                    "source_page": candidate.source_page,
                    "reason": str(error),
                    "rejected_at": utc_now(),
                },
            )
            rejected += 1
            if isinstance(error, (TimeoutError, URLError)):
                consecutive_network_failures += 1
                if consecutive_network_failures >= 8:
                    break
            else:
                consecutive_network_failures = 0
        time.sleep(delay_seconds)

    return {"accepted": accepted, "rejected_this_run": rejected}


def fetch_candidate_pool(pages: int, per_page: int = 200, delay_seconds: float = 1.05) -> list[Candidate]:
    pool: list[Candidate] = []
    for page in range(1, pages + 1):
        payload = fetch_json(build_observation_url(page=page, per_page=per_page))
        pool.extend(extract_candidates(payload.get("results") or []))
        if page < pages:
            time.sleep(delay_seconds)
    return pool


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    yield from read_manifest(path)
