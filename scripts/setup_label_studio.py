"""Create local-only credentials for the annotation workspace."""

from __future__ import annotations

from pathlib import Path
import json
import secrets


def main() -> None:
    destination = Path(".tools/label-studio-credentials.json")
    if destination.exists():
        print(f"Keeping existing credentials in {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    credentials = {
        "username": "sundew@local.invalid",
        "password": secrets.token_urlsafe(18),
        "token": secrets.token_hex(20),
    }
    destination.write_text(json.dumps(credentials, indent=2) + "\n", encoding="utf-8")
    print(f"Created local credentials in {destination}")


if __name__ == "__main__":
    main()
