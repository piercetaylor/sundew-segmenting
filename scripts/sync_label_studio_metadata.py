"""Sync task morphology metadata and the tracked interface without touching reviews."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from sundew_segmentation.growth_forms import growth_form_for_taxon


BASE_URL = "http://127.0.0.1:8080"


def request_json(path: str, token: str, method: str = "GET", body: Any | None = None) -> Any:
    request = Request(
        f"{BASE_URL}{path}",
        data=None if body is None else json.dumps(body).encode("utf-8"),
        method=method,
        headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        payload = response.read()
        return json.loads(payload) if payload else None


def main() -> None:
    credentials = json.loads(Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8"))
    project_info = json.loads(Path(".tools/label-studio-project.json").read_text(encoding="utf-8"))
    token, project_id = credentials["token"], project_info["id"]
    payload = request_json(f"/api/tasks?project={project_id}&page_size=1000&fields=all", token)
    tasks = payload.get("tasks") or payload.get("results") or payload

    backup = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": request_json(f"/api/projects/{project_id}", token),
        "task_data": [{"id": task["id"], "data": task["data"]} for task in tasks],
    }
    backup_dir = Path(".tools/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"label-studio-metadata-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    backup_path.write_text(json.dumps(backup, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for task in tasks:
        data = dict(task["data"])
        growth_form = growth_form_for_taxon(str(data.get("species", "")))
        if growth_form == "unknown":
            raise SystemExit(f"No growth-form mapping for task {task['id']}: {data.get('species')}")
        data["growth_form"] = growth_form
        request_json(f"/api/tasks/{task['id']}/", token, method="PATCH", body={"data": data})
        counts[growth_form] = counts.get(growth_form, 0) + 1

    config = Path("annotation/label-config.xml").read_text(encoding="utf-8")
    request_json(f"/api/projects/{project_id}", token, method="PATCH", body={"label_config": config})
    print(json.dumps({"updated_tasks": len(tasks), "growth_form_counts": counts, "backup": str(backup_path)}, indent=2))


if __name__ == "__main__":
    main()
