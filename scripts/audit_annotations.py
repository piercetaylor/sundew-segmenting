"""Audit task-level Label Studio completion and annotation consistency."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def request_json(path: str, token: str) -> Any:
    request = Request(
        "http://127.0.0.1:8080" + path,
        headers={"Authorization": f"Token {token}"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def quality(annotation: dict[str, Any]) -> str:
    for result in annotation.get("result", []):
        if result.get("type") == "choices" and result.get("from_name") == "quality":
            return next(iter(result.get("value", {}).get("choices", [])), "unrated")
    return "unrated"


def main() -> None:
    credentials = json.loads(Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8"))
    project = json.loads(Path(".tools/label-studio-project.json").read_text(encoding="utf-8"))
    payload = request_json(
        f"/api/tasks?project={project['id']}&page_size=1000&fields=all",
        credentials["token"],
    )
    tasks = payload.get("tasks") or payload.get("results") or payload

    rows = []
    missing = []
    duplicates = []
    complete_without_mask = []
    for task in tasks:
        annotations = [item for item in task.get("annotations", []) if not item.get("was_cancelled")]
        if not annotations:
            missing.append(task["id"])
            current_quality = "unreviewed"
            annotation_id = None
        else:
            latest = max(annotations, key=lambda item: item.get("updated_at", ""))
            current_quality = quality(latest)
            annotation_id = latest["id"]
            if len(annotations) > 1:
                duplicates.append({"task_id": task["id"], "annotation_ids": [item["id"] for item in annotations]})
            has_mask = any(item.get("type") == "brushlabels" for item in latest.get("result", []))
            if current_quality == "complete" and not has_mask:
                complete_without_mask.append(task["id"])
        rows.append({
            "task_id": task["id"],
            "annotation_id": annotation_id,
            "split": task["data"]["split"],
            "species": task["data"]["species"],
            "growth_form": task["data"].get("growth_form", "unknown"),
            "quality": current_quality,
        })

    report = {
        "tasks": len(tasks),
        "annotated_tasks": len(tasks) - len(missing),
        "quality_by_task": dict(Counter(row["quality"] for row in rows)),
        "missing_task_ids": missing,
        "duplicate_annotations": duplicates,
        "complete_without_mask": complete_without_mask,
        "rows": rows,
    }
    output = Path("data/reports/annotation-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
