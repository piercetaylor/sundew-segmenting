"""Rank remaining Label Studio tasks for split and morphology coverage."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def request_json(path: str, token: str) -> Any:
    request = Request("http://127.0.0.1:8080" + path, headers={"Authorization": f"Token {token}"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def annotation_quality(annotation: dict[str, Any]) -> str:
    for result in annotation.get("result", []):
        if result.get("type") == "choices" and result.get("from_name") == "quality":
            return next(iter(result.get("value", {}).get("choices", [])), "unrated")
    return "unrated"


def main() -> None:
    credentials = json.loads(Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8"))
    project = json.loads(Path(".tools/label-studio-project.json").read_text(encoding="utf-8"))
    payload = request_json(f"/api/tasks?project={project['id']}&page_size=1000&fields=all", credentials["token"])
    tasks = payload.get("tasks") or payload.get("results") or payload
    rows = []
    for task in tasks:
        annotations = [item for item in task.get("annotations", []) if not item.get("was_cancelled")]
        annotation = max(annotations, key=lambda item: item.get("updated_at", "")) if annotations else None
        quality = annotation_quality(annotation) if annotation else "unreviewed"
        growth_form = task["data"].get("growth_form", "unknown")
        split = task["data"]["split"]
        score = 100
        if quality == "complete":
            score = 1000
        else:
            score -= 40 if split in {"validation", "test"} else 0
            score -= 30 if growth_form != "rosette" else 0
            score -= 15 if quality == "ambiguous" else 0
            score += 20 if quality == "reject" else 0
        rows.append({
            "priority": score, "task_id": task["id"], "split": split,
            "growth_form": growth_form, "species": task["data"]["species"], "quality": quality,
            "reviewed": bool(annotation),
            "url": f"http://127.0.0.1:8080/projects/{project['id']}/data?task={task['id']}",
        })
    rows.sort(key=lambda row: (row["priority"], row["task_id"]))
    output = Path("data/reports/annotation-priority.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    remaining = [row for row in rows if row["quality"] != "complete"]
    print(json.dumps({
        "output": str(output), "remaining": len(remaining),
        "remaining_by_growth_form": dict(Counter(row["growth_form"] for row in remaining)),
        "next_20": [{key: row[key] for key in ("task_id", "split", "growth_form", "species", "quality")} for row in remaining[:20]],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
