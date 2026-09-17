"""Create and populate the local Label Studio project idempotently."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json


BASE_URL = "http://127.0.0.1:8080"
PROJECT_TITLE = "Sundew Segmentation v1"
ML_BACKEND_URL = "http://127.0.0.1:9090"
ML_BACKEND_TITLE = "MobileSAM CPU"


def request_json(path: str, token: str, method: str = "GET", body: Any | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
            return json.loads(payload) if payload else None
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Label Studio API {method} {path} failed: {error.code} {details}") from error


def main() -> None:
    credentials = json.loads(Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8"))
    token = credentials["token"]
    projects_payload = request_json("/api/projects?page_size=100", token)
    projects = projects_payload.get("results", projects_payload) if isinstance(projects_payload, dict) else projects_payload
    project = next((item for item in projects if item.get("title") == PROJECT_TITLE), None)

    if project is None:
        project = request_json(
            "/api/projects",
            token,
            method="POST",
            body={
                "title": PROJECT_TITLE,
                "description": "Human-reviewed binary masks for visible living sundew tissue. SAM outputs are proposals until accepted.",
                "label_config": Path("annotation/label-config.xml").read_text(encoding="utf-8"),
                "show_instruction": True,
                "show_skip_button": True,
                "enable_empty_annotation": False,
            },
        )
        print(f"Created project {project['id']}")
    else:
        print(f"Using existing project {project['id']}")

    project = request_json(f"/api/projects/{project['id']}", token)
    storage_path = str(Path("data/curated/images").resolve())
    storages = request_json(f"/api/storages/localfiles?project={project['id']}", token)
    storage = next((item for item in storages if item.get("path") == storage_path), None)
    if storage is None:
        storage = request_json(
            "/api/storages/localfiles",
            token,
            method="POST",
            body={
                "title": "Curated sundew images",
                "description": "Read-only image source. Tasks are imported from the provenance manifest.",
                "project": project["id"],
                "path": storage_path,
                "regex_filter": r".*\.(jpg|jpeg)$",
                "use_blob_urls": True,
            },
        )
        print(f"Registered local image storage {storage['id']}")
    else:
        print(f"Using existing local image storage {storage['id']}")

    if int(project.get("task_number") or 0) == 0:
        tasks = json.loads(Path("data/annotations/label-studio-tasks.json").read_text(encoding="utf-8"))
        result = request_json(f"/api/projects/{project['id']}/import", token, method="POST", body=tasks)
        print(f"Imported {len(tasks)} tasks: {result}")
    else:
        print(f"Project already contains {project['task_number']} tasks")

    backends = request_json(f"/api/ml?project={project['id']}", token)
    backend = next((item for item in backends if item.get("url") == ML_BACKEND_URL), None)
    if backend is None:
        backend = request_json(
            "/api/ml/",
            token,
            method="POST",
            body={
                "project": project["id"],
                "title": ML_BACKEND_TITLE,
                "url": ML_BACKEND_URL,
                "timeout": 120,
                "is_interactive": True,
            },
        )
        print(f"Connected MobileSAM backend {backend['id']}")
    else:
        print(f"Using existing MobileSAM backend {backend['id']}")

    final = request_json(f"/api/projects/{project['id']}", token)
    Path(".tools/label-studio-project.json").write_text(
        json.dumps(
            {
                "id": final["id"],
                "title": final["title"],
                "task_number": final["task_number"],
                "ml_backend_id": backend["id"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Ready: project {final['id']} with {final['task_number']} tasks")


if __name__ == "__main__":
    main()
