#!/usr/bin/env python
"""Create (or reuse) an LS project with label_config.xml and connect the SAM3 ML backend
(interactive mode ON; masks come back as BrushLabels). Optionally import images from a folder.

  python scripts/setup_project.py [--title "SAM3 annotation"] [--images images/] [--backend-url http://host.docker.internal:9090]
"""
import argparse
import sys
import time
from pathlib import Path

import lsutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="SAM3 annotation")
    ap.add_argument("--label-config", default=str(lsutil.ROOT / "label_config.xml"))
    ap.add_argument("--backend-url", default="http://host.docker.internal:9090",
                    help="URL of the ML backend AS SEEN FROM THE LABEL STUDIO CONTAINER")
    ap.add_argument("--images", help="folder of images to import as tasks (uploaded to LS)")
    ap.add_argument("--update-config", action="store_true", help="push label_config.xml to an existing project")
    a = ap.parse_args()

    s = lsutil.ls_session()
    for _ in range(60):
        try:
            if s.get(lsutil.LS_URL + "/health", timeout=3).ok:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        sys.exit("Label Studio is not reachable - run `make up` first")

    cfg = Path(a.label_config).read_text()
    r = s.get(f"{lsutil.LS_URL}/api/projects", params={"title": a.title, "page_size": 100})
    if r.status_code == 401:
        sys.exit("401 from Label Studio: legacy token auth disabled or token wrong. See README 'API token'.")
    r.raise_for_status()
    proj = next((p for p in r.json()["results"] if p["title"] == a.title), None)
    if proj is None:
        proj = s.post(f"{lsutil.LS_URL}/api/projects", json={"title": a.title, "label_config": cfg}).json()
        print("created project", proj["id"])
    else:
        print("reusing project", proj["id"])
        if a.update_config:
            r = s.patch(f"{lsutil.LS_URL}/api/projects/{proj['id']}", json={"label_config": cfg})
            if not r.ok:
                sys.exit(f"Could not update the label config: {r.status_code} {r.text[:400]}\n"
                         "(existing annotations using removed tags/labels can block this; delete them first)")
            print("label config updated")
    pid = proj["id"]

    mls = s.get(f"{lsutil.LS_URL}/api/ml", params={"project": pid}).json()
    ml = next((m for m in mls if m["url"] == a.backend_url), None)
    if ml is None:
        r = s.post(f"{lsutil.LS_URL}/api/ml", json={
            "project": pid, "url": a.backend_url, "title": "SAM3 (MPS)", "is_interactive": True,
            "description": "SAM3 click/box prompts -> brush mask"})
        if not r.ok:
            sys.exit(f"Could not connect ML backend: {r.status_code} {r.text}\n(Is `make backend` running?)")
        ml = r.json()
        print("connected ML backend", ml["id"])
    else:
        print("ML backend already connected", ml["id"])

    if a.images:
        files = [("file", (p.name, p.open("rb"))) for p in sorted(Path(a.images).iterdir())
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]
        if files:
            r = s.post(f"{lsutil.LS_URL}/api/projects/{pid}/import", files=files)
            r.raise_for_status()
            print("imported", r.json().get("task_count"), "tasks")
    print(f"Open {lsutil.LS_URL}/projects/{pid}/data")
    print("Reminder: in the labeling view turn ON the Auto-Annotation toggle (bottom bar), pick a label in the "
          "Click/Box prompt rows, then click / drag on the image.")


if __name__ == "__main__":
    main()
