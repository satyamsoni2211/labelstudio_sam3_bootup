SHELL := /bin/bash
PROJECT ?= 1
OUT ?= exports
PY = cd sam3_backend && uv run python
.PHONY: ui-test up down logs sync backend backend-start backend-stop backend-status backend-logs setup test \
        export-json export-coco export-yolo export-masks export-all

up:               ## start Label Studio (docker) on :8080
	docker compose up -d
down:             ## stop Label Studio (data stays in ./ls-data)
	docker compose down
logs:             ## follow Label Studio logs
	docker compose logs -f label-studio
sync:             ## install backend deps
	cd sam3_backend && uv sync
backend:          ## run SAM3 backend in the foreground (MPS) on :9090
	cd sam3_backend && PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python _wsgi.py
backend-start:    ## run SAM3 backend in the background (logs/backend.log, logs/backend.pid)
	scripts/backend_ctl.sh start
backend-stop:
	scripts/backend_ctl.sh stop
backend-status:
	scripts/backend_ctl.sh status
backend-logs:
	tail -f logs/backend.log
setup:            ## create LS project + connect the backend (+ import ./images)
	$(PY) ../scripts/setup_project.py --images ../images
test:             ## end-to-end smoke test (needs `make up` and a running backend)
	$(PY) ../scripts/smoke_test.py

# Exports: PROJECT=<id> (see the project URL), OUT=<dir>
export-json:      ## raw Label Studio JSON export
	mkdir -p $(OUT) && $(PY) -c "import sys; sys.path.insert(0,'../scripts'); import lsutil, json; r=lsutil.ls_session().get(f'{lsutil.LS_URL}/api/projects/$(PROJECT)/export', params={'exportType':'JSON','download_all_tasks':'false'}); r.raise_for_status(); open('../$(OUT)/project.json','w').write(json.dumps(r.json(), indent=1)); print('wrote $(OUT)/project.json')"
export-coco:      ## COCO polygons + boxes -> $(OUT)/coco/result.json
	$(PY) ../scripts/export_coco.py --project $(PROJECT) --out ../$(OUT)/coco/result.json
export-yolo:      ## YOLO detection (yolo_box) + segmentation (yolo_seg) -> $(OUT)/yolo
	$(PY) ../scripts/export_yolo.py --project $(PROJECT) --out ../$(OUT)/yolo
export-masks:     ## PNG masks (semantic + instance) -> $(OUT)/masks
	$(PY) ../scripts/export_masks.py --project $(PROJECT) --out ../$(OUT)/masks
export-all: export-json export-coco export-yolo export-masks

ui-test:          ## drive the real LS UI headlessly (Playwright): click/box prompts -> masks, submit, exports (PROJECT=6)
	$(PY) ../scripts/ui_test.py --project $(or $(UI_PROJECT),6)
