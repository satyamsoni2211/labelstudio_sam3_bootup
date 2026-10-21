---
layout: home

hero:
  name: Label Studio + SAM3
  text: Click. Segment. Done.
  tagline: SAM3 click and box segmentation inside Label Studio, running natively on your Mac with Apple GPU (MPS). No cloud GPU. No manual polygon tracing.
  image:
    src: /logo.svg
    alt: Label Studio + SAM3
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quick-start
    - theme: alt
      text: How It Works
      link: /guide/architecture

features:
  - title: Click or Box Prompts
    details: Click an object — or drag a box around it — and SAM3 returns an editable brush mask in about 0.2 seconds. Refine with extra clicks; Alt/Option+click removes.
  - title: Native on Apple Silicon
    details: The SAM3 backend runs on the Mac with PyTorch MPS. Label Studio runs in Docker and talks to it at host.docker.internal:9090. CPU and CUDA fallbacks included.
  - title: Real Exports
    details: Purpose-built exporters for COCO, YOLO detection and segmentation, and PNG masks (semantic + instance). They handle brush RLE correctly — the built-in exporters do not.
  - title: Fully Local
    details: Images, annotations, and model weights stay on your machine. The only download is the gated facebook/sam3 weights from Hugging Face on first start.
---

## What you get

You click on an object (or drag a box) and SAM3 returns an editable **brush mask**. Polygons are optional via `SAM3_OUTPUT`. Exports cover COCO, YOLO (boxes + segmentation), and PNG masks.

```
browser ──► Label Studio (Docker, :8080) ──http://host.docker.internal:9090──► SAM3 backend (native, MPS, :9090)
                                          ◄── backend downloads images through the LS API (token in .env) ──┘
```

## The five commands

```bash
cp .env.example .env   # then fill in HF_TOKEN (see Prerequisites)
make sync              # once: install backend dependencies (uv sync)
make up                # Label Studio on http://localhost:8080
make backend-start     # SAM3 backend in the background (log: logs/backend.log)
make setup             # create project "SAM3 annotation", connect the backend, import ./images/*
```

Then open [http://localhost:8080](http://localhost:8080), turn **Auto-Annotation ON**, pick the magic tool and a label, and click. Full walkthrough: [Quick Start](/guide/quick-start) → [Annotating with SAM3](/guide/annotating).

::: warning Before you start
`facebook/sam3` is a **gated** Hugging Face model. You must accept the licence and use a token with gated-repo read access, or the backend will fail with 401/403 on first start. See [Prerequisites](/guide/prerequisites).
:::
