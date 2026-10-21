# Deploying These Docs

This site is a VitePress static site (`docs/`, config in `docs/.vitepress/config.ts`), built with the scripts in the repository's root `package.json`.

## Local development

```bash
npm install        # once
npm run docs:dev   # dev server with hot reload
npm run docs:build # production build → docs/.vitepress/dist
npm run docs:preview # preview the production build
```

Requires Node.js ≥ 18.

## Deploy to Vercel

The repo ships a `vercel.json` so Vercel needs no dashboard configuration:

| Setting | Value |
|---|---|
| Framework preset | VitePress (auto-detected) |
| Build command | `npm run docs:build` |
| Output directory | `docs/.vitepress/dist` |
| Install command | `npm install` |

### Via CLI

```bash
npm i -g vercel   # once
vercel            # preview deployment (from the repo root)
vercel --prod     # production deployment
```

### Via GitHub import

1. Push the repository to GitHub.
2. In the Vercel dashboard, **Add New → Project** and import the repo.
3. Vercel reads `vercel.json` — just confirm and deploy. Every push then redeploys automatically.

The site is fully static (no server code, no environment variables), so no further configuration is needed. Light/dark mode follows the reader's system preference with a manual toggle in the nav bar.
