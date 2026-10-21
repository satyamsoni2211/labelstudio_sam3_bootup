import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Label Studio + SAM3',
  description: 'Click and box segmentation in Label Studio, powered by SAM3 on Apple Silicon.',
  lang: 'en-US',
  cleanUrls: true,
  ignoreDeadLinks: [/^http:\/\/localhost/],
  lastUpdated: true,
  appearance: true,
  head: [
    ['meta', { name: 'theme-color', content: '#ffffff' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'Label Studio + SAM3',
    nav: [
      { text: 'Guide', link: '/guide/overview', activeMatch: '/guide/' },
      { text: 'Reference', link: '/reference/configuration', activeMatch: '/reference/' },
      { text: 'Troubleshooting', link: '/help/troubleshooting' },
      {
        text: 'Links',
        items: [
          { text: 'Label Studio', link: 'https://labelstud.io' },
          { text: 'facebook/sam3 on Hugging Face', link: 'https://huggingface.co/facebook/sam3' },
          { text: 'uv', link: 'https://docs.astral.sh/uv/' },
        ],
      },
    ],
    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'Overview', link: '/guide/overview' },
          { text: 'Architecture', link: '/guide/architecture' },
        ],
      },
      {
        text: 'Getting Started',
        items: [
          { text: 'Prerequisites', link: '/guide/prerequisites' },
          { text: 'Quick Start', link: '/guide/quick-start' },
          { text: 'Installation in Detail', link: '/guide/installation' },
        ],
      },
      {
        text: 'Guides',
        items: [
          { text: 'Annotating with SAM3', link: '/guide/annotating' },
          { text: 'Editing Labels', link: '/guide/labels' },
          { text: 'Exporting Annotations', link: '/guide/exports' },
          { text: 'Testing', link: '/guide/testing' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'Configuration & Environment', link: '/reference/configuration' },
          { text: 'Commands (Makefile)', link: '/reference/commands' },
          { text: 'Backend Internals', link: '/reference/backend' },
          { text: 'Scripts Reference', link: '/reference/scripts' },
        ],
      },
      {
        text: 'Help',
        items: [
          { text: 'Troubleshooting & FAQ', link: '/help/troubleshooting' },
          { text: 'Deploying These Docs', link: '/help/deploying' },
        ],
      },
    ],
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/satyamsoni2211/labelstudio_sam3_bootup' },
    ],
    footer: {
      message: 'Label Studio + SAM3 — click/box segmentation on Apple Silicon.',
      copyright: 'Runs locally: Label Studio on :8080, SAM3 backend on :9090.',
    },
    editLink: {
      pattern: 'https://github.com/satyamsoni2211/labelstudio_sam3_bootup/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },
    outline: { level: [2, 3], label: 'On this page' },
  },
  markdown: {
    lineNumbers: true,
  },
})
