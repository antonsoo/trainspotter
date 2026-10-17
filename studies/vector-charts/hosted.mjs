// Follow the real public report, disconnect, download, and reopen its vector.
// node studies/vector-charts/hosted.mjs SITE_DIR OUTPUT_DIR [BASE_URL]
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const [site, outputArg, base = 'https://antonsoo.github.io/trainspotter/'] = process.argv.slice(2);
assert.ok(site && outputArg, 'SITE_DIR OUTPUT_DIR [BASE_URL]');
const output = resolve(outputArg); await mkdir(output, { recursive: true });
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const reports = [
  { file: 'divergence.html', label: 'steps', count: 5 },
  { file: 'wandb-logging-only.html', label: 'logging steps', count: 2 },
];
const expected = new Map();
for (const report of reports) {
  const html = await readFile(resolve(site, report.file));
  const payloads = [...html.toString().matchAll(/href="data:image\/svg\+xml;base64,([^"]+)"/g)].map(match => Buffer.from(match[1], 'base64'));
  assert.equal(payloads.length, report.count);
  expected.set(report.file, { html, payloads });
}
const views = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch();
  try {
    for (const width of [1440, 375]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 } });
      try {
        const page = await context.newPage(), errors = [], csp = [], external = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => {
          if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== new URL(base).origin) external.push(request.url());
        });
        await page.addInitScript(() => {
          window.svgCsp = [];
          document.addEventListener('securitypolicyviolation', event => window.svgCsp.push(event.violatedDirective));
        });
        const rows = [];
        for (const report of reports) {
          await context.setOffline(false);
          const response = await page.goto(new URL(report.file, base).href);
          assert.ok(response.ok());
          const body = await response.body(), selected = expected.get(report.file);
          assert.equal(hash(body), hash(selected.html), 'Canonical report bytes must match the deployed clean build');
          const links = page.locator('.chart-download'); assert.equal(await links.count(), report.count);
          assert.deepEqual((await new AxeBuilder({ page }).analyze()).violations, []);
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
          await context.setOffline(true);
          const pending = page.waitForEvent('download'); await links.first().focus(); await page.keyboard.press('Enter');
          const download = await pending; assert.equal(await download.failure(), null);
          const file = resolve(output, `${engine}-${width}-${report.file}.svg`); await download.saveAs(file);
          const actual = await readFile(file); assert.deepEqual(actual, selected.payloads[0]);
          const standalone = await context.newPage();
          try {
            standalone.on('pageerror', error => errors.push(error.message));
            await standalone.goto(pathToFileURL(file).href);
            const info = await standalone.evaluate(() => {
              const root = document.documentElement, metadata = JSON.parse(root.querySelector('metadata').textContent);
              return { namespace: root.namespaceURI, metadata, axisFill: getComputedStyle(root.querySelector('.axis-x')).fill,
                rootWidth: root.getBoundingClientRect().width, text: root.textContent,
                scripts: root.querySelectorAll('script, foreignObject, image').length };
            });
            assert.equal(info.namespace, 'http://www.w3.org/2000/svg'); assert.equal(info.metadata.axisLabel, report.label);
            assert.equal(info.axisFill, 'rgb(110, 124, 135)'); assert.ok(info.rootWidth <= width + .1); assert.equal(info.scripts, 0);
            assert.ok(info.text.includes('Horizontal axis: ' + report.label));
            await standalone.locator('svg').first().screenshot({ path: resolve(output, `${engine}-${width}-${report.file}.png`) });
            rows.push({ report: report.file, reportSha256: hash(body), availableDownloads: report.count,
              offlineBeforeDownload: true, savedBytes: actual.length, savedSha256: hash(actual),
              byteEqualCleanBuild: true, metadata: info.metadata, axisFill: info.axisFill, fitsViewport: true });
          } finally { await standalone.close(); }
          csp.push(...await page.evaluate(() => window.svgCsp));
        }
        assert.deepEqual(errors, []); assert.deepEqual(csp, []); assert.deepEqual(external, []);
        views.push({ engine, width, rows, errors, csp, external, axeViolations: [] });
        console.log(engine, width, rows.length, 'actual offline hosted downloads');
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(resolve(output, 'hosted-browser.json'), JSON.stringify({ base, checkedAt: new Date().toISOString(),
  scope: 'Two retained native reports in four browser contexts. Exact clean canonical HTML, real offline downloads, and independently reopened SVG files. No physical-print or detector-accuracy claim.', views }, null, 2) + '\n');
