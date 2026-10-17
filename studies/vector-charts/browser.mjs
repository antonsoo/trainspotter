// Download vectors from file:// HTML reports and open the saved SVGs independently.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const [captureArg, outputArg] = process.argv.slice(2);
assert.ok(captureArg && outputArg, 'CAPTURE_DIR OUTPUT_DIR');
const capture = resolve(captureArg), output = resolve(outputArg);
const manifest = JSON.parse(await readFile(resolve(capture, 'capture.json')));
await mkdir(output, { recursive: true });
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const results = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch();
  try {
    for (const width of [1440, 375]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 } });
      await context.setOffline(true);
      const rows = [], errors = [], requests = [], csp = [];
      try {
        const page = await context.newPage();
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
        await page.addInitScript(() => {
          window.svgCsp = [];
          document.addEventListener('securitypolicyviolation', event => window.svgCsp.push(event.violatedDirective));
        });
        for (const item of manifest.cases) {
          await page.goto(pathToFileURL(resolve(capture, item.file)).href);
          assert.equal(await page.locator('script').count(), 0);
          const links = page.locator('.chart-download');
          assert.equal(await links.count(), item.exports.length);
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
          assert.deepEqual((await new AxeBuilder({ page }).analyze()).violations, []);
          if (item.name === 'recorded-divergence') {
            await page.locator('.chart-download').first().focus();
            await page.screenshot({ path: resolve(output, `${engine}-${width}-report.png`), fullPage: true });
          }
          const downloads = [];
          for (let i = 0; i < item.exports.length; i++) {
            const pending = page.waitForEvent('download');
            const link = links.nth(i); await link.focus(); await page.keyboard.press('Enter');
            const download = await pending; assert.equal(await download.failure(), null);
            const filename = `${engine}-${width}-${item.name}-${i}.svg`;
            const path = resolve(output, filename); await download.saveAs(path);
            const bytes = await readFile(path); assert.equal(sha(bytes), item.exports[i].sha256);
            const standalone = await context.newPage();
            const standaloneErrors = [], standaloneRequests = [];
            standalone.on('pageerror', error => standaloneErrors.push(error.message));
            standalone.on('request', request => { if (/^https?:/.test(request.url())) standaloneRequests.push(request.url()); });
            try {
              await standalone.goto(pathToFileURL(path).href);
              const actual = await standalone.evaluate(() => {
                const root = document.documentElement;
                const info = JSON.parse(root.querySelector('metadata').textContent);
                const inner = root.querySelector('svg.chart');
                return { namespace: root.namespaceURI, metadata: info, hasChart: !!inner,
                  paths: inner.querySelectorAll('path').length,
                  geometry: { rootWidth: root.getBoundingClientRect().width, rootHeight: root.getBoundingClientRect().height,
                    innerWidth: inner.width.baseVal.value, innerHeight: inner.height.baseVal.value,
                    footerTop: root.querySelector('.export-footer').getBoundingClientRect().top,
                    axesBottom: Math.max(...[...inner.querySelectorAll('.axis-x')].map(node => node.getBoundingClientRect().bottom)),
                    viewportWidth: innerWidth },
                  axes: [...inner.querySelectorAll('.axis-y, .axis-x')].map(node => ({ text: node.textContent, fill: getComputedStyle(node).fill })),
                  grids: [...inner.querySelectorAll('.grid')].map(node => getComputedStyle(node).stroke),
                  valueLabels: [...inner.querySelectorAll('.off-scale-label')].map(node => ({ text: node.textContent,
                    left: node.getBoundingClientRect().left, right: node.getBoundingClientRect().right,
                    frameLeft: root.getBoundingClientRect().left, frameRight: root.getBoundingClientRect().right,
                    separators: [...node.querySelectorAll('tspan')].filter(span => span.textContent.includes('/')).map(span => getComputedStyle(span).fill) })),
                  scripts: root.querySelectorAll('script, image, foreignObject').length,
                  eventAttributes: [...root.querySelectorAll('*')].flatMap(node => [...node.attributes].filter(attr => /^on/i.test(attr.name)).map(attr => attr.name)),
                  unsafeExecuted: !!window.injected, visibleText: root.textContent };
              });
              assert.equal(actual.namespace, 'http://www.w3.org/2000/svg'); assert.equal(actual.hasChart, true);
              assert.deepEqual(actual.metadata, item.exports[i].metadata);
              assert.ok(actual.axes.every(axis => axis.fill === 'rgb(110, 124, 135)'));
              assert.ok(actual.grids.every(stroke => stroke === 'rgb(31, 40, 48)'));
              assert.ok(actual.geometry.rootWidth <= width + .1, JSON.stringify(actual.geometry));
              assert.ok(actual.geometry.footerTop > actual.geometry.axesBottom, JSON.stringify(actual.geometry));
              for (const label of actual.valueLabels) {
                assert.ok(label.left >= label.frameLeft - 1 && label.right <= label.frameRight + 1, JSON.stringify(label));
                assert.ok(label.separators.every(fill => fill === 'rgb(110, 124, 135)'));
              }
              assert.equal(actual.scripts, 0); assert.deepEqual(actual.eventAttributes, []); assert.equal(actual.unsafeExecuted, false);
              if (!item.inputComplete) assert.ok(actual.visibleText.includes('Input incomplete'));
              if (item.name === 'partial-peer') assert.ok(actual.visibleText.includes('No trajectory: train/loss'));
              if (item.name === 'recorded-logging-axis') assert.equal(actual.metadata.axisLabel, 'logging steps');
              if (item.name === 'recorded-divergence' && i === 0) {
                await standalone.locator('svg').first().screenshot({ path: resolve(output, `${engine}-${width}-standalone.png`) });
              }
              assert.deepEqual(standaloneErrors, []); assert.deepEqual(standaloneRequests, []);
              downloads.push({ filename, suggestedFilename: download.suggestedFilename(), bytes: bytes.length, sha256: sha(bytes),
                metadata: actual.metadata, geometry: actual.geometry, valueLabels: actual.valueLabels, axes: actual.axes, grids: actual.grids, svgNamespace: actual.namespace });
            } finally { await standalone.close(); }
          }
          csp.push(...await page.evaluate(() => window.svgCsp));
          rows.push({ name: item.name, downloads });
        }
        assert.deepEqual(errors, []); assert.deepEqual(requests, []); assert.deepEqual(csp, []);
        results.push({ engine, width, offlineBeforeOpening: true, rows, errors, requests, csp });
        console.log(JSON.stringify({ engine, width, reports: rows.length, downloads: rows.reduce((n, row) => n + row.downloads.length, 0) }));
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(resolve(output, 'browser.json'), JSON.stringify({ scope: 'Actual file:// report downloads, reopened as independent SVG files with the browser offline before navigation.', results }, null, 2) + '\n');
