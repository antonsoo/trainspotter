// Inspect a generated million-observation report in real offline browsers.
// node tests/long-history.mjs /path/to/million.html /path/to/million.json
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import AxeBuilder from '@axe-core/playwright';
import { chromium, firefox } from 'playwright';

const [htmlPath, jsonPath] = process.argv.slice(2).map(value => resolve(value));
assert.ok(htmlPath && jsonPath, 'Pass the generated HTML and JSON report paths');
const html = await readFile(htmlPath);
const report = JSON.parse(await readFile(jsonPath, 'utf8'));
assert.deepEqual(report.findings.map(f => [f.detector, f.step_start, f.step_end]), [
  ['spikes', 737000, 737000], ['plateau', 200000, 600000],
]);
assert.ok(html.length < 100_000, 'The SVG must remain bounded by display resolution');
const screenshots = process.env.TRAINSPOTTER_LONG_SCREENSHOTS;
if (screenshots) await mkdir(screenshots, { recursive: true });
const evidence = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch({ headless: true });
  try {
    for (const [layout, viewport] of [
      ['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 320, height: 780 }],
    ]) {
      const context = await browser.newContext({ viewport });
      const page = await context.newPage();
      const errors = [], requests = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('request', r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
      const start = performance.now();
      await page.goto(pathToFileURL(htmlPath).href);
      for (const finding of report.findings) {
        const article = page.locator('article.finding').filter({
          has: page.getByRole('heading', { name: finding.title, exact: true }),
        });
        assert.equal(await article.count(), 1);
        assert.equal(await article.locator('.message').innerText(), finding.message);
      }
      const summary = page.locator('.input-evidence summary');
      await summary.focus();
      await page.keyboard.press('Enter');
      assert.equal(await page.locator('.input-evidence').getAttribute('open'), '');
      assert.match(await page.locator('.input-evidence').innerText(), /1,000,000|1000000/);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      if (layout === 'mobile') {
        const chart = page.locator('.chart-scroll').first();
        await chart.focus();
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction(el => el.scrollLeft > 0, await chart.elementHandle());
        await chart.evaluate(el => el.scrollTo({ left: 0, behavior: 'instant' }));
      }
      const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      assert.deepEqual(axe.violations, []);
      assert.deepEqual(errors, []);
      assert.deepEqual(requests, []);
      if (screenshots && engine === 'chromium') {
        await summary.focus();
        await page.keyboard.press('Enter');
        await page.screenshot({ path: join(screenshots, `${engine}-${layout}.png`), fullPage: true });
      }
      evidence.push({ engine, version: browser.version(), layout, viewport,
        workflow_ms: performance.now() - start, findings: report.findings.length,
        violations: axe.violations.length, external_requests: requests.length, page_errors: errors });
      await context.close();
    }
  } finally { await browser.close(); }
}
const result = { html: basename(htmlPath), html_bytes: html.length,
  html_sha256: createHash('sha256').update(html).digest('hex'), checks: evidence };
if (process.env.TRAINSPOTTER_LONG_BROWSER_EVIDENCE) {
  await writeFile(process.env.TRAINSPOTTER_LONG_BROWSER_EVIDENCE, `${JSON.stringify(result, null, 2)}\n`);
}
console.log(JSON.stringify(result, null, 2));
