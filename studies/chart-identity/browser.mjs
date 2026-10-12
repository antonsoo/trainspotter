import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const [directory, replayPath, receiptPath, screenshotDirectory] = process.argv.slice(2);
const replay = JSON.parse(await readFile(replayPath, 'utf8'));
assert.equal(replay.expectation, 'after');
await mkdir(screenshotDirectory, { recursive: true });
const results = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch();
  try {
    for (const width of [1440, 375]) for (const reference of replay.records) {
      const file = resolve(directory, reference.name + '.html');
      const bytes = await readFile(file);
      const digest = createHash('sha256').update(bytes).digest('hex');
      assert.equal(digest, reference.htmlSha256);
      const context = await browser.newContext({ viewport: { width, height: 950 },
        colorScheme: 'dark', offline: true });
      try {
        const page = await context.newPage();
        const errors = [], httpRequests = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => { if (/^https?:/.test(request.url())) httpRequests.push(request.url()); });
        await page.addInitScript(() => {
          window.policyViolations = [];
          document.addEventListener('securitypolicyviolation', event => window.policyViolations.push(event.violatedDirective));
        });
        await page.goto(pathToFileURL(file).href);
        assert.deepEqual(await page.locator('.finding h3').allTextContents(), ['Possible overfitting']);
        assert.equal(await page.locator('script').count(), 0);
        const provenance = reference.report.run.chart_identity;
        if (provenance.interpretation === 'declared_single_run') {
          assert.match(await page.locator('.input-notes').innerText(), /CSV alone cannot verify this declaration/);
        }
        const details = page.locator('.input-evidence');
        await details.locator('summary').focus();
        await page.keyboard.press('Enter');
        assert.equal(await details.getAttribute('open'), '');
        const cells = await details.locator('tbody td').allTextContents();
        for (const label of Object.values(provenance.columns)) assert.ok(cells.includes(label), label);
        assert.deepEqual(errors, []);
        assert.deepEqual(httpRequests, []);
        assert.deepEqual(await page.evaluate(() => window.policyViolations), []);
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
        assert.deepEqual(audit.violations, []);
        let screenshot = null;
        if (reference.name === 'ambiguous') {
          screenshot = `${engine}-${width}.png`;
          await page.screenshot({ path: resolve(screenshotDirectory, screenshot), fullPage: true });
        }
        results.push({ engine, version: browser.version(), width, name: reference.name,
          htmlSha256: digest, errors, httpRequests, axeViolations: 0, screenshot });
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(receiptPath, JSON.stringify({ offline: true, results }, null, 2) + '\n');
console.log(JSON.stringify({ browserChecks: results.length, screenshots: results.filter(r => r.screenshot).length }));
