// Open the actual CLI reports offline; retain original chart-column provenance.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const [directory, capturePath, output] = process.argv.slice(2);
assert.ok(directory && capturePath && output, 'REPORT_DIRECTORY CAPTURE_JSON OUTPUT_DIRECTORY');
const replay = JSON.parse(await readFile(resolve(directory, 'replay.json')));
const capture = JSON.parse(await readFile(capturePath));
assert.equal(replay.expectation, 'after');
await mkdir(output, { recursive: true });
const results = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch();
  try {
    for (const width of [1440, 375]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, offline: true });
      try {
        const page = await context.newPage();
        const errors = [], requests = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
        const reports = [];
        for (const item of replay.cases) {
          const file = resolve(directory, item.name + '.html');
          const digest = createHash('sha256').update(await readFile(file)).digest('hex');
          assert.equal(digest, item.htmlSha256);
          await page.goto(pathToFileURL(file).href);
          assert.deepEqual(await page.locator('.finding h3').allTextContents(), item.findings.map(f => f.title));
          assert.equal(await page.locator('script').count(), 0);
          const details = page.locator('.input-evidence');
          await details.locator('summary').focus();
          await page.keyboard.press('Enter');
          assert.equal(await details.getAttribute('open'), '');
          const original = capture.cases.find(c => c.name === item.name);
          if (original.kind === 'chart') {
            const cells = await details.locator('tbody td').allTextContents();
            for (const metric of ['loss', 'eval_loss', 'learning_rate']) {
              assert.ok(cells.includes(original.label + ' - ' + metric), item.name + ': ' + metric);
            }
          }
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
          const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
          assert.deepEqual(audit.violations, []);
          let screenshot = null;
          if (item.name === 'chart-multiline') {
            screenshot = `${engine}-${width}.png`;
            await page.screenshot({ path: resolve(output, screenshot), fullPage: true });
          }
          reports.push({ name: item.name, htmlSha256: digest, screenshot });
        }
        assert.deepEqual(errors, []);
        assert.deepEqual(requests, []);
        results.push({ engine, version: browser.version(), width, offlineBeforeNavigation: true,
          reports, errors, requests, axeViolations: 0 });
        console.log(JSON.stringify({ engine, width, reports: reports.length }));
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(resolve(output, 'browser.json'), JSON.stringify({ results }, null, 2) + '\n');
