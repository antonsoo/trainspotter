// Inspect a generated partial-input report without network access.
// Usage: node studies/record-lengths/browser.mjs REPORT.html [SCREENSHOT_DIRECTORY]
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';

const [report, screenshots] = process.argv.slice(2);
assert.ok(report);
if (screenshots) await mkdir(screenshots, { recursive: true });
const results = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch();
  try {
    for (const width of [1440, 375]) {
      const context = await browser.newContext({ viewport: { width, height: 960 }, offline: true });
      const page = await context.newPage();
      const errors = [];
      const network = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('request', request => { if (/^https?:/.test(request.url())) network.push(request.url()); });
      await page.goto(pathToFileURL(resolve(report)).href);
      assert.equal(await page.locator('#input-heading').innerText(), 'Input incomplete');
      assert.match(await page.locator('.input-notes').first().innerText(), /12 event-file bytes could not be decoded/);
      assert.match(await page.locator('body').innerText(), /No findings in the readable data. Input is incomplete./);
      assert.equal(await page.locator('svg.chart').count(), 1);
      const sources = page.locator('details').filter({ has: page.locator('th[scope="row"]').filter({ hasText: /^train\/loss$/ }) });
      await sources.locator('summary').click();
      const row = sources.locator('tr').filter({ has: page.getByRole('rowheader', { name: 'train/loss', exact: true }) });
      assert.match(await row.innerText(), /20\s+20/);
      if (width === 375) {
        for (const region of [sources.locator('.table-scroll'), page.locator('.chart-scroll')]) {
          await region.focus();
          await page.keyboard.press('ArrowRight');
          await page.waitForFunction(node => node.scrollLeft > 0, await region.elementHandle());
          await region.evaluate(node => { node.scrollLeft = 0; node.blur(); });
        }
      }
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors, []);
      assert.deepEqual(network, []);
      if (screenshots) await page.screenshot({ path: resolve(screenshots, `${engine}-${width}.png`), fullPage: true });
      results.push({ engine, version: browser.version(), width, offline: true, recorded: 20,
        analyzed: 20, unreadBytes: 12, inputComplete: false, pageErrors: errors, externalRequests: network });
      await context.close();
    }
  } finally {
    await browser.close();
  }
}
console.log(JSON.stringify(results, null, 2));
