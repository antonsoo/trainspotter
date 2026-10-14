// Inspect actual exported reports offline; the renderer supplies every chart and count.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, firefox } from 'playwright';

const { values } = parseArgs({ options: {
  before: { type: 'string' }, after: { type: 'string' }, out: { type: 'string' },
} });
if (!values.before || !values.after || !values.out) throw Error('Use --before HTML --after HTML --out DIRECTORY');
await mkdir(values.out, { recursive: true });
const results = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch({ headless: true });
  try {
    for (const width of [1440, 375]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme: 'dark', offline: true });
      try {
        for (const phase of ['before', 'after']) {
          const page = await context.newPage(), errors = [], network = [];
          page.on('pageerror', error => errors.push(error.message));
          page.on('request', request => { if (/^https?:/.test(request.url())) network.push(request.url()); });
          await page.goto(pathToFileURL(resolve(values[phase])).href);
          const details = page.locator('.input-evidence');
          await details.locator('summary').click();
          const row = details.locator('tbody tr').filter({ has: page.getByRole('rowheader', { name: 'train/loss', exact: true }) });
          const cells = await row.locator('td').allTextContents();
          assert.equal(cells[1], phase === 'before' ? '0' : '5');
          assert.equal(await page.locator('#input-heading').count(), phase === 'before' ? 1 : 0);
          if (phase === 'before') assert.equal(await page.locator('#input-heading').innerText(), 'Input incomplete');
          const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
          assert.equal(horizontalOverflow, false);
          assert.deepEqual(errors, []); assert.deepEqual(network, []);
          const screenshot = `${engine}-${width}-${phase}.png`;
          await page.screenshot({ path: `${values.out}/${screenshot}`, fullPage: true });
          results.push({ engine, browserVersion: browser.version(), width, phase, offline: true,
            lossReadings: Number(cells[1]), inputIncomplete: phase === 'before', horizontalOverflow,
            pageErrors: errors, externalRequests: network, screenshot,
            screenshotSha256: createHash('sha256').update(await readFile(`${values.out}/${screenshot}`)).digest('hex') });
          await page.close();
        }
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(`${values.out}/browser.json`, JSON.stringify(results, null, 2) + '\n');
console.log('Eight offline report controls passed in Chromium/Firefox at 1440/375px.');
