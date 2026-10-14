// Inspect actual exported reports offline; the renderer supplies every chart and count.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

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
        for (const label of ['latest-nonfinite', 'stale-nonfinite']) for (const phase of ['before', 'after']) {
          const page = await context.newPage(), errors = [], network = [];
          page.on('pageerror', error => errors.push(error.message));
          page.on('request', request => { if (/^https?:/.test(request.url())) network.push(request.url()); });
          const path = resolve(values[phase], label, 'report.html');
          await page.goto(pathToFileURL(path).href);
          const htmlSha256 = createHash('sha256').update(await readFile(path)).digest('hex');
          const details = page.locator('.input-evidence');
          await details.locator('summary').click();
          const row = details.locator('tbody tr').filter({ has: page.getByRole('rowheader', { name: 'train/loss', exact: true }) });
          const cells = await row.locator('td').allTextContents();
          assert.equal(cells[1], '40');
          const nonfinite = label === 'latest-nonfinite' ? phase === 'after' : phase === 'before';
          assert.equal(await page.getByRole('heading', { name: 'Non-finite training loss', exact: true }).count(), nonfinite ? 1 : 0);
          assert.equal(await page.locator('#input-heading').count(), 0);
          const axe = (await new AxeBuilder({ page }).analyze()).violations.map(row => row.id);
          assert.deepEqual(axe, []);
          const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
          assert.equal(horizontalOverflow, false);
          assert.deepEqual(errors, []); assert.deepEqual(network, []);
          const screenshot = label === 'latest-nonfinite' ? `${engine}-${width}-${phase}.png` : null;
          if (screenshot) await page.screenshot({ path: `${values.out}/${screenshot}`, fullPage: true });
          results.push({ label, engine, browserVersion: browser.version(), width, phase, offline: true,
            lossReadings: Number(cells[1]), nonfinite, horizontalOverflow, axeViolations: axe,
            pageErrors: errors, externalRequests: network, htmlSha256, screenshot,
            screenshotSha256: screenshot ? createHash('sha256').update(await readFile(`${values.out}/${screenshot}`)).digest('hex') : null });
          await page.close();
        }
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(`${values.out}/browser.json`, JSON.stringify(results, null, 2) + '\n');
console.log('Sixteen offline report controls passed in Chromium/Firefox at 1440/375px.');
