import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
const root = fileURLToPath(new URL('..', import.meta.url));
const base = process.env.TRAINSPOTTER_SITE_URL || 'http://127.0.0.1:8789';
const installedReports = process.env.TRAINSPOTTER_INSTALLED_AXIS_REPORTS;
const digest = value => createHash('sha256').update(value).digest('hex');
const scenarios = [
  ['wandb-cadence', 'stable-training', 'Timing comparison', 'steps'],
  ['wandb-logging-only', 'stable-logging', 'Timing unavailable', 'logging steps'],
  ['wandb-slowdown', 'slow-training', 'Timing comparison', 'steps'],
];
const evidence = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch({ headless: true });
  try {
    for (const [layout, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 320, height: 780 }]]) {
      const context = await browser.newContext({ viewport, acceptDownloads: true });
      try {
        const page = await context.newPage();
        const errors = [], external = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => {
          if (/^https?:/.test(request.url()) && !request.url().startsWith(base)) external.push(request.url());
        });
        for (const [slug, name, heading, axis] of scenarios) {
          const input = await readFile(`${root}/examples/wandb-axes/${name}.csv`);
          for (const origin of ['served', 'file-offline']) {
            await context.setOffline(origin !== 'served');
            if (origin === 'served') {
              await page.goto(base);
              const card = page.locator(`a[href="${slug}.html"]`);
              await card.focus();
              await Promise.all([page.waitForURL(`**/${slug}.html`), page.keyboard.press('Enter')]);
            } else await page.goto(pathToFileURL(`${root}/site/${slug}.html`).href);
            if (origin === 'served') {
              const response = await fetch(`${base}/${slug}.html`);
              assert.equal(response.status, 200);
              assert.deepEqual(Buffer.from(await response.arrayBuffer()), await readFile(`${root}/site/${slug}.html`));
            }
            assert.equal(await page.locator('#timing-heading').textContent(), heading);
            assert.equal(await page.locator('.readouts dt').filter({ hasText: new RegExp(`^${axis}$`) }).count(), 1);
            assert.ok((await page.locator('.sample-context').innerText()).includes('SDK'));
            const pending = page.waitForEvent('download');
            await page.getByRole('link', { name: 'Download the input log', exact: true }).click();
            const download = await pending;
            const bytes = await readFile(await download.path());
            assert.deepEqual(bytes, input);
            assert.equal(download.suggestedFilename(), `${name}.csv`);
            await page.locator('.timing-evidence summary').focus();
            await page.keyboard.press('Enter');
            assert.equal(await page.locator('.timing-evidence details').getAttribute('open'), '');
            assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
            assert.deepEqual(errors, []);
            assert.deepEqual(external, []);
            evidence.push({ engine, browser_version: browser.version(), layout, origin, slug, axis,
              input_sha256: digest(input), downloaded_sha256: digest(bytes),
              report_sha256: digest(await readFile(`${root}/site/${slug}.html`)), keyboardNavigation: true, externalRequests: 0, errors: 0 });
          }
        }
        if (installedReports) {
          await context.setOffline(true);
          await page.goto(pathToFileURL(`${installedReports}/slow-custom-selected.html`).href);
          assert.ok((await page.locator('.input-notes').innerText()).includes('selected explicitly (--step-key)'));
          assert.ok((await page.locator('.readouts').innerText()).includes('optimizer/updates'));
          assert.equal(await page.locator('#timing-heading').textContent(), 'Timing comparison');
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
          assert.deepEqual(errors, []);
          assert.deepEqual(external, []);
          evidence.push({ engine, browser_version: browser.version(), layout, origin: 'installed-wheel-file-offline', scenario: 'explicit-custom-counter', errors: 0, externalRequests: 0 });
        }
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
if (process.env.TRAINSPOTTER_GALLERY_EVIDENCE) {
  await writeFile(process.env.TRAINSPOTTER_GALLERY_EVIDENCE, JSON.stringify(evidence, null, 2) + '\n');
}
console.log(`${evidence.length} gallery/installed-report workflows passed; every served/offline CSV download matches its input bytes.`);
