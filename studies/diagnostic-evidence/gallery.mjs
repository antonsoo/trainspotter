// Check actual gallery navigation and embedded input downloads, then repeat offline.
// node studies/diagnostic-evidence/gallery.mjs URL SITE_DIR OUTPUT [SCREENSHOTS]
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import AxeBuilder from '@axe-core/playwright';
import { chromium, firefox } from 'playwright';

const root = fileURLToPath(new URL('../../', import.meta.url));
const [base, site, output, screenshots] = process.argv.slice(2);
assert.ok(base && site && output, 'URL SITE_DIR OUTPUT [SCREENSHOTS] are required');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const scenarios = [
  ['loss-cadence', 'slow-sparse', 0],
  ['gradient-concentration', 'unclipped-before', 1],
];
const evidence = [];
if (screenshots) await mkdir(screenshots, { recursive: true });
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch({ headless: true });
  try {
    for (const [layout, viewport] of [
      ['desktop', { width: 1440, height: 1000 }],
      ['mobile', { width: 375, height: 812 }],
    ]) {
      const context = await browser.newContext({ viewport, acceptDownloads: true });
      try {
        const page = await context.newPage();
        const errors = [], external = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => {
          if (/^https?:/.test(request.url()) && !request.url().startsWith(base)) {
            external.push(request.url());
          }
        });
        for (const [slug, input, count] of scenarios) {
          const source = await readFile(join(root, 'examples/diagnostic-evidence', `${input}.csv`));
          const report = await readFile(join(site, `${slug}.html`));
          for (const origin of ['served', 'file-offline']) {
            await context.setOffline(origin !== 'served');
            if (origin === 'served') {
              await page.goto(base);
              await page.locator(`a[href="${slug}.html"]`).focus();
              await Promise.all([page.waitForURL(`**/${slug}.html`), page.keyboard.press('Enter')]);
              const response = await fetch(new URL(`${slug}.html`, base.endsWith('/') ? base : `${base}/`));
              assert.equal(response.status, 200);
              assert.deepEqual(Buffer.from(await response.arrayBuffer()), report);
            } else {
              await page.goto(pathToFileURL(resolve(site, `${slug}.html`)).href);
            }
            const text = await page.locator('.findings-log').innerText();
            assert.equal(await page.locator('article.finding').count(), count);
            if (count) {
              assert.ok(text.includes('Gradient norms concentrated near their maximum'));
              assert.ok(text.includes('cannot establish whether clipping was used'));
              assert.ok(!text.includes('Gradient clipping likely saturated'));
            } else assert.ok(!text.includes('Suspiciously low loss early'));
            assert.ok((await page.locator('.sample-context').innerText()).includes('PyTorch'));
            const pending = page.waitForEvent('download');
            await page.getByRole('link', { name: 'Download the input log', exact: true }).click();
            const downloaded = await pending;
            const bytes = await readFile(await downloaded.path());
            assert.deepEqual(bytes, source);
            assert.equal(downloaded.suggestedFilename(), `${input}.csv`);
            assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
            assert.deepEqual(errors, []);
            assert.deepEqual(external, []);
            const axe = await new AxeBuilder({ page }).analyze();
            assert.deepEqual(axe.violations, []);
            let screenshot;
            if (screenshots && engine === 'chromium' && origin === 'file-offline') {
              screenshot = `${slug}-${layout}.png`;
              await page.screenshot({ path: join(screenshots, screenshot), fullPage: true });
            }
            evidence.push({ engine, browser_version: browser.version(), layout, origin, slug,
              expected_findings: count, input_sha256: digest(source), report_sha256: digest(report),
              downloaded_sha256: digest(bytes), screenshot, axe_violations: 0, errors: 0,
              external_requests: 0, keyboard_navigation: true });
          }
        }
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(output, JSON.stringify(evidence, null, 2) + '\n');
console.log(`${evidence.length} served/offline gallery workflows passed.`);
