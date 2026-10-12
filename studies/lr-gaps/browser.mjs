import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const [directory, replayPath, receiptPath, screenshotDirectory] = process.argv.slice(2);
const replay = JSON.parse(await readFile(replayPath, 'utf8'));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
await mkdir(screenshotDirectory, { recursive: true });
const results = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch();
  try {
    for (const width of [1440, 375]) for (const name of ['unreadable', 'unknown-rise-phase']) {
      const reference = replay.records.find(row => row.name === name);
      assert.ok(reference);
      const file = resolve(directory, name + '.html');
      const bytes = await readFile(file);
      assert.equal(sha(bytes), reference.htmlSha256);
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
        assert.deepEqual(await page.locator('.finding h3').allTextContents(), reference.report.findings.map(f => f.title));
        assert.match(await page.locator('#input-heading').innerText(), /Input incomplete/);
        assert.equal(await page.locator('script').count(), 0);
        assert.ok(await page.locator('.marker-unreadable').count() > 0);
        if (replay.expectation === 'after') {
          const jumps = page.locator('.finding').filter({ has: page.locator('h3', { hasText: 'LR discontinuity' }) });
          assert.equal(await jumps.count(), reference.lrDiscontinuities.length);
          for (const text of await jumps.allTextContents()) assert.match(text, /no schedule is inferred across/);
        }
        const details = page.locator('.unreadable-evidence');
        await details.locator('summary').focus();
        await page.keyboard.press('Enter');
        assert.equal(await details.getAttribute('open'), '');
        const link = details.getByRole('link', { name: 'Download all unreadable positions' });
        await link.focus();
        const pending = page.waitForEvent('download');
        await page.keyboard.press('Enter');
        const download = await pending;
        assert.equal(download.suggestedFilename(), 'trainspotter-unreadable-values.json');
        const downloaded = await readFile(await download.path());
        const data = JSON.parse(downloaded);
        assert.equal(data.source_path, name + '.csv');
        assert.deepEqual(data.unreadable_steps, { lr: [name === 'unreadable' ? 10 : 40] });
        assert.equal(await link.evaluate(node => document.activeElement === node), true);
        const violations = (await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice']).analyze()).violations;
        assert.deepEqual(violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) })), []);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        const policyViolations = await page.evaluate(() => window.policyViolations);
        assert.deepEqual(errors, []);
        assert.deepEqual(httpRequests, []);
        assert.deepEqual(policyViolations, []);
        let screenshot = null;
        if (name === 'unreadable') {
          const filename = `${engine}-${width}.png`;
          await page.screenshot({ path: resolve(screenshotDirectory, filename), fullPage: true });
          screenshot = { filename, sha256: sha(await readFile(resolve(screenshotDirectory, filename))) };
        }
        results.push({ engine, browserVersion: browser.version(), width, name,
          htmlBytes: bytes.length, htmlSha256: sha(bytes), openedFromFileWhileOffline: true,
          lrDiscontinuities: reference.lrDiscontinuities, download: { data, sha256: sha(downloaded) },
          errors, httpRequests, policyViolations, axeViolations: 0, horizontalOverflow: false, screenshot });
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(receiptPath, JSON.stringify({ expectation: replay.expectation, results }, null, 2) + '\n');
console.log(`${results.length} offline CLI HTML workflows passed (${replay.expectation}).`);
