import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const [directory, receiptPath, mode, screenshotDirectory, replayPath] = process.argv.slice(2);
assert.ok(['baseline', 'verify'].includes(mode));
const root = resolve(directory);
const replay = JSON.parse(await readFile(replayPath, 'utf8'));
const cases = ['captured-truncated', 'local-duplicate'];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
await mkdir(screenshotDirectory, { recursive: true });
const results = [];
for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch();
  try {
    for (const width of [1440, 375]) for (const name of cases) {
      const reference = replay.records.find(r => r.name === name);
      assert.ok(reference);
      const asset = await readFile(resolve(root, name + '.html'));
      assert.equal(sha(asset), reference.htmlSha256);
      const context = await browser.newContext({ viewport: { width, height: 950 }, colorScheme: 'dark', offline: true });
      try {
        const page = await context.newPage();
        const errors = [], httpRequests = [];
        page.on('pageerror', e => errors.push(e.message));
        page.on('request', r => { if (/^https?:/.test(r.url())) httpRequests.push(r.url()); });
        await page.addInitScript(() => {
          window.policyViolations = [];
          document.addEventListener('securitypolicyviolation', e => window.policyViolations.push(e.violatedDirective));
        });
        await page.goto(pathToFileURL(resolve(root, name + '.html')).href);
        assert.deepEqual(await page.locator('.finding h3').allTextContents(), reference.report.findings.map(f => f.title));
        assert.match(await page.locator('#input-heading').innerText(), /Input incomplete/);
        assert.equal(await page.locator('script').count(), 0);
        const lines = await page.locator('svg.chart path[stroke-width="1.75"]').count();
        assert.equal(lines, mode === 'baseline' ? 1 : name === 'captured-truncated' ? 3 : 2);
        let download = null;
        const details = page.locator('.unreadable-evidence');
        if (mode === 'verify') {
          await details.locator('summary').focus();
          await page.keyboard.press('Enter');
          assert.equal(await details.getAttribute('open'), '');
          assert.ok(await page.locator('.marker-unreadable').count() > 0);
          const link = details.getByRole('link', { name: 'Download all unreadable positions' });
          await link.focus();
          const pending = page.waitForEvent('download');
          await page.keyboard.press('Enter');
          const file = await pending;
          assert.equal(file.suggestedFilename(), 'trainspotter-unreadable-values.json');
          const bytes = await readFile(await file.path());
          const data = JSON.parse(bytes);
          assert.equal(data.source_path, name + '.jsonl');
          assert.deepEqual(data.unreadable_steps, { 'train/loss': reference.expectedUnreadableSteps });
          assert.equal(await link.evaluate(node => document.activeElement === node), true);
          download = { bytes: bytes.length, sha256: sha(bytes), data };
          assert.equal(await page.locator('.finding h3').filter({ hasText: 'Non-finite' }).count(), 0);
        } else assert.equal(await details.count(), 0);
        const violations = (await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice']).analyze()).violations;
        assert.deepEqual(violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) })), []);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        assert.deepEqual(errors, []);
        assert.deepEqual(httpRequests, []);
        const policyViolations = await page.evaluate(() => window.policyViolations);
        assert.deepEqual(policyViolations, []);
        const screenshot = `${engine}-${width}-${name}.png`;
        await page.screenshot({ path: resolve(screenshotDirectory, screenshot), fullPage: true });
        results.push({ engine, browserVersion: browser.version(), width, name,
          htmlBytes: asset.length, htmlSha256: sha(asset), openedFromFileWhileOffline: true,
          plottedSegments: lines, titles: reference.report.findings.map(f => f.title),
          unreadablePositions: mode === 'baseline' ? [] : reference.expectedUnreadableSteps,
          download, errors, httpRequests, policyViolations, axeViolations: 0, horizontalOverflow: false,
          screenshot, screenshotSha256: sha(await readFile(resolve(screenshotDirectory, screenshot))) });
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(receiptPath, JSON.stringify({ mode, results }, null, 2) + '\n');
console.log(`${results.length} actual CLI HTML workflows passed as offline local files (${mode}).`);
