// Check local, downloaded, or hosted examples against their actual input bytes.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const root = new URL('../../', import.meta.url);
const base = process.env.TRAINSPOTTER_SITE_URL || 'http://127.0.0.1:4310/';
const output = process.env.TRAINSPOTTER_SITE_EVIDENCE;
const screenshots = process.env.TRAINSPOTTER_SITE_SCREENSHOTS;
const cases = [
  { slug: 'scheduler-resume', source: 'examples/schedulers/captured/linear-resume-reset.csv',
    heading: 'LR rose after decay', evidence: ['low_step=201', 'high_step=239', 'confirmed_step=202'],
    context: /scheduler counter is deliberately reset/, absent: /LR discontinuity/ },
  { slug: 'evaluation-overlap', source: 'examples/evaluation/captured/late-training.csv',
    heading: 'Evaluation loss rising', evidence: ['training_status=insufficient_overlap', 'best_step=7'],
    context: /evaluation readings end at step 60; training readings start at 100/, absent: /Possible overfitting/ },
  { slug: 'evaluation-cohorts', source: 'examples/evaluation/captured/fixed-cohort-switch.csv',
    heading: 'Evaluation values fluctuate', evidence: ['direction_changes=18', 'observations=20'],
    context: /step means evaluation pass, not optimizer update/ },
];
const observations = [];
if (screenshots) await mkdir(screenshots, { recursive: true });
for (const [name, launcher] of Object.entries({ chromium, firefox })) {
  const browser = await launcher.launch({ headless: true });
  try {
    for (const width of [1440, 320]) {
      const context = await browser.newContext({ viewport: { width, height: 960 }, colorScheme: 'dark' });
      const page = await context.newPage();
      const errors = [], requests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
      await page.addInitScript(() => {
        window.policyViolations = [];
        addEventListener('securitypolicyviolation', e => window.policyViolations.push(e.violatedDirective));
      });
      await page.goto(base);
      assert.equal(await page.locator('a.card').count(), 12);
      assert.deepEqual((await new AxeBuilder({ page }).analyze()).violations.map(v => v.id), []);
      for (const sample of cases) {
        await page.goto(base);
        const card = page.locator(`a.card[href="${sample.slug}.html"]`);
        const counts = await card.locator('.chip').allTextContents();
        await card.click();
        // Hosted responses can stream the context before the findings arrive.
        await page.waitForURL(new URL(`${sample.slug}.html`, base).href, { waitUntil: 'load' });
        const context = await page.locator('.sample-context').innerText();
        assert.match(context, sample.context);
        const finding = page.locator('article.finding').filter({ has: page.getByRole('heading', { name: sample.heading, exact: true }) });
        assert.equal(await finding.count(), 1);
        const details = await finding.locator('.evidence').innerText();
        for (const token of sample.evidence) assert.ok(details.includes(token), `${sample.slug}: ${token} missing in ${details}`);
        if (sample.absent) assert.doesNotMatch(await page.locator('.findings-log').innerText(), sample.absent);
        const actualCounts = await page.locator('.summary > .count').evaluateAll(nodes => nodes.map(n => `${n.querySelector('.n').textContent} ${n.querySelector('.l').textContent}`));
        assert.deepEqual(actualCounts, counts);
        const anchor = page.getByRole('link', { name: 'Download the input log', exact: true });
        await anchor.focus();
        assert.equal(await anchor.evaluate(n => document.activeElement === n), true);
        const pending = page.waitForEvent('download');
        await page.keyboard.press('Enter');
        const download = await pending;
        const chunks = [];
        for await (const chunk of await download.createReadStream()) chunks.push(chunk);
        const bytes = Buffer.concat(chunks);
        assert.deepEqual(bytes, await readFile(new URL(sample.source, root)));
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        const accessibility = (await new AxeBuilder({ page }).analyze()).violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) }));
        assert.deepEqual(accessibility, []);
        assert.deepEqual(await page.evaluate(() => window.policyViolations), []);
        if (screenshots && name === 'chromium' && sample.slug === 'evaluation-overlap' && width === 1440) await page.screenshot({ path: `${screenshots}/overlap-desktop.png`, fullPage: true });
        if (screenshots && name === 'firefox' && sample.slug === 'evaluation-cohorts' && width === 320) await page.screenshot({ path: `${screenshots}/cohorts-mobile.png`, fullPage: true });
        observations.push({ browser: name, width, sample: sample.slug, counts, source_sha256: createHash('sha256').update(bytes).digest('hex'), keyboard_download_matches_input: true, accessibility_violations: 0 });
      }
      assert.deepEqual(errors, []);
      const origin = new URL(base).origin;
      assert.ok(requests.every(url => new URL(url).origin === origin), requests.join('\n'));
      await context.close();
    }
  } finally { await browser.close(); }
}
if (output) await writeFile(output, JSON.stringify({ base, observations }, null, 2) + '\n');
console.log(`Verified ${observations.length} browser/sample/viewport workflows; input downloads match source bytes`);
