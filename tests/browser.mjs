import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import AxeBuilder from '@axe-core/playwright';
import { chromium, firefox } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const scratch = await mkdtemp(join(tmpdir(), 'trainspotter-reports-'));
const screenshotDir = process.env.TRAINSPOTTER_SCREENSHOTS;
const evidence = [];
async function checkKeyboardScroll(page, region) {
  await region.focus();
  await page.keyboard.press('ArrowRight');
  await page.waitForFunction(node => node.scrollLeft > 0, await region.elementHandle());
  await region.evaluate(node => new Promise(resolve => {
    let previous = -1;
    let stable = 0;
    const frame = () => {
      stable = node.scrollLeft === previous ? stable + 1 : 0;
      previous = node.scrollLeft;
      if (stable >= 3) resolve();
      else requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }));
  await region.evaluate(node => { node.scrollTo({ left: 0, behavior: 'instant' }); });
}
try {
  const build = spawnSync('uv', ['run', '--locked', 'python', 'scripts/browser_fixtures.py', scratch], {
    cwd: root, encoding: 'utf8', timeout: 60_000,
  });
  assert.equal(build.status, 0, build.stderr);
  const reports = JSON.parse(build.stdout);
  if (screenshotDir) await mkdir(screenshotDir, { recursive: true });
  for (const [engine, launcher] of Object.entries({ chromium, firefox })) {
    const browser = await launcher.launch({ headless: true });
    try {
      for (const [layout, viewport, colorScheme] of [
        ['desktop', { width: 1440, height: 1000 }, 'dark'],
        ['mobile', { width: 320, height: 780 }, 'light'],
      ]) {
        const context = await browser.newContext({ viewport, colorScheme });
        try {
          for (const report of reports) {
            const page = await context.newPage();
            try {
              const errors = [];
              const requests = [];
              page.on('pageerror', error => errors.push(error.message));
              page.on('request', request => {
                if (/^https?:/.test(request.url())) requests.push(request.url());
              });
              await page.addInitScript(() => {
                window.policyViolations = [];
                document.addEventListener('securitypolicyviolation', event => {
                  window.policyViolations.push(`${event.violatedDirective}: ${event.blockedURI}`);
                });
              });
              await page.goto(pathToFileURL(join(scratch, report.file)).href);
              assert.equal(await page.locator('h1').innerText(), 'TRAINSPOTTER');
              assert.equal(await page.locator('script').count(), 0);
              assert.equal(await page.locator('main').count(), 1);
              assert.equal(await page.evaluate(() => Boolean(window.injected)), false);
              if (!report.complete) {
                assert.match(await page.locator('.input-notes[aria-labelledby="input-heading"]').innerText(), /Input incomplete/);
                assert.doesNotMatch(await page.locator('.findings-log').innerText(), /Clean run/);
              }
              const summary = page.locator('.input-evidence summary');
              await summary.focus();
              const focus = await summary.evaluate(node => ({
                focused: document.activeElement === node,
                outline: getComputedStyle(node).outlineStyle,
              }));
              assert.equal(focus.focused, true);
              assert.notEqual(focus.outline, 'none');
              await page.keyboard.press('Enter');
              assert.equal(await page.locator('.input-evidence').getAttribute('open'), '');
              assert.equal(await page.locator('.input-evidence tbody tr').count(), report.metrics);
              if (report.repeatedGroups) {
                const repeat = page.locator('.repeated-evidence');
                await repeat.locator('summary').focus();
                await page.keyboard.press('Enter');
                assert.equal(await repeat.getAttribute('open'), '');
                assert.equal(await repeat.locator('tbody tr').count(), Math.min(report.repeatedGroups, 100));
                const pending = page.waitForEvent('download');
                await repeat.getByRole('link', { name: 'Download all repeated-step evidence' }).click();
                const downloaded = JSON.parse(await readFile(await (await pending).path(), 'utf8'));
                const original = JSON.parse(await readFile(join(scratch, report.file.replace(/\.html$/, '.json')), 'utf8'));
                assert.deepEqual(downloaded.repeated_steps, original.run.repeated_steps);
                assert.equal(downloaded.repeated_steps.length, report.repeatedGroups);
              }
              if (layout === 'mobile') {
                for (const table of await page.locator('.table-scroll').all()) {
                  await checkKeyboardScroll(page, table);
                }
                const chart = page.locator('.chart-scroll').first();
                await checkKeyboardScroll(page, chart);
                await summary.focus();
              }
              assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
              const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice']).analyze();
              assert.deepEqual(audit.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), [],
                `${engine} ${layout} ${report.file}`);
              assert.deepEqual(errors, []);
              assert.deepEqual(requests, []);
              assert.deepEqual(await page.evaluate(() => window.policyViolations), []);
              const json = JSON.parse(await readFile(join(scratch, report.file.replace(/\.html$/, '.json')), 'utf8'));
              assert.equal(json.summary.input_complete, report.complete);
              if (screenshotDir && engine === 'chromium' && (
                (layout === 'desktop' && report.file.startsWith('divergence')) ||
                (layout === 'mobile' && report.file.startsWith('synthetic-partial')) ||
                report.file === 'conflicting.html' || report.file === 'single-step.html'
              )) {
                await page.screenshot({ path: join(screenshotDir, `${layout}-${report.file}.png`), fullPage: true });
              }
              evidence.push({ engine, layout, ...report, axeViolations: 0, externalRequests: 0, policyViolations: 0 });
            } finally {
              await page.close();
            }
          }
        } finally {
          await context.close();
        }
      }
    } finally {
      await browser.close();
    }
  }
  if (process.env.TRAINSPOTTER_BROWSER_EVIDENCE) {
    await writeFile(process.env.TRAINSPOTTER_BROWSER_EVIDENCE, JSON.stringify(evidence, null, 2) + '\n');
  }
  console.log(`${evidence.length} report workflows passed: Chromium + Firefox, desktop + 320px mobile, keyboard, axe, CSP, no network.`);
} finally {
  await rm(scratch, { recursive: true, force: true });
}
