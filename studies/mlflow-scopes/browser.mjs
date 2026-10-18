// URL OUTPUT INSTALLED_PYTHON: actual sample, native download and standalone SVG.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const [url, directory, python] = process.argv.slice(2);
assert(url && directory && python, 'URL OUTPUT INSTALLED_PYTHON');
const output = resolve(directory);
await mkdir(output, { recursive: true });
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const oracle = JSON.parse(await readFile(resolve(root, 'studies/mlflow-scopes/native/oracle.json'), 'utf8'));
const native = oracle.cases.find(item => item.name === 'mixed-dataset');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const records = [];
for (const [engine, launch] of Object.entries({ chromium, firefox })) {
  const browser = await launch.launch();
  try {
    for (const width of [1440, 375]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme: width === 375 ? 'light' : 'dark' });
      try {
        const page = await context.newPage(), errors = [], requests = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
        await page.addInitScript(() => {
          window.scopeDiagnostics = { csp: [], rejections: [] };
          document.addEventListener('securitypolicyviolation', event => window.scopeDiagnostics.csp.push(event.violatedDirective));
          window.addEventListener('unhandledrejection', event => window.scopeDiagnostics.rejections.push(String(event.reason)));
        });
        await page.goto(url);
        if (url.endsWith('/')) {
          await Promise.all([
            page.waitForURL(new URL('mlflow-model-scopes.html', url).href),
            page.getByRole('link', { name: /A change of metric scope is not training divergence/ }).click(),
          ]);
        }
        await context.setOffline(true);
        assert.equal(await page.locator('script').count(), 0);
        assert.match(await page.locator('.input-notes').first().innerText(), /Input incomplete/);
        assert.match(await page.locator('.input-notes').first().innerText(), /cannot recover its source scopes/);
        assert.equal(await page.locator('.finding h3').count(), 0);
        assert.match(await page.locator('.findings-log').innerText(), /Input is incomplete/);
        assert.doesNotMatch(await page.locator('.findings-log').innerText(), /Clean run|Diverged/);
        const disclosure = page.locator('.input-evidence summary');
        await disclosure.focus(); await page.keyboard.press('Enter');
        const sources = await page.locator('.input-evidence').innerText();
        assert.match(sources, /loss \[MLflow run aggregate\]/);
        assert.match(sources, /40/);
        assert.doesNotMatch(sources, /train\/loss/);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        assert.deepEqual((await new AxeBuilder({ page }).analyze()).violations, []);
        const stem = `${engine}-${width}`;
        await page.screenshot({ path: resolve(output, `${stem}.png`), fullPage: true });
        let pending = page.waitForEvent('download');
        await page.getByRole('link', { name: 'Download the input log' }).click();
        const database = await pending;
        assert.equal(await database.failure(), null);
        const dbPath = resolve(output, `${stem}.sqlite`);
        await database.saveAs(dbPath);
        assert.equal(sha(await readFile(dbPath)), oracle.database.sha256);
        const replay = spawnSync(python, ['-m', 'trainspotter.cli', 'analyze', dbPath,
          '--run', native.run_id, '--output', 'json', '--fail-on', 'error'], { encoding: 'utf8', timeout: 30000 });
        assert.equal(replay.status, 2, replay.stderr);
        const report = JSON.parse(replay.stdout);
        assert.equal(report.run.input_complete, false);
        assert.deepEqual(report.findings, []);
        assert.equal(report.run.metrics['loss [MLflow run aggregate]'].count, 40);
        assert.equal(report.run.mlflow.metric_scopes.recorded_rows, 20);
        pending = page.waitForEvent('download');
        await page.getByRole('link', { name: /^Download .* chart as SVG$/ }).click();
        const vector = await pending;
        assert.equal(await vector.failure(), null);
        const svgPath = resolve(output, `${stem}.svg`);
        await vector.saveAs(svgPath);
        const standalone = await context.newPage();
        let metadata;
        try {
          await standalone.goto(pathToFileURL(svgPath).href);
          metadata = JSON.parse(await standalone.locator('metadata').textContent());
          assert.equal(metadata.inputComplete, false);
          assert.deepEqual(metadata.plottedMetrics, ['loss [MLflow run aggregate]']);
          assert.equal(metadata.axisKind, 'logging_step');
          assert(metadata.inputNotes.some(note => note.includes('cannot recover its source scopes')));
          assert(await standalone.locator('svg.chart path').count() > 0);
        } finally { await standalone.close(); }
        assert.deepEqual(errors, []);
        const diagnostics = await page.evaluate(() => window.scopeDiagnostics);
        assert.deepEqual(diagnostics, { csp: [], rejections: [] });
        assert.deepEqual(requests, url.endsWith('/') ? [url, new URL('mlflow-model-scopes.html', url).href] : [url]);
        records.push({ engine, browserVersion: browser.version(), width,
          inputComplete: false, observations: 40, modelRowsNotAdded: 20, findings: 0,
          nativeDatabaseSha256: oracle.database.sha256, installedCliExit: replay.status,
          svgSha256: sha(await readFile(svgPath)), svgMetadata: metadata,
          errors, requests, diagnostics, overflow: false });
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(resolve(output, 'browser.json'), JSON.stringify({ checkedAtUtc: new Date().toISOString(), url,
  scope: 'Actual served sample and original native database download, offline installed CLI replay and standalone SVG. No analysis responses or assets mocked.', records }, null, 2) + '\n');
console.log(`${records.length} native scope report workflows passed`);
