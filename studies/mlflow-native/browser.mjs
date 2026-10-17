// Open the actual sample, download its native database and chart, and replay both offline.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium, firefox } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const [url, outArg, python] = process.argv.slice(2);
assert.ok(url && outArg && python, 'URL OUTPUT_DIR INSTALLED_PYTHON');
const output = resolve(outArg);
await mkdir(output, { recursive: true });
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const fixture = resolve(root, 'studies/mlflow-native/fixtures/mlflow-3.17.1');
const manifest = JSON.parse(await readFile(resolve(fixture, 'capture.json'), 'utf8'));
const native = manifest.stores.find(item => item.kind === 'sqlite').runs.find(item => item.name === 'slow-training');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const inputSha = sha(await readFile(resolve(fixture, 'mlflow.db')));
const results = [];
for (const [engine, launch] of Object.entries({ chromium, firefox })) {
  const browser = await launch.launch();
  try {
    for (const width of [1440, 375]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 } });
      const requests = [], errors = [];
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
      try {
        await page.goto(url);
        await context.setOffline(true);
        assert.equal(await page.locator('script').count(), 0);
        assert.match(await page.locator('.input-notes').first().innerText(), /MLflow run 'slow-training'/);
        assert.match(await page.locator('.input-notes').first().innerText(), new RegExp(native.runId));
        assert.equal(await page.locator('.finding h3').innerText(), 'Throughput regression');
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        assert.deepEqual((await new AxeBuilder({ page }).analyze()).violations, []);
        const stem = `${engine}-${width}`;
        await page.screenshot({ path: resolve(output, `${stem}-report.png`), fullPage: true });
        let pending = page.waitForEvent('download');
        await page.getByRole('link', { name: 'Download the input log' }).click();
        const database = await pending;
        assert.equal(await database.failure(), null);
        const dbPath = resolve(output, `${stem}.db`);
        await database.saveAs(dbPath);
        assert.equal(sha(await readFile(dbPath)), inputSha);
        const cli = spawnSync(python, ['-m', 'trainspotter.cli', 'analyze', dbPath, '--run', native.runId,
                                     '--step-key', 'step', '--output', 'json'], {
          cwd: output, encoding: 'utf8', timeout: 30_000,
        });
        assert.equal(cli.status, 0, cli.stderr);
        const report = JSON.parse(cli.stdout);
        assert.equal(report.run.mlflow.run.run_id, native.runId);
        assert.equal(report.timing.status, 'compared');
        assert.ok(report.timing.ratio > 2 && report.timing.ratio < 4);
        const link = page.getByRole('link', { name: /^Download .* chart as SVG$/ });
        assert.equal(await link.count(), 1);
        pending = page.waitForEvent('download');
        await link.focus(); await page.keyboard.press('Enter');
        const vector = await pending;
        assert.equal(await vector.failure(), null);
        const svgPath = resolve(output, `${stem}.svg`);
        await vector.saveAs(svgPath);
        const standalone = await context.newPage();
        try {
          await standalone.goto(pathToFileURL(svgPath).href);
          const metadata = JSON.parse(await standalone.locator('metadata').textContent());
          assert.equal(metadata.sourceFormat, 'mlflow');
          assert.equal(metadata.selectedRun, native.runId);
          assert.equal(metadata.axisKind, 'step');
          assert.ok(metadata.inputNotes.some(note => note.includes('slow-training')));
          assert.ok(metadata.inputNotes.some(note => note.includes('cannot reconstruct infinities')));
          assert.ok(await standalone.locator('svg.chart path').count() > 0);
          await standalone.locator('svg').first().screenshot({ path: resolve(output, `${stem}-standalone.png`) });
          results.push({ engine, width, nativeDatabaseSha256: inputSha, installedCliTiming: report.timing,
                         svgSha256: sha(await readFile(svgPath)), metadata, errors, requests });
        } finally { await standalone.close(); }
        assert.deepEqual(errors, []);
        assert.deepEqual(requests, /^https?:/.test(url) ? [url] : []);
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
}
await writeFile(resolve(output, 'browser.json'), JSON.stringify({ url, scope: 'Actual sample, offline downloads and installed CLI replay; no new training or remote MLflow server.', results }, null, 2) + '\n');
console.log(JSON.stringify({ contexts: results.length, nativeDatabasesReplayed: results.length, standaloneCharts: results.length }));
