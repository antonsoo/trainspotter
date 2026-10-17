## Report

The [sample gallery](https://antonsoo.github.io/trainspotter/) includes a
[sparse log of a late loss drop](https://antonsoo.github.io/trainspotter/loss-cadence.html),
[constant gradients without clipping](https://antonsoo.github.io/trainspotter/gradient-concentration.html),
[W&B run with changing logging cadence](https://antonsoo.github.io/trainspotter/wandb-cadence.html),
[timing without a training counter](https://antonsoo.github.io/trainspotter/wandb-logging-only.html),
[a measured slowdown](https://antonsoo.github.io/trainspotter/wandb-slowdown.html),
[scheduler reset below its old peak](https://antonsoo.github.io/trainspotter/scheduler-resume.html),
[disjoint training and evaluation intervals](https://antonsoo.github.io/trainspotter/evaluation-overlap.html),
and a [frozen model evaluated on changing cohorts](https://antonsoo.github.io/trainspotter/evaluation-cohorts.html).
Each report explains the controlled experiment and embeds its exact input log
for download, including when opened from disk. These reports use the source
checkout; the detector repairs are not yet in the PyPI release.

![Top of the trainspotter HTML report on the bundled divergence example: a header with the run's source and step range, an error/warning/info summary with a health-strip timeline, and the combined train/loss + eval/loss chart -- 320 steps of ordinary cosine-schedule training, then a labeled spike as the run breaks -- followed by the findings log, the non-finite training loss first.](assets/html-report-divergence.png)

*(top of the report -- [full report, all 5 charts + all 7 findings](assets/html-report-divergence-full.png))*

![Terminal output of `trainspotter analyze divergence.trainer_state.json --fail-on error`, showing all 7 findings ordered by severity -- one ERROR (the training loss became Inf), two WARNINGs (the evaluation-loss and training-loss spikes just before it), then four INFO findings (the evaluation rise attributed to the spike, the LR discontinuity that caused it, the gradient-norm explosion and the non-finite gradient norm) -- each with step range, detector name, message, and a one-line fix suggestion.](assets/terminal-divergence.png)

Browse every bundled example's full HTML report at
**[antonsoo.github.io/trainspotter](https://antonsoo.github.io/trainspotter/)**
(built from the source checkout by `scripts/build_site.py`). The two images above are real output from
`examples/divergence.trainer_state.json`, a run built to fail (see
[Demo data](#real-demo-data) for how).

Current source reports also include [standalone SVG downloads](vector-charts.md)
for their displayed trajectories. The charts and diagnostics remain available
without scripts or an internet connection.
