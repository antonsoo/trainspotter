# Learning-rate evidence around unusable readings

The fix is `00dab47290d99d68477dc510ea25f3b570e615f4`, against baseline
`44c5379f404d7016eb3ae49ef4d2e99fa0748429`. Package metadata remains 0.2.6.
This is source and locally installed wheel verification, not a PyPI release
or hosted-demo deployment.

One damaged learning-rate cell previously hid supported changes elsewhere.
In the controlled optimizer capture, valid drops at steps 39-40 and 79-80
disappeared when step 10 was unreadable, NaN or infinite. Replacing that cell
with -1 instead invented a jump at steps 9-10. The repaired reports retain
the two supported drops and keep every unusable reading as a boundary.
The [study](../studies/lr-gaps/README.md) documents the exact rule and limits.

## Capture and CLI

The 120-step CPU PyTorch capture uses actual SGD and MultiStepLR calls.
Each damaged export differs from the complete log in exactly one LR cell;
optimizer state and objective values are unchanged. A 70-digit Decimal
closed-form check agrees with every captured weight and objective, with
maximum absolute difference `8.881784197001252e-16`.

The CLI replay checks six captured exports and three synthetic boundary
controls, through JSON and standalone HTML. Unreadable inputs retain exit 2
and their precise coordinate. Non-LR findings and reader metadata remain
unchanged in all nine before/after cases. Every minimum-dependency JSON report
equals its Python 3.14 counterpart exactly. See the [minimum replay](../studies/lr-gaps/results/minimum.json)
and [modern replay](../studies/lr-gaps/results/installed314.json).

## Fresh archive and wheel

An empty directory received a `git archive` of the fix. The locked Python
3.12 environment passes Ruff, mypy on 28 source files, all 605 tests, and
`uv build`. The build creates the wheel from the source distribution.
The wheel was installed into isolated Python 3.10 and 3.14 environments;
tests there use `-c /dev/null` to avoid depending on checkout configuration.

| Environment | NumPy | Full tests | Nine-case CLI replay |
| --- | --- | --- | --- |
| Clean source, Python 3.12.12 | 2.5.3 | 605 passed | Baseline/current source investigation |
| Installed wheel, Python 3.10.21 | 2.2.6 | 605 passed | Passed |
| Installed wheel, Python 3.10.21, dependency minimums | 1.24.0 | 605 passed | Passed |
| Installed wheel, Python 3.14.7 | 2.5.4 | 605 passed | Passed |

The minimum environment also pins TensorBoard 2.12.0 and scikit-learn 1.5.0,
along with the declared development-tool floors. The initial installed test
runs each emitted one pytest cache warning because `/dev/null` selected a
nonwritable cache root. The minimum run disables that cache provider; all
tests pass. These are harness warnings, not skipped tests or product errors.

The [artifact record](../studies/lr-gaps/results/artifacts.json) compares 87
source/build inputs with the clean archive and all 29 installed package files
with both wheel members and clean source bytes. It records the source archive,
wheel and sdist hashes. The 115,557-byte wheel has SHA-256
`8521f63a1d21821ad9703d61e597c2386e3289cd8525563f34f7829eaa31a9f4`.

The retained outputs include [install](../studies/lr-gaps/results/clean-install.txt),
[lint](../studies/lr-gaps/results/clean-lint.txt), [types](../studies/lr-gaps/results/clean-types.txt),
[tests](../studies/lr-gaps/results/clean-tests.txt), [build](../studies/lr-gaps/results/clean-build.txt),
[minimum tests](../studies/lr-gaps/results/minimum-tests.txt), and
[Python 3.14 tests](../studies/lr-gaps/results/tests314.txt).
Text logs normalize scratch paths and final blank lines; JSON reports and
retained HTML preserve their recorded bytes.

## Browser and historical compatibility

The clean archive also passes the repository's full `npm run test:browser`:
**204 report workflows** in Chromium/Firefox at desktop and 320px, with
keyboard, accessibility, CSP and no-network checks. [Complete output](../studies/lr-gaps/results/clean-browser.txt).

The installed Python 3.14 HTML reports pass eight offline file workflows across
Chromium and Firefox at 1440px and 375px. The baseline passes the same eight
workflows while showing its missing LR findings. Each workflow verifies the
displayed findings, incomplete-input warning, keyboard-operated download of
the unreadable LR coordinate, accessibility and width constraints. There are
no HTTP requests, page errors, CSP violations or accessibility findings.
The [receipt](../studies/lr-gaps/results/browser-installed.json) binds the
actual opened HTML and screenshots by SHA-256. All eight retained before/after
screenshots were inspected, including the final installed captures.

The public-log replay compares complete finding dictionaries, including
messages and evidence, with the same current reader. It finds no changes in
872 findings over 5,048,345 LR observations from 4,052 usable LR histories.
The available corpus contains 4,177 readable logs; 82 listed files remain
unavailable. The [receipt](../studies/lr-gaps/results/corpus.json) names those
omissions and pins the final detector hash. This establishes compatibility on
that available cache, not calibrated diagnostic accuracy.

Findings remain informational; intended step schedules can produce these
notes. The method cannot reconstruct LR behavior through a gap or establish
that a logged rate was actually used by an optimizer. The minimum six usable
observations applies within each discontinuity segment, and the threshold
still uses the whole run's usable range. No hosted report is updated here.
