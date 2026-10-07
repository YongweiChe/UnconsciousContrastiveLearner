## Errata for the paper's experiments

Corrections to the experimental results of the ICLR 2025 paper. Numbers are from the *Results* section below.

- **Fig. 2(c), cosine critic.** Monte Carlo's poor accuracy was a code bug: the estimator ignored the critic's learned temperature. Fixed, it reaches 0.98 recall@1 (Direct 0.99), so footnote 2's explanation should be removed.
- **Fig. 3, independent A–B and B–C models.** The conclusion holds, but Monte Carlo needs more bridge samples than shown: at M = 100k it reaches 0.88–0.94 against 0.99 for the oracle, close to the 0.955 that exact density ratios achieve at that M.
- **Fig. 8 (App. C.4).** The original experiment did not test Assumption 1. It is replaced by one that routes a fraction ρ of the A–C signal around B: Direct and Monte Carlo fall to chance as ρ → 1 while the oracle stays near 1.0 (table below).
- **Sec. 6.2.1 / App. C.1, CLIP, CLAP and LanguageBind.** The experiments were refactored and the conclusions largely hold; the reported numbers (62% vs 14% and 70% vs 58% Recall@10) should be replaced with those below.
- **Sec. 6.3 / App. D, language-conditioned RL.** The conclusion holds with a larger effect: LogSumExp improves success over Direct by 32–47 points rather than 20–30%.

## Results

### Synthetic (`python -m didactic.run --experiment main`, 10 seeds)

Recall@1 among 32 held-out candidates (chance 0.03) at the final epoch, for the critics cosine / dot / −½‖·‖². Monte Carlo uses M = 5k / 100k bridge samples.

| | Direct | Monte Carlo, M = 5k | Monte Carlo, M = 100k |
|---|---|---|---|
| Shared φ_B (Fig. 2) | 0.99 / 0.39 / 1.00 | 0.92 / 0.87 / 0.87 | 0.98 / 0.96 / 0.97 |
| Independent A–B and B–C models (Fig. 3) | 0.03 / 0.03 / 0.04 | 0.82 / 0.73 / 0.70 | 0.94 / 0.89 / 0.88 |
| Ground Truth (trained on A–C pairs) | 1.00 / 0.99 / 1.00 | – | – |

References from the true densities (`python -m didactic.ceiling`): Bayes-optimal recall 0.999; Monte Carlo with exact density ratios 0.79 at M = 5k and 0.955 at M = 100k.

**Fig. 8** (`python -m didactic.run --experiment ci_ablation`, 5 seeds): recall@1 as a fraction ρ of the A–C signal bypasses B. Monte Carlo and exact Lemma 1 use M = 100k; Monte Carlo is the range over the three critics, and Direct is shown for cosine and −½‖·‖².

| ρ | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| Bayes-optimal | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| Ground Truth (trained on A–C pairs) | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 |
| Lemma 1 with exact density ratios | 0.95 | 0.91 | 0.85 | 0.68 | 0.03 |
| Monte Carlo | 0.96–0.98 | 0.91–0.93 | 0.82–0.85 | 0.59–0.63 | 0.03 |
| Direct | 0.99 | 0.93–0.96 | 0.83–0.89 | 0.57–0.63 | 0.03 |

### Language-conditioned RL (`python -m rl.evaluate`, 3 seeds x 200 episodes per maze)

Success rate and SPL (success weighted by shortest-path / path length), mean ± s.e.; both policies use the same trained agent, starts, labels and action noise.

| Maze | Direct success | LogSumExp success | Direct SPL | LogSumExp SPL |
|---|---|---|---|---|
| fork | 0.24 ± 0.02 | **0.71 ± 0.02** | 0.15 ± 0.01 | **0.56 ± 0.02** |
| island | 0.61 ± 0.02 | **0.95 ± 0.01** | 0.40 ± 0.02 | **0.69 ± 0.01** |
| blank | 0.58 ± 0.02 | **0.90 ± 0.01** | 0.35 ± 0.02 | **0.64 ± 0.02** |

### Pretrained models (AudioCaps, 815 clips: 326 test / 82 val / 407 bridge)

Recall@1 over 100 trials of 25 test candidates (chance 0.04), using each pretrained pair's trained logit scale.

| Experiment | Direct | Monte Carlo | Model trained on the pair |
|---|---|---|---|
| CLIP image ↔ CLAP audio via AudioSet ontology (632 labels) | 0.04 | 0.30 | – |
| CLIP image ↔ CLAP audio via 48.6k-caption pool | 0.05 | 0.35 | – |
| LanguageBind image ↔ audio via ontology | 0.37 | 0.26 | – |
| LanguageBind image ↔ audio via 48.6k-caption pool | 0.40 | 0.31 | – |
| ImageBind + CLAP, image ↔ text via audio (App. C.2) | – | 0.36 | 0.41 (OpenCLIP ViT-H in ImageBind), 0.36 (CLIP ViT-B-32) |
| ImageBind + CLIP, audio ↔ text via images (App. C.2) | 0.50 | 0.22 | 0.74 (CLAP) |
| ImageBind, audio ↔ text via images | 0.52 | 0.19 | – |
