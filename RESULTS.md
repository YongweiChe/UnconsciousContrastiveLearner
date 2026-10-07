## Errata for the paper's experiments

Corrections to the experimental results of the ICLR 2025 paper. Numbers are from the *Results* section below.

- **Fig. 2(c), cosine critic.** The poor Monte Carlo accuracy was a code bug: the estimator dropped the learned logit scale (temperature) of the critic. With the scale included, Monte Carlo works with the cosine critic (recall@1 0.92 at M = 5k, 0.98 at M = 100k; Direct 0.99), and the accompanying explanation (footnote 2: the critic cannot represent log-ratios outside [1/e, e]) should be removed. The conclusions of Fig. 2(a) (all methods succeed) and Fig. 2(b) (Direct fails with the dot critic while Monte Carlo recovers) hold.
- **Fig. 3, independent A–B and B–C models.** The conclusion holds — Direct comparison is at chance while Monte Carlo approaches the Ground Truth oracle — but it requires more bridge samples than the figure suggests. The original evaluation scored training triplets and included each pair's true B in the bridge pool, which inflated Monte Carlo at small M. With a held-out evaluation and an independent bridge pool, Monte Carlo reaches 0.70–0.82 at M = 5k and 0.88–0.94 at M = 100k, against 0.99 for the oracle; the Monte Carlo estimator with *exact* density ratios reaches 0.955 at M = 100k, so the remaining gap reflects finite M and converges as M grows.
- **Fig. 8, violating conditional independence (App. C.4).** The original Fig. 8 did not test Assumption 1. Its accuracy drop came from an implementation that added an independent noise vector to every dimension of A and C, which degrades all methods including the oracle; with κ implemented as described (one scalar shared along the all-ones direction), accuracy does not degrade at all (Monte Carlo 0.97–0.98 for κ from 0 to 32), because a one-dimensional shared nuisance is easy to ignore and removes none of the information carried by B. We replace it with a new Fig. 8 that does test violations of Assumption 1: a hidden variable U, independent of B, feeds both A and C, and ρ ∈ [0, 1] is the fraction of the A–C signal routed through U instead of B, at constant total signal (ρ = 0 satisfies Assumption 1; ρ = 1 makes B independent of A and C). The Bayes-optimal recall and an oracle trained on (A, C) pairs stay near 1.0 for every ρ, while Direct, Monte Carlo, and Lemma 1 evaluated with exact density ratios all degrade steadily to chance as ρ → 1 — the degradation is predicted by the theory, not an artifact of training. Over 5 seeds (recall@1 among 32 candidates; Monte Carlo and Lemma 1 at M = 100k), for ρ = 0 / 0.25 / 0.5 / 0.75 / 1: exact Lemma 1 0.95 / 0.91 / 0.85 / 0.68 / chance; Monte Carlo 0.96–0.98 / 0.91–0.93 / 0.82–0.85 / 0.59–0.63 / chance across the three critics; Direct (cosine, −½‖·‖²) 0.99 / 0.93–0.96 / 0.83–0.89 / 0.57–0.63 / chance; Bayes-optimal 0.999 and Ground Truth 0.99–1.00 throughout (`figures/didactic/fig8_ci_bypass.png`, `python -m didactic.run --experiment ci_ablation`).
- **Sec. 6.2.1 / App. C.1, CLIP, CLAP and LanguageBind.** These experiments were refactored (held-out bridge samples, each model's own trained temperature, a fixed recall protocol) and the same conclusions largely hold: comparing CLIP image and CLAP audio embeddings directly is at chance (R@10 0.36 with 25 candidates), while Monte Carlo through text recovers R@10 0.76 over the AudioSet ontology and 0.83 over a caption pool, close to LanguageBind's direct comparison (0.88); LanguageBind's own Monte Carlo estimate reaches 0.77–0.82. The reported numbers (62% vs 14%, 70% vs 58% Recall@10) should be replaced with these.
- **App. C.2, swapping the intermediate modality.** Through an audio bridge (image ↔ text, bridging ImageBind's image–audio model with CLAP's audio–text model), Monte Carlo reaches recall@1 0.36, matching a CLIP ViT-B-32 trained directly on image–text pairs (0.36) and close to the larger OpenCLIP ViT-H that ImageBind uses for images and text (0.41). This setting has no unpaired direct comparison, since ImageBind's image and text encoders are themselves a model trained on image–text pairs. Through an image bridge (audio ↔ text, bridging ImageBind's audio–image model with CLIP's image–text model), Monte Carlo still retrieves well above chance (0.22 vs 0.04; one-sided t-test over 100 trials, p < 10⁻³⁹) but falls well short of the direct comparison of ImageBind's audio and text embeddings (0.50), which were never trained together: a single video frame does not determine the sound described by an AudioCaps caption, so conditional independence (Assumption 1) holds only approximately for this bridge. The claim that Monte Carlo matches Direct regardless of the intermediate modality should be qualified accordingly.
- **Sec. 6.3 / App. D, language-conditioned RL.** The conclusion holds with a larger effect: LogSumExp improves success over Direct by 32–47 points (fork 0.24 → 0.71, island 0.61 → 0.95, blank 0.58 → 0.90) rather than 20–30%. The original evaluation scored actions with a critic different from the trained one and computed SPL incorrectly.

## Results

### Synthetic (`python -m didactic.run --experiment main`, 10 seeds)

Recall@1 among 32 held-out candidates (chance 0.03) at the final epoch, for the critics cosine / dot / −½‖·‖². Monte Carlo uses M = 5k / 100k bridge samples.

| | Direct | Monte Carlo, M = 5k | Monte Carlo, M = 100k |
|---|---|---|---|
| Shared φ_B (Fig. 2) | 0.99 / 0.39 / 1.00 | 0.92 / 0.87 / 0.87 | 0.98 / 0.96 / 0.97 |
| Independent A–B and B–C models (Fig. 3) | 0.03 / 0.03 / 0.04 | 0.82 / 0.73 / 0.70 | 0.94 / 0.89 / 0.88 |
| Ground Truth (trained on A–C pairs) | 1.00 / 0.99 / 1.00 | – | – |

References from the true densities (`python -m didactic.ceiling`): Bayes-optimal recall 0.999; Monte Carlo with exact density ratios 0.79 at M = 5k and 0.955 at M = 100k.

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
