# Figures

Generated from `results/` by the report scripts; regenerate with the commands below. Numbers behind every figure are in [`RESULTS.md`](../RESULTS.md).

| File | Paper | Contents | Regenerate |
|---|---|---|---|
| `didactic/main_unconscious.png` | Fig. 2 | Shared bridge encoder φ_B: Direct, Monte Carlo (M = 5k) and Ground Truth recall over training, per critic | `python -m didactic.report --experiment main` |
| `didactic/main_disparate.png` | Fig. 3 | Independent A–B and B–C models (no shared encoder) | same |
| `didactic/fig8_ci_bypass.png` | Fig. 8 | Violating Assumption 1: recall vs the fraction ρ of the A–C signal that bypasses B, with Bayes-optimal and exact-Lemma-1 references | `python -m didactic.report --experiment ci_ablation` |
| `didactic/embeddings_2d.png` | Fig. 7 | φ_B of held-out samples with 2-d embeddings, per critic | `python -m didactic.report --experiment embed_2d` |
| `real_world/comparison.png` | Fig. 4, App. C.2 | Direct vs Monte Carlo vs a model trained on the pair, recall@1 and @10, for CLIP + CLAP, LanguageBind and the ImageBind configurations | `python -m real_world.report` |
| `real_world/scaling.png` | Fig. 5 | Monte Carlo recall@1 vs the number of bridge samples M, against Direct | same |
| `rl/fig13_language_navigation.png` | Fig. 13 | Success rate and SPL of Direct vs LogSumExp navigation in three mazes | `python -m rl.report` |
