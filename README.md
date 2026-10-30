# Adversarial Defense Framework for Intrusion Detection

*English · [Français](README.fr.md)*

Reproduction and critical audit of:

**Awad, Z., Zakaria, M., & Hassan, R. (2025).** *An Enhanced Ensemble Defense Framework for Boosting Adversarial Robustness of Intrusion Detection Systems.* Scientific Reports, 15, 14177.
DOI: [10.1038/s41598-025-94023-z](https://doi.org/10.1038/s41598-025-94023-z)

A deep neural network IDS is trained on **CIC-IDS2017** (2.8 M flows, 15 classes), attacked with
**six adversarial attacks** (FGSM, BIM, PGD, DeepFool, JSMA, C&W) in a semi-white-box setting, then
protected by **four defenses** (adversarial training, label smoothing, Gaussian augmentation,
denoising autoencoder) and **three ensemble methods**. The full 9-step pipeline runs end to end on
Alliance Canada GPU clusters (A100 / H100, SLURM).

Research internship project, started July 2026.

---

## Key findings

**1. The paper's central claim holds, but barely.** The equal-weight ensemble beats the best
single defense on raw gain (+0.1189 vs +0.0863 macro F1). Once the cost on clean traffic is
included, the gap falls to **0.002**, which a single run per configuration cannot separate.

**2. No single metric is enough.** Label smoothing ties the baseline on weighted F1 (0.9979), the
metric the paper reports, yet it gains nothing under attack (+0.0001) and **rejects 39 % of
legitimate traffic under PGD** (minimum BENIGN recall 0.6069).

**3. Two attacks defeat every defense.** Across 18 configurations, nothing exceeds 0.068 macro F1
against JSMA or 0.092 against PGD.

**4. The ensemble's real value is insurance, not accuracy.** Adversarial training alone comes within
0.002 of the ensemble. What the ensemble adds is protection against one member collapsing:
minimum BENIGN recall rises from 0.6069 (label smoothing alone) to 0.9810 (majority vote).

**5. The audit found a structural discrepancy in the paper.** Its feature selection deliberately
keeps the 58 *least* important features, so attacks stay functional; this reproduction kept the
most important ones. The paper also contains a wrong precision formula (Eq. 7) and three
mutually inconsistent headline values. All are documented in
[`docs/methodology.md`](docs/methodology.md); a faithful re-run lives on the
[`reproduction-fidele`](../../tree/reproduction-fidele) branch.

---

## Results

<p align="center">
  <img src="results/figures/defenses_gain_vs_cout.png" width="720"
       alt="Average gain under attack versus cost on clean data, per defense">
</p>

*Gain under attack versus cost on clean traffic. Above the red diagonal, the defense earns more
under attack than it costs in normal operation. (Figure labels are in French.)*

<p align="center">
  <img src="results/figures/defenses_heatmap.png" width="720"
       alt="Macro F1 heatmap for each configuration against each attack">
</p>

*Macro F1 of every configuration against every attack. PGD and JSMA columns stay red throughout.*

**Reference table.** Gain = mean macro-F1 change over the six attacks versus the baseline;
cost = change on clean data; net = gain + cost. Minimum BENIGN recall is the worst value over the
six attacks. Every number is generated from
[`results/figures/defenses_summary.csv`](results/figures/defenses_summary.csv) by
`scripts/17_plot_defenses.py`; none is typed by hand.

| Configuration | Macro F1 | Weighted F1 | Binary MCC | Min BENIGN recall | Gain | Cost | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8411 | 0.9979 | 0.9936 | 0.8810 | — | — | — |
| **Ensemble, equal weights** | 0.8188 | 0.9972 | 0.9922 | 0.9644 | **+0.1189** | −0.0224 | **+0.0965** |
| **Adversarial training (PGD, 150 epochs)** | **0.8494** | 0.9972 | 0.9914 | 0.8834 | +0.0863 | **+0.0082** | +0.0945 |
| Ensemble, optimized weights | 0.8264 | 0.9977 | 0.9931 | 0.9531 | +0.1081 | −0.0148 | +0.0933 |
| Ensemble, majority vote | 0.8278 | 0.9943 | 0.9811 | **0.9810** | +0.0906 | −0.0133 | +0.0773 |
| Gaussian augmentation (σ 0.02) | 0.6813 | 0.9696 | 0.8901 | 0.9582 | +0.1084 | −0.1598 | −0.0513 |
| Label smoothing | 0.8359 | 0.9979 | 0.9936 | **0.6069** | +0.0001 | −0.0052 | −0.0051 |
| Denoising autoencoder (v5) | 0.4498 | 0.9300 | 0.7540 | 0.9176 | +0.0737 | −0.3914 | −0.3177 |

Clean baseline: 99.79 % accuracy. Under attack, macro F1 drops to 0.06–0.33.

---

## Method at a glance

| Step | What was done |
|---|---|
| Data | CIC-IDS2017, 2,830,743 rows; 307,078 duplicates removed; 58 features kept |
| Split | Train / validation / test, MinMax scaling, capped SMOTE after the split |
| Baseline | DNN 512-256-15, 100 epochs, model selected on validation only |
| Threat model | Semi-white box: attacks crafted on a substitute model, transferred to the IDS |
| Attacks | FGSM, BIM, PGD, DeepFool, JSMA, C&W, parameters from the paper's Table 2, full test set (831,864 flows) |
| Defenses | Adversarial training, label smoothing, Gaussian augmentation, denoising autoencoder |
| Ensembles | Majority vote, equal-weight soft voting, Nelder-Mead weights optimized on clean validation |

**Engineering safeguards.** `configs/config.yaml` is the single source of truth. Every cached
artifact (data, baseline, substitute, each attack) carries a configuration fingerprint and is
invalidated automatically when the configuration changes. This caught and fixed ten
methodological issues, from a missing substitute model to test-set leakage in model selection,
each documented with its measured effect.

## Known limitations

- **One run per configuration**: differences of ~0.002 are not statistically separable. Three to
  five seeds (~6 GPU hours) would settle them.
- **No adaptive attacker**: attacks target the undefended model, so reported gains are optimistic
  upper bounds. The paper shares this limitation.
- **Benign traffic is perturbed too**, which an attacker could not do in practice.

The full list, with proposed next steps, is in [`docs/methodology.md`](docs/methodology.md).

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install torchattacks==3.5.1 --no-deps
pip install adversarial-robustness-toolbox --no-deps

kaggle datasets download -d chethuhn/network-intrusion-dataset -p data/raw --unzip
```

Scripts in `scripts/` are numbered by pipeline step and submitted with `sbatch` on a SLURM cluster.
Execution order, measured run times and cluster notes are in [`README.fr.md`](README.fr.md#utilisation).

## Repository layout

```
src/        data, models (DNN, substitute), attacks, defenses, utils (config + fingerprints)
scripts/    executable scripts, numbered by pipeline step
configs/    config.yaml, single source of truth
results/    figures and summary CSV (logs, checkpoints and adversarial sets are not versioned)
docs/       detailed methodology, step-by-step log, comparison with the paper
tests/
```

## Documentation

- [`docs/methodology.md`](docs/methodology.md): full log of the nine steps, the ten methodological
  fixes, the detailed comparison with the paper, limitations and next steps (French)
- [`README.fr.md`](README.fr.md): French summary, installation and usage

## Author

**Mamadou Alimou Sadjo Diallo**, B.Sc. Applied Computer Science, Université de Moncton
[github.com/alimou-sadjo-diallo](https://github.com/alimou-sadjo-diallo) · alimoutraz01@gmail.com

## License

Academic project, for research use only.
