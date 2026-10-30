# Framework de défense adversariale pour IDS

*[English](README.md) · Français*

Reproduction et extension de l'article :

**Awad, Z., Zakaria, M., & Hassan, R. (2025).** *An Enhanced Ensemble Defense Framework for Boosting Adversarial Robustness of Intrusion Detection Systems.* Scientific Reports, 15, 14177.
DOI : [10.1038/s41598-025-94023-z](https://doi.org/10.1038/s41598-025-94023-z)

---

## Contexte

Projet de stage de recherche.
Date de début : juillet 2026.

## Objectif

Reproduire et évaluer un framework de défense par ensemble contre les attaques adversariales sur des systèmes de détection d'intrusion (IDS) basés sur l'apprentissage profond.

Le projet suit les 9 étapes du framework proposé par Awad et al. (2025) :

1. Collection des données
2. Preprocessing
3. Sélection de features
4. Séparation train / test
5. Entraînement du DNN baseline
6. Génération des attaques adversariales
7. Test de vulnérabilité du baseline
8. Application des 4 mécanismes de défense
9. Agrégation par ensemble

---

## Statut d'avancement

| Étape | Description | Statut |
|---|---|---|
| 1 | Collection des données | Terminée |
| 2 | Preprocessing | Terminée |
| 3 | Feature selection | Terminée |
| 4 | Split train/val/test + Normalisation + SMOTE | Terminée (5 sept. 2026) |
| 5 | Baseline DNN | Terminée (v5, 100 epochs, sélection sur validation) |
| 6 | Attaques adversariales | Terminée (semi-white box, test complet) |
| 7 | Test de vulnérabilité | Terminée, plus analyse sur test rééquilibré |
| 8 | Mécanismes de défense | **Terminée** (7-8 sept. 2026) |
| 9 | Agrégation par ensemble | **Terminée** (8 sept. 2026) |

Le pipeline complet a été exécuté de bout en bout. Les exécutions de référence de chaque défense sont figées dans
`configs/config.yaml`, section `reference_runs`, avec leur justification.

---

## Résumé des résultats

Tous les chiffres viennent de `results/figures/defenses_summary.csv`, produit
par `scripts/17_plot_defenses.py` qui lit directement les fichiers de
résultats. Aucun n'est recopié à la main : deux valeurs fausses s'étaient
glissées dans une version antérieure de ce document par transcription.

**Baseline sur données propres** : accuracy 99.79 %, F1 macro 0.8411,
F1 pondéré 0.9979.

**Vulnérabilité en semi-white box** : les six attaques ramènent le F1 macro
entre 0.06 et 0.33. Les accuracies restent entre 82 et 88 %, artefact du
plancher fixé par les 83.1 % de trafic bénin du test — voir l'étape 7.

**Tableau de référence.** Le gain est la moyenne des écarts au baseline sur
les six attaques, le coût l'écart sur données propres, le bilan net leur
somme. Le rappel BENIGN minimal est le minimum sur les six attaques.

| Configuration | F1 macro | F1 pondéré | MCC binaire | Rappel BENIGN min | Gain | Coût | Bilan net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8411 | 0.9979 | 0.9936 | 0.8810 | — | — | — |
| **Ensemble parts égales** | 0.8188 | 0.9972 | 0.9922 | 0.9644 | **+0.1189** | −0.0224 | **+0.0965** |
| **Adversarial (PGD, 150)** | **0.8494** | 0.9972 | 0.9914 | 0.8834 | +0.0863 | **+0.0082** | +0.0945 |
| Ensemble optimisé | 0.8264 | 0.9977 | 0.9931 | 0.9531 | +0.1081 | −0.0148 | +0.0933 |
| Ensemble vote majoritaire | 0.8278 | 0.9943 | 0.9811 | **0.9810** | +0.0906 | −0.0133 | +0.0773 |
| Gaussienne (sigma 0.02) | 0.6813 | 0.9696 | 0.8901 | 0.9582 | +0.1084 | −0.1598 | −0.0513 |
| Lissage des étiquettes | 0.8359 | 0.9979 | 0.9936 | **0.6069** | +0.0001 | −0.0052 | −0.0051 |
| Autoencodeur (v5) | 0.4498 | 0.9300 | 0.7540 | 0.9176 | +0.0737 | −0.3914 | −0.3177 |

**Ce que ce tableau établit.**

La proposition centrale de l'article est vérifiée, mais de justesse :
l'ensemble à parts égales dépasse la meilleure défense individuelle de
+0.033 en gain brut. En bilan net l'écart tombe à **0.002**, indépartageable
sans mesure de variance.

**Aucune métrique ne suffit seule.** Le lissage est premier en F1 pondéré
(0.9979, identique au baseline) et dernier en rappel BENIGN minimal (0.6069,
soit 39 % du trafic légitime rejeté sous PGD). Une défense qui ne gagne rien
(+0.0001) paraît parfaite sous la métrique que publie l'article.

**Le MCC binaire nous place au-dessus de l'article.** Sa Table 9 donne 0.596
à 0.698 selon la défense ; nous obtenons 0.754 à 0.994. C'est la comparaison
la moins contestable dont nous disposions, le MCC étant robuste au
déséquilibre — mais nos valeurs portent sur un test à 83.1 % de bénin et les
leurs sur un test de composition non documentée.

**Deux attaques résistent à tout.** Dix-huit configurations testées, rien ne
dépasse 0.068 de F1 macro contre JSMA et 0.092 contre PGD.

**Les défenses sont monomaniaques.** Chacune tire l'essentiel de son gain
d'une seule attaque : l'adversarial 71 % de FGSM, la gaussienne 54 % de
DeepFool. Cela contredit le récit de synergie complémentaire du papier, dont
la Figure 7 montre au contraire des défenses uniformes sur les six attaques.

Le journal complet des neuf étapes, les correctifs méthodologiques, la
comparaison détaillée avec l'article et les limites connues sont dans
[`docs/methodology.md`](docs/methodology.md).

---

## Structure du projet

```
IDS-Adversarial-Defense/
├── src/
│   ├── data/              Chargement et preprocessing
│   ├── models/            dnn.py (baseline), substitute.py
│   ├── attacks/           Implémentations des attaques
│   ├── defenses/          common.py (module partagé des 4 défenses)
│   └── utils/             config.py (chargement + empreintes)
├── notebooks/             Exploration
├── scripts/               Scripts exécutables (numérotés par étape)
├── configs/               config.yaml — source unique de vérité
├── data/                  Non versionné, lien vers scratch
│   ├── raw/               CSV originaux
│   └── processed/         Données prétraitées + data_fingerprint.json
├── results/
│   ├── logs/              Logs d'exécution
│   ├── checkpoints/       baseline, substitut, 4 défenses
│   ├── attacks/           Exemples adversariaux du test
│   │   └── train/         Exemples du train, pour le DAE (08b)
│   └── figures/           Graphiques
└── tests/
```

---

## Installation

Python 3.12 sur Linux, ou Python 3.11 sur les serveurs Alliance Canada.

```bash
python -m venv venv
source venv/bin/activate

pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install torchattacks==3.5.1 --no-deps
pip install adversarial-robustness-toolbox --no-deps
```

### Téléchargement du dataset

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download \
    -d chethuhn/network-intrusion-dataset \
    -p data/raw \
    --unzip
```

---

## Utilisation

```bash
python scripts/00_test_environment.py
python scripts/02_explore_dataset.py
python scripts/03_preprocess_dataset.py

sbatch scripts/04_feature_selection.sh
sbatch scripts/05_split_and_prepare.sh
sbatch scripts/06_train_baseline.sh
sbatch scripts/07_plot_results.sh

sbatch scripts/08_generate_attacks.sh
# 09_generate_attacks_jsma_sample.sh est desactive : ne pas le soumettre.
sbatch scripts/10_evaluate_and_plot_attacks.sh
sbatch scripts/15_balanced_evaluation.sh

sbatch scripts/08b_generate_train_attacks.sh   # prerequis du DAE

sbatch scripts/11_defense_adversarial_training.sh
sbatch scripts/12_defense_label_smoothing.sh
sbatch scripts/13_defense_gaussian_augmentation.sh
sbatch scripts/14_defense_denoising_autoencoder.sh

sbatch scripts/16_ensemble_aggregation.sh
```

Les étapes 05, 06 et 08 se chaînent avec `--dependency=afterok` :

```bash
J05=$(sbatch --parsable scripts/05_split_and_prepare.sh)
J06=$(sbatch --parsable --dependency=afterok:$J05 scripts/06_train_baseline.sh)
J08=$(sbatch --parsable --dependency=afterok:$J06 scripts/08_generate_attacks.sh)
```

**Temps d'exécution mesurés** (narval, A100) :

| Étape | Durée |
|---|---:|
| 04 Feature selection | 52 s |
| 05 Split + SMOTE | 19 s |
| 06 Baseline (100 epochs) | 37 min |
| 08 Attaques (test complet) | 2 h 53 |
| 08b Attaques (train, 400k) | 45 min |
| 11 AT-PGD (100 epochs) | ~3 h |
| 12 LS, 13 GA (100 epochs) | ~45 min |
| 14 DAE (100 epochs) | 9 à 50 min |
| 16 Ensemble | < 1 min |

Certains `.sh` contiennent des directives `--account` et `--gres` spécifiques
au cluster. Sur narval les comptes sont `def-smoolak_cpu` et
`def-smoolak_gpu` (le compte sans suffixe n'existe pas) et les GPU sont des
A100 ; sur nibi l'alias sans suffixe est résolu automatiquement et les GPU
sont des H100. Ces différences ne sont pas versionnées.

---

## Attaques adversariales implémentées

Paramètres alignés sur la Table 2. Générées sur le **substitut**, évaluées par
transfert sur le baseline.

| Attaque | Référence | Type | Norme | Hyperparamètres |
|---|---|---|---|---|
| FGSM | Goodfellow et al., 2014 | Single-step | L∞ | eps=0.2 |
| BIM | Kurakin et al., 2016 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| PGD | Madry et al., 2017 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| DeepFool | Moosavi-Dezfooli et al., 2015 | Iterative | L2 | epsilon=1e-6 (overshoot), max_iter=100 |
| JSMA | Papernot et al., 2015 | Feature-based | L0 | theta=0.1, gamma=1.0, untargeted |
| C&W | Carlini & Wagner, 2016 | Optimization | L2 | max_iter=9, confidence=0.0, untargeted |

Pour C&W, les paramètres laissés implicites par l'article sont explicites dans
la configuration (`binary_search_steps=10`, `initial_const=0.01`,
`learning_rate=0.01`) plutôt que subis comme défauts de bibliothèque. Avec
`max_iter=9`, l'attaque reste probablement sous-itérée.

## Mécanismes de défense

| Script | Défense | Approche | Hyperparamètres |
|---|---|---|---|
| `11` | Adversarial Training | PGD à la volée, mix 50/50 | eps=0.2, 7 pas, alpha=0.071 |
| `12` | Label Smoothing | Cross-entropy adoucie | alpha=0.1 |
| `13` | Gaussian Augmentation | Bruit gaussien à l'entraînement | sigma=0.1 |
| `14` | Denoising Autoencoder | Autoencodeur 58→40→58 en amont | 4 attaques, clean_ratio=0.5 |

Le label smoothing (alpha=0.1) est appliqué aux quatre défenses,
conformément aux Algorithmes 3 et 4.

**Hyperparamètres non documentés par l'article** et donc choisis :
`sigma=0.1` pour GA, `eps=0.2` et 7 pas pour AT, `alpha=0.1` pour LS
(d'après Müller et al. 2019), `hidden_dim`, `clean_ratio` et `l1_reg` pour le
DAE, ainsi que la taille du sous-échantillon de `08b` (400 000).

---
## Environnement de développement

- **Développement local** : Arch Linux (Python 3.12) + Windows PowerShell
- **Exécution intensive** : Alliance Canada (nibi et narval), Python 3.11,
  jobs SLURM avec GPU H100 / A100
- **Synchronisation** : Git + GitHub

---

## Références principales

- Awad et al. (2025). *Enhanced Ensemble Defense Framework.* Scientific Reports.
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Carlini & Wagner (2016). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P.
- Papernot et al. (2015). *The Limitations of Deep Learning in Adversarial Settings.* IEEE EuroS&P.
- Moosavi-Dezfooli et al. (2015). *DeepFool.* CVPR.
- Vincent et al. (2008). *Extracting and Composing Robust Features with Denoising Autoencoders.* ICML.
- Gu & Rigazio (2014). *Towards Deep Neural Network Architectures Robust to Adversarial Examples.* ICLR workshop.
- Zhang et al. (2017). *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.* IEEE TIP.
- Müller et al. (2019). *When Does Label Smoothing Help?* NeurIPS.
- Zantedeschi et al. (2017). *Efficient Defenses Against Adversarial Attacks.* AISec.
- Chawla et al. (2002). *SMOTE.* JAIR.
- Tsipras et al. (2019). *Robustness May Be at Odds with Accuracy.* ICLR.

---

## Licence

Projet académique. Utilisation à des fins de recherche uniquement.
