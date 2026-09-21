"""
Evaluation a trafic benin intact.

Nos X_adv contiennent les 20 000 lignes perturbees, dont les 10 000 benignes.
Or un attaquant perturbe SES flux d'attaque pour passer inapercu ; il ne
controle pas le trafic legitime de sa victime. Perturber les lignes benignes
mesure donc un scenario qui ne peut pas se produire.

Ce script reevalue en ne gardant la version attaquee que pour les lignes
d'attaque, et la version propre pour les lignes benignes. Aucun calcul GPU
lourd : c'est un masque sur des fichiers deja generes.

Le papier ne precise pas ce qu'il fait. Sa phrase "The baseline IDS classifier
would incorrectly identify the normal as intrusions if perturbations were
added, and vice versa" suggere qu'il perturbe tout, comme nous. Mais c'est
ambigu.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import DNN
from src.utils.commun import metriques, predire
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
ORDRE = ["Clean"] + ATTAQUES
PAPIER = {"Clean": 98.11, "FGSM": 54.5, "BIM": 45.0, "PGD": 46.0,
          "DeepFool": 53.0, "JSMA": 81.0, "CW": 36.0}


def charger_modele(cfg, fichier, device):
    ck = torch.load(Path(cfg.paths["checkpoints"]) / fichier,
                    weights_only=False, map_location=device)
    m = DNN(input_dim=cfg.dataset["num_features"],
            hidden=tuple(cfg.model["hidden_layers"]),
            output_dim=cfg.dataset["num_classes"]).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--baseline", default="baseline_varlr_ep150.pth")
    parseur.add_argument("--attaques", default="",
                         help="suffixe des X_adv, par exemple _l2")
    args = parseur.parse_args()

    print("=" * 96)
    print("Evaluation a trafic benin intact")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 96 + "\n")

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))
    atk_dir = Path(cfg.paths["attacks"])
    bs = cfg.evaluation["batch_size"]

    model = charger_modele(cfg, args.baseline, device)
    X = joblib.load(atk_dir / "X_test_clean.pkl")
    y = joblib.load(atk_dir / "y_test_clean.pkl")

    est_benin = (y == 0)
    print(f"Modele   : {args.baseline}")
    print(f"Attaques : {args.attaques or 'L-inf, Table 2'}")
    print(f"Test     : {len(y):,} lignes, {int(est_benin.sum()):,} benignes "
          f"conservees intactes, {int((~est_benin).sum()):,} attaquees\n")

    res_tout, res_partiel = {}, {}
    m0 = metriques(y, predire(model, X, device, bs), labels, "Clean")
    res_tout["Clean"] = res_partiel["Clean"] = m0

    for nom in ATTAQUES:
        X_adv = joblib.load(atk_dir / f"X_adv_test_{nom.lower()}{args.attaques}.pkl")
        res_tout[nom] = metriques(y, predire(model, X_adv, device, bs),
                                  labels, nom)
        # Seules les lignes d'attaque gardent leur version perturbee
        X_mix = X_adv.copy()
        X_mix[est_benin] = X[est_benin]
        res_partiel[nom] = metriques(y, predire(model, X_mix, device, bs),
                                     labels, nom)
        del X_adv, X_mix

    print("=" * 96)
    print("Tout perturbe (ce qu'on faisait) contre benin intact (realiste)")
    print("=" * 96)
    print(f"{'Attaque':<10} | {'Accuracy':^24} | {'F1 macro':^24} | "
          f"{'Papier':>8}")
    print(f"{'':10} | {'tout':>7} {'intact':>8} {'ecart':>7} | "
          f"{'tout':>7} {'intact':>8} {'ecart':>7} | {'acc.':>8}")
    print("-" * 96)

    for nom in ORDRE:
        a, b = res_tout[nom], res_partiel[nom]
        print(f"{nom:<10} | {100*a['accuracy']:>6.2f}% {100*b['accuracy']:>7.2f}% "
              f"{100*(b['accuracy']-a['accuracy']):>+7.2f} | "
              f"{a['f1_macro']:>7.4f} {b['f1_macro']:>8.4f} "
              f"{b['f1_macro']-a['f1_macro']:>+7.4f} | {PAPIER[nom]:>7.1f}%")

    att = ATTAQUES
    e_tout = np.mean([abs(100*res_tout[n]["accuracy"] - PAPIER[n]) for n in att])
    e_part = np.mean([abs(100*res_partiel[n]["accuracy"] - PAPIER[n]) for n in att])
    print("-" * 96)
    print(f"Ecart absolu moyen avec le papier : {e_tout:.1f} points en "
          f"perturbant tout, {e_part:.1f} en gardant le benin intact")
    print()

    print("=" * 96)
    print("Detection des attaques : TPR et rappel BENIGN")
    print("=" * 96)
    print(f"{'Attaque':<10} | {'TPR':^19} | {'Rappel BENIGN':^19}")
    print(f"{'':10} | {'tout':>8} {'intact':>9} | {'tout':>8} {'intact':>9}")
    print("-" * 96)
    for nom in ORDRE:
        a, b = res_tout[nom], res_partiel[nom]
        print(f"{nom:<10} | {a['tpr']:>8.3f} {b['tpr']:>9.3f} | "
              f"{a['recall_benign']:>8.3f} {b['recall_benign']:>9.3f}")
    print()
    print("Un rappel BENIGN de 1.000 en colonne intact est attendu : ces lignes")
    print("n'ont pas ete touchees. Ce qui compte est le TPR, qui mesure la")
    print("detection reelle des attaques.")
    print("=" * 96)

    lignes = []
    for nom in ORDRE:
        for variante, r in (("tout_perturbe", res_tout),
                            ("benin_intact", res_partiel)):
            d = {"variante": variante, "attaque": nom,
                 "papier_accuracy": PAPIER[nom] / 100}
            d.update({k: v for k, v in r[nom].items()
                      if k not in ("confusion_matrix", "attaque")})
            lignes.append(d)

    sfx = args.attaques
    csv = Path(cfg.paths["figures"]) / f"benin_intact{sfx}.csv"
    pd.DataFrame(lignes).to_csv(csv, index=False, float_format="%.6f")
    sortie = (Path(cfg.paths["logs"]) /
              f"benin_intact{sfx}_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    joblib.dump({"tout": res_tout, "intact": res_partiel,
                 "baseline": args.baseline, "attaques": args.attaques}, sortie)
    print(f"\nCSV : {csv}")
    print(f"Sauvegarde : {sortie}")


if __name__ == "__main__":
    main()
