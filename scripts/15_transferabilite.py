"""
Mesure de la transferabilite des attaques.

Les attaques sont generees sur le substitut puis transferees sur le
detecteur. Si elles marchent tres bien sur le substitut mais mal sur le
detecteur, la transferabilite est faible et les chiffres du detecteur restent
hauts. C'est la derniere piste pour expliquer notre ecart avec le papier.

Trois choses mesurees :
  1. l'effet de chaque attaque sur le substitut lui-meme (white box)
  2. son effet sur le detecteur (transfert, ce qu'on rapporte)
  3. le taux de transfert : part des attaques reussies sur le substitut qui
     reussissent aussi sur le detecteur

Le papier annonce un substitut a 0.98 de F1, le notre est a 0.9700. S'il
transferait moins bien que le notre, ses attaques seraient plus faibles et ses
accuracies plus hautes — ce qui correspondrait a l'ecart observe.
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
from src.models.dnn import DNN, Substitut
from src.utils.commun import metriques, predire
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
PAPIER = {"FGSM": 54.5, "BIM": 45.0, "PGD": 46.0,
          "DeepFool": 53.0, "JSMA": 81.0, "CW": 36.0}


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--baseline", default="baseline_varlr_ep150.pth")
    parseur.add_argument("--substitut", default="substitut.pth")
    parseur.add_argument("--attaques", default="")
    args = parseur.parse_args()

    print("=" * 96)
    print("Transferabilite des attaques")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 96 + "\n")

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))
    atk_dir = Path(cfg.paths["attacks"])
    ck_dir = Path(cfg.paths["checkpoints"])
    bs = cfg.evaluation["batch_size"]

    ck = torch.load(ck_dir / args.baseline, weights_only=False, map_location=device)
    detecteur = DNN(input_dim=cfg.dataset["num_features"],
                    hidden=tuple(cfg.model["hidden_layers"]),
                    output_dim=cfg.dataset["num_classes"]).to(device)
    detecteur.load_state_dict(ck["model_state_dict"])
    detecteur.eval()

    ck_s = torch.load(ck_dir / args.substitut, weights_only=False,
                      map_location=device)
    substitut = Substitut(input_dim=cfg.dataset["num_features"],
                          hidden=tuple(cfg.substitute["hidden_layers"]),
                          output_dim=cfg.dataset["num_classes"]).to(device)
    substitut.load_state_dict(ck_s["model_state_dict"])
    substitut.eval()

    print(f"Detecteur : {args.baseline}  ({ck['epoch']} passages, "
          f"val_acc {ck['val_acc']:.4f})")
    print(f"Substitut : {args.substitut}  "
          f"(F1 pondere {ck_s.get('f1_weighted', float('nan')):.4f})")
    print(f"Attaques  : {args.attaques or 'L-inf, Table 2'}\n")

    X = joblib.load(atk_dir / "X_test_clean.pkl")
    y = joblib.load(atk_dir / "y_test_clean.pkl")

    pred_det_clean = predire(detecteur, X, device, bs)
    pred_sub_clean = predire(substitut, X, device, bs)
    acc_det = metriques(y, pred_det_clean, labels)["accuracy"]
    acc_sub = metriques(y, pred_sub_clean, labels)["accuracy"]
    print(f"Sur donnees propres : detecteur {100*acc_det:.2f}%, "
          f"substitut {100*acc_sub:.2f}%")
    # Accord entre les deux modeles : plus il est haut, plus leurs frontieres
    # de decision se ressemblent, donc plus le transfert sera facile.
    accord = float((pred_det_clean == pred_sub_clean).mean())
    print(f"Accord entre les deux modeles : {100*accord:.2f}%\n")

    lignes = []
    print("=" * 96)
    print(f"{'Attaque':<10} | {'Substitut':^17} | {'Detecteur':^17} | "
          f"{'Transfert':>10} | {'Papier':>8}")
    print(f"{'':10} | {'acc.':>8} {'chute':>8} | {'acc.':>8} {'chute':>8} | "
          f"{'':>10} | {'acc.':>8}")
    print("-" * 96)

    for nom in ATTAQUES:
        X_adv = joblib.load(atk_dir / f"X_adv_test_{nom.lower()}{args.attaques}.pkl")
        p_sub = predire(substitut, X_adv, device, bs)
        p_det = predire(detecteur, X_adv, device, bs)

        a_sub = metriques(y, p_sub, labels)["accuracy"]
        a_det = metriques(y, p_det, labels)["accuracy"]

        # Une attaque "reussit" quand elle fait passer une prediction correcte
        # a incorrecte, sur le modele considere.
        reussi_sub = (pred_sub_clean == y) & (p_sub != y)
        reussi_det = (pred_det_clean == y) & (p_det != y)
        # Taux de transfert : parmi les reussites sur le substitut, la part
        # qui reussit aussi sur le detecteur.
        n_sub = int(reussi_sub.sum())
        taux = float((reussi_sub & reussi_det).sum() / n_sub) if n_sub else 0.0

        print(f"{nom:<10} | {100*a_sub:>7.2f}% {100*(acc_sub-a_sub):>7.2f} | "
              f"{100*a_det:>7.2f}% {100*(acc_det-a_det):>7.2f} | "
              f"{100*taux:>9.1f}% | {PAPIER[nom]:>7.1f}%")

        lignes.append({
            "attaque": nom,
            "acc_substitut": a_sub, "acc_detecteur": a_det,
            "chute_substitut": acc_sub - a_sub,
            "chute_detecteur": acc_det - a_det,
            "taux_transfert": taux,
            "n_reussi_substitut": n_sub,
            "n_reussi_detecteur": int(reussi_det.sum()),
            "papier_accuracy": PAPIER[nom] / 100,
        })
        del X_adv

    print("-" * 96)
    df = pd.DataFrame(lignes)
    print(f"Taux de transfert moyen : {100*df['taux_transfert'].mean():.1f}%")
    print()

    print("Lecture. Un taux de transfert eleve signifie que les attaques du")
    print("substitut trompent aussi le detecteur : les deux modeles ont des")
    print("frontieres de decision proches. Un taux bas signifie que l'attaque")
    print("est specifique au substitut et se perd au transfert.")
    print()
    if df["taux_transfert"].mean() > 0.6:
        print("Notre transfert est ELEVE. Si le leur etait plus faible, leurs")
        print("attaques atteindraient moins le detecteur et leurs accuracies")
        print("resteraient plus hautes, ce qui expliquerait l'ecart.")
    else:
        print("Notre transfert est MODERE. Cette piste explique mal l'ecart.")
    print("=" * 96)

    sfx = args.attaques
    csv = Path(cfg.paths["figures"]) / f"transferabilite{sfx}.csv"
    df.to_csv(csv, index=False, float_format="%.6f")
    sortie = (Path(cfg.paths["logs"]) /
              f"transferabilite{sfx}_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    joblib.dump({"lignes": lignes, "accord_clean": accord,
                 "acc_detecteur_clean": acc_det,
                 "acc_substitut_clean": acc_sub,
                 "baseline": args.baseline, "substitut": args.substitut}, sortie)
    print(f"\nCSV : {csv}")
    print(f"Sauvegarde : {sortie}")


if __name__ == "__main__":
    main()
