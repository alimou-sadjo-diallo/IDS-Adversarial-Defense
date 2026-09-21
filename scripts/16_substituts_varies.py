"""
Effet du choix du substitut sur les resultats rapportes.

Notre substitut est d'accord a 97.58 % avec le detecteur : leurs frontieres
de decision sont quasi confondues, donc 78.2 % des attaques transferent. On
est presque en white box sans l'avoir voulu, alors que le papier decrit le
semi-white box comme le scenario realiste.

Ce script fabrique des substituts de plus en plus ELOIGNES du detecteur et
mesure si son accuracy remonte. Le detecteur n'est pas touche : c'est lui
qu'on teste.

  papier      58-100-100-15, tout le train, graine 42   (le notre)
  graine      idem mais graine 7
  petit       58-30-30-15
  minuscule   58-15-15-15
  partiel     58-100-100-15 sur 20 % du train

Si l'accuracy du detecteur monte quand l'accord baisse, l'ecart avec le
papier s'explique par un parametre que personne ne documente.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

if not hasattr(np, "product"):
    np.product = np.prod

import pandas as pd
import torch
import torch.nn as nn
import torchattacks
from art.attacks.evasion import CarliniL2Method, DeepFool, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import DNN, Substitut
from src.utils.commun import charger_donnees, metriques, predire
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
PAPIER = {"FGSM": 54.5, "BIM": 45.0, "PGD": 46.0,
          "DeepFool": 53.0, "JSMA": 81.0, "CW": 36.0}

# nom -> (couches, graine, fraction du train)
VARIANTES = {
    "papier":    ((100, 100), 42, 1.00),
    "graine":    ((100, 100),  7, 1.00),
    "petit":     (( 30,  30), 42, 1.00),
    "minuscule": (( 15,  15), 42, 1.00),
    "partiel":   ((100, 100), 42, 0.20),
}


def entrainer_substitut(cfg, couches, graine, fraction, X_tr, y_tr, X_va,
                        y_va, device, labels):
    """lr 0.01, 30 passages, batch 256, comme le papier."""
    torch.manual_seed(graine)
    np.random.seed(graine)

    if fraction < 1.0:
        rng = np.random.default_rng(graine)
        n = int(len(X_tr) * fraction)
        idx = rng.choice(len(X_tr), size=n, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    s = cfg.substitute
    model = Substitut(input_dim=cfg.dataset["num_features"],
                      hidden=couches,
                      output_dim=cfg.dataset["num_classes"]).to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=s["batch_size"], shuffle=True, num_workers=4,
        pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=s["learning_rate"])

    for _ in range(s["epochs"]):
        model.train()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

    model.eval()
    f1 = metriques(y_va, predire(model, X_va, device, 512), labels)["f1_weighted"]
    n_par = sum(p.numel() for p in model.parameters())
    return model, f1, n_par, len(X_tr)


def generer(nom, model, art_clf, X, y, cfg, device, bs):
    p = cfg.attacks[nom]
    X_adv = np.zeros_like(X)

    if nom == "FGSM":
        atk = torchattacks.FGSM(model, eps=p["eps"])
    elif nom == "BIM":
        atk = torchattacks.BIM(model, eps=p["eps"], alpha=p["alpha"],
                               steps=p["steps"])
    elif nom == "PGD":
        atk = torchattacks.PGD(model, eps=p["eps"], alpha=p["alpha"],
                               steps=p["steps"])
    elif nom == "DeepFool":
        atk = DeepFool(classifier=art_clf, epsilon=p["epsilon"],
                       max_iter=p["max_iter"], batch_size=bs, verbose=False)
    elif nom == "JSMA":
        atk = SaliencyMapMethod(classifier=art_clf, theta=p["theta"],
                                gamma=p["gamma"], batch_size=bs, verbose=False)
    else:
        atk = CarliniL2Method(classifier=art_clf, max_iter=p["max_iter"],
                              confidence=p["confidence"],
                              binary_search_steps=p["binary_search_steps"],
                              initial_const=p["initial_const"],
                              learning_rate=p["learning_rate"],
                              batch_size=bs, verbose=False)

    for i in range(0, len(X), bs):
        fin = min(i + bs, len(X))
        if nom in ("FGSM", "BIM", "PGD"):
            xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
            yb = torch.tensor(y[i:fin], dtype=torch.long).to(device)
            X_adv[i:fin] = atk(xb, yb).detach().cpu().numpy()
        else:
            X_adv[i:fin] = atk.generate(x=X[i:fin], y=None)

    bas, haut = cfg.clip_values
    return np.clip(X_adv, bas, haut)


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--baseline", default="baseline_varlr_ep150.pth")
    args = parseur.parse_args()

    print("=" * 100)
    print("Effet du choix du substitut")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 100 + "\n")

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))
    atk_dir = Path(cfg.paths["attacks"])
    bs = cfg.evaluation["batch_size"]

    ck = torch.load(Path(cfg.paths["checkpoints"]) / args.baseline,
                    weights_only=False, map_location=device)
    detecteur = DNN(input_dim=cfg.dataset["num_features"],
                    hidden=tuple(cfg.model["hidden_layers"]),
                    output_dim=cfg.dataset["num_classes"]).to(device)
    detecteur.load_state_dict(ck["model_state_dict"])
    detecteur.eval()
    print(f"Detecteur : {args.baseline}, il n'est PAS modifie\n")

    X_tr, y_tr, X_va, y_va, _, _ = charger_donnees(cfg.paths["processed"])
    X = joblib.load(atk_dir / "X_test_clean.pkl")
    y = joblib.load(atk_dir / "y_test_clean.pkl")

    pred_det_clean = predire(detecteur, X, device, bs)
    acc_det = metriques(y, pred_det_clean, labels)["accuracy"]
    print(f"Detecteur sur donnees propres : {100*acc_det:.2f}%\n")

    lignes = []
    for nom_var, (couches, graine, frac) in VARIANTES.items():
        t0 = time.time()
        print("=" * 100)
        print(f"Substitut « {nom_var} » : {couches}, graine {graine}, "
              f"{frac:.0%} du train")
        print("=" * 100)

        sub, f1_sub, n_par, n_tr = entrainer_substitut(
            cfg, couches, graine, frac, X_tr, y_tr, X_va, y_va, device, labels)

        pred_sub_clean = predire(sub, X, device, bs)
        acc_sub = metriques(y, pred_sub_clean, labels)["accuracy"]
        accord = float((pred_det_clean == pred_sub_clean).mean())
        print(f"  {n_par:,} parametres, {n_tr:,} lignes d'entrainement")
        print(f"  F1 pondere validation : {f1_sub:.4f}")
        print(f"  Accuracy sur le test  : {100*acc_sub:.2f}%")
        print(f"  ACCORD avec le detecteur : {100*accord:.2f}%\n")

        art_clf = PyTorchClassifier(
            model=sub, loss=nn.CrossEntropyLoss(),
            input_shape=(cfg.dataset["num_features"],),
            nb_classes=cfg.dataset["num_classes"],
            clip_values=cfg.clip_values,
            device_type="gpu" if device.type == "cuda" else "cpu")

        print(f"  {'Attaque':<10} {'Det. acc.':>10} {'Transfert':>10} "
              f"{'Papier':>9}")
        transferts, accs = [], []
        for nom_atk in ATTAQUES:
            X_adv = generer(nom_atk, sub, art_clf, X, y, cfg, device, bs)
            p_sub = predire(sub, X_adv, device, bs)
            p_det = predire(detecteur, X_adv, device, bs)
            a_det = metriques(y, p_det, labels)["accuracy"]

            r_sub = (pred_sub_clean == y) & (p_sub != y)
            r_det = (pred_det_clean == y) & (p_det != y)
            n = int(r_sub.sum())
            taux = float((r_sub & r_det).sum() / n) if n else 0.0

            print(f"  {nom_atk:<10} {100*a_det:>9.2f}% {100*taux:>9.1f}% "
                  f"{PAPIER[nom_atk]:>8.1f}%")
            transferts.append(taux)
            accs.append(a_det)
            lignes.append({
                "substitut": nom_var, "couches": str(couches),
                "graine": graine, "fraction_train": frac,
                "n_parametres": n_par, "f1_substitut": f1_sub,
                "accord": accord, "attaque": nom_atk,
                "acc_detecteur": a_det, "taux_transfert": taux,
                "papier_accuracy": PAPIER[nom_atk] / 100,
            })
            del X_adv

        ecart = np.mean([abs(100*a - PAPIER[n])
                         for a, n in zip(accs, ATTAQUES)])
        print(f"\n  Accuracy moyenne du detecteur : {100*np.mean(accs):.2f}%")
        print(f"  Transfert moyen : {100*np.mean(transferts):.1f}%")
        print(f"  Ecart moyen au papier : {ecart:.1f} points")
        print(f"  ({(time.time()-t0)/60:.1f} min)\n")

    df = pd.DataFrame(lignes)
    print("=" * 100)
    print("SYNTHESE")
    print("=" * 100)
    print(f"{'Substitut':<12} {'Accord':>8} {'Transfert':>10} "
          f"{'Det. acc.':>10} {'Ecart papier':>13}")
    print("-" * 100)
    for nom_var in VARIANTES:
        d = df[df.substitut == nom_var]
        ecart = np.mean([abs(100*r.acc_detecteur - 100*r.papier_accuracy)
                         for r in d.itertuples()])
        print(f"{nom_var:<12} {100*d.accord.iloc[0]:>7.2f}% "
              f"{100*d.taux_transfert.mean():>9.1f}% "
              f"{100*d.acc_detecteur.mean():>9.2f}% {ecart:>12.1f}")
    print("-" * 100)

    corr = df.groupby("substitut").agg(
        accord=("accord", "first"), acc=("acc_detecteur", "mean")).corr()
    print(f"\nCorrelation entre accord et accuracy du detecteur : "
          f"{corr.iloc[0, 1]:+.3f}")
    print("Une correlation negative forte signifie que plus le substitut")
    print("ressemble au detecteur, plus il le fait tomber bas.")
    print("=" * 100)

    csv = Path(cfg.paths["figures"]) / "substituts_varies.csv"
    df.to_csv(csv, index=False, float_format="%.6f")
    sortie = (Path(cfg.paths["logs"]) /
              f"substituts_varies_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    joblib.dump({"lignes": lignes, "acc_detecteur_clean": acc_det,
                 "baseline": args.baseline}, sortie)
    print(f"\nCSV : {csv}")
    print(f"Sauvegarde : {sortie}")


if __name__ == "__main__":
    main()
