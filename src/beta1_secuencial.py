"""BETA 1 - Version secuencial de partida (refactor de knn_digits_sec.py).

Modos:
  original : bucle Python identico al codigo base (argsort + Counter)
  loop     : mismo bucle con desempate determinista (distancia, indice)
  vector   : NumPy vectorizado (kernel gemm o direct) + argpartition

Ejemplos:
  python src/beta1_secuencial.py                       # load_digits, 3 modos
  python src/beta1_secuencial.py --check-augment       # valida el generador
  python src/beta1_secuencial.py --n-train 46000 --n-test 3600 --modes vector
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_core as kc  # noqa: E402


def knn_predict_original(test_point, X_train, y_train, k):
    """Copia literal del codigo base."""
    from collections import Counter
    distances = [kc.euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]


def knn_predict_loop(test_point, X_train, y_train, k):
    """Bucle original pero con orden estable (distancia, indice global)."""
    from collections import Counter
    distances = np.array([kc.euclidean_distance(test_point, x) for x in X_train])
    k_indices = np.argsort(distances, kind='stable')[:k]
    return Counter([y_train[i] for i in k_indices]).most_common(1)[0][0]


def predict_vector(Xtr, ytr, Xte, k, kernel='gemm'):
    gidx = np.arange(len(Xtr), dtype=np.int64)
    _, lab, _ = kc.local_topk(Xte, Xtr, ytr, gidx, k, kernel=kernel)
    return kc.majority_vote(lab)


def run(Xtr, ytr, Xte, yte, k, mode, kernel='gemm'):
    t0 = time.perf_counter()
    if mode == 'original':
        y_pred = np.array([knn_predict_original(x, Xtr, ytr, k) for x in Xte])
    elif mode == 'loop':
        y_pred = np.array([knn_predict_loop(x, Xtr, ytr, k) for x in Xte])
    else:
        y_pred = predict_vector(Xtr, ytr, Xte, k, kernel)
    acc = float(np.mean(y_pred == yte))
    t = time.perf_counter() - t0
    fl = kc.flops_parallel_region(len(Xtr), len(Xte), Xtr.shape[1])
    return y_pred, dict(version='beta1', mode=mode, kernel=kernel if mode == 'vector' else 'loop',
                        n_train=len(Xtr), n_test=len(Xte), d=Xtr.shape[1], k=k, p=1,
                        T_total=t, accuracy=acc, total_flops=fl, gflops_per_sec=fl / 1e9 / t)


def check_augment(k=3, seed=2026):
    """Verifica que el generador preserve la geometria y no degrade accuracy."""
    Xtr, Xte, ytr, yte = kc.load_split()
    rows = []
    for f in [1, 2, 4, 8]:
        Xa, Xb, ya, yb = kc.make_dataset(len(Xtr) * f, len(Xte) * f, seed)
        new = slice(len(Xtr), None)
        # (1) accuracy en el test ORIGINAL usando train aumentado
        acc_orig = float(np.mean(predict_vector(Xa, ya, Xte, k) == yte))
        # (2) accuracy en test aumentado
        acc_aug = float(np.mean(predict_vector(Xa, ya, Xb, k) == yb))
        # (3) estadisticos por clase: medias por clase original vs sinteticas
        dmean = []
        for c in range(10):
            if f > 1:
                dmean.append(np.linalg.norm(Xa[new][ya[new] == c].mean(0) - Xtr[ytr == c].mean(0)))
        # (4) distancia al vecino mas cercano: original->original vs sintetico->original
        rows.append(dict(factor=f, n_train=len(Xa), n_test=len(Xb), acc_test_original=acc_orig,
                         acc_test_augmented=acc_aug,
                         max_classmean_shift=float(max(dmean)) if dmean else 0.0,
                         global_mean_orig=float(Xtr.mean()), global_mean_aug=float(Xa.mean()),
                         global_std_orig=float(Xtr.std()), global_std_aug=float(Xa.std()),
                         class_balance_aug=np.bincount(ya, minlength=10).tolist()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=0, help='0 = load_digits original')
    ap.add_argument('--n-test', type=int, default=0)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--modes', default='original,loop,vector')
    ap.add_argument('--kernel', default='gemm', choices=['gemm', 'direct'])
    ap.add_argument('--check-augment', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    if a.check_augment:
        for r in kc_rows(a):
            print(json.dumps(r))
        return

    if a.n_train:
        Xtr, Xte, ytr, yte = kc.make_dataset(a.n_train, a.n_test, a.seed)
    else:
        Xtr, Xte, ytr, yte = kc.load_split()
    preds = {}
    for mode in a.modes.split(','):
        yp, rec = run(Xtr, ytr, Xte, yte, a.k, mode, a.kernel)
        preds[mode] = yp
        print(json.dumps(rec))
        if a.out:
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
    ms = list(preds)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            diff = np.flatnonzero(preds[ms[i]] != preds[ms[j]])
            print(f'# {ms[i]} vs {ms[j]}: {len(diff)} predicciones distintas {diff.tolist()[:20]}')


def kc_rows(a):
    return check_augment(a.k, a.seed)


if __name__ == '__main__':
    main()
