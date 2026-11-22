"""Suite de validacion: igualdad de predicciones beta1 / beta2 / beta3 / sklearn.

Modo local (sin MPI, simula la particion por bloques para p = 1..17):
    python tests/validate_knn.py
Modo MPI real (Khipu, cualquier p):
    mpirun -np 7 python tests/validate_knn.py --mpi

Referencia: beta1 'original' (copia literal de knn_digits_sec.py).
sklearn KNeighborsClassifier(algorithm='brute') se compara aparte: su voto
desempata por la etiqueta MENOR (scipy mode) y no por el vecino mas cercano,
por lo que cualquier diferencia se reporta y se clasifica (empate de votos /
empate de distancias) en lugar de ocultarse.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import knn_core as kc  # noqa: E402
import beta1_secuencial as b1  # noqa: E402

FAIL = []


def check(name, cond, extra=''):
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        FAIL.append(name)


def simulated_dag(Xtr, ytr, Xte, k, p, kernel):
    """Misma secuencia que beta3/dag pero con p rangos simulados."""
    counts, displs = kc.block_counts(len(Xtr), p)
    parts = []
    for c, o in zip(counts, displs):
        gid = np.arange(o, o + c, dtype=np.int64)
        parts.append(kc.local_topk(Xte, Xtr[o:o + c], ytr[o:o + c], gid, k, kernel=kernel))
    _, L, _ = kc.merge_topk([q[0] for q in parts], [q[1] for q in parts], [q[2] for q in parts], k)
    return kc.majority_vote(L)


def tie_report(Xtr, ytr, Xte, k):
    D = kc.sqdist_gemm(Xte, Xtr)
    s = np.sort(D, 1)
    return np.flatnonzero(s[:, k - 1] == s[:, k])


def sklearn_compare(Xtr, ytr, Xte, ref, k):
    from sklearn.neighbors import KNeighborsClassifier
    sk = KNeighborsClassifier(n_neighbors=k, algorithm='brute').fit(Xtr, ytr).predict(Xte)
    diff = np.flatnonzero(sk != ref)
    ties = set(tie_report(Xtr, ytr, Xte, k).tolist())
    gid = np.arange(len(Xtr), dtype=np.int64)
    _, L, _ = kc.local_topk(Xte, Xtr, ytr, gid, k)
    vote_ties = {j for j in diff if np.bincount(L[j], minlength=10).max() < (k + 1) // 2 + (k % 2 == 0)}
    unexplained = [j for j in diff if j not in ties and j not in vote_ties]
    return sk, diff, ties, vote_ties, unexplained


def local_suite(k):
    Xtr, Xte, ytr, yte = kc.load_split()
    ties = tie_report(Xtr, ytr, Xte, k)
    print(f'# load_digits: n_train={len(Xtr)} n_test={len(Xte)} k={k}; '
          f'consultas con empate de distancia en la frontera k: {ties.tolist()}')
    ref, r0 = b1.run(Xtr, ytr, Xte, yte, k, 'original')
    check('accuracy original == 0.9833', abs(r0['accuracy'] - 0.98333333) < 1e-6, f"({r0['accuracy']:.4f})")
    for mode, kern in [('loop', 'loop'), ('vector', 'gemm'), ('vector', 'direct')]:
        yp, _ = b1.run(Xtr, ytr, Xte, yte, k, mode, kern)
        check(f'beta1 {mode}/{kern} == original', np.array_equal(yp, ref))
    for kern in ['gemm', 'direct']:
        for p in list(range(1, 18)) + [32, 64]:
            yp = simulated_dag(Xtr, ytr, Xte, k, p, kern)
            check(f'DAG simulado p={p:2d} ({kern}, n_tr%p={len(Xtr) % p})', np.array_equal(yp, ref))
    # caso borde: bloques con menos de k elementos
    yp = simulated_dag(Xtr[:20], ytr[:20], Xte, k, 16, 'gemm')
    yr, _ = b1.run(Xtr[:20], ytr[:20], Xte, yte, k, 'loop')
    check('bloques con < k puntos (n_tr=20, p=16)', np.array_equal(yp, yr))
    # vote helper == Counter
    rng = np.random.default_rng(0)
    labs = rng.integers(0, 10, (5000, 5))
    check('majority_vote == Counter.most_common', np.array_equal(kc.majority_vote(labs), kc.vote_counter(labs)))
    # gemm exacto vs formula directa (datos enteros)
    Xa, Xb, ya, yb = kc.make_dataset(4000, 500)
    check('gemm == direct exacto (datos aumentados)',
          np.array_equal(kc.sqdist_gemm(Xb, Xa), kc.sqdist_direct(Xb, Xa)))
    yl, _ = b1.run(Xa, ya, Xb, yb, k, 'loop')
    for p in [3, 7, 13]:
        check(f'aumentado n_tr=4000 DAG p={p} == loop', np.array_equal(simulated_dag(Xa, ya, Xb, k, p, 'gemm'), yl))
    # sklearn
    sk, diff, ties_s, vt, unexp = sklearn_compare(Xtr, ytr, Xte, ref, k)
    print(f'# sklearn brute: accuracy={np.mean(sk == yte):.4f}; difieren {len(diff)} consultas {diff.tolist()} '
          f'(empate distancias: {sorted(set(diff) & ties_s)}, empate votos: {sorted(vt)})')
    check('sklearn: toda diferencia explicada por empates', len(unexp) == 0, f'no explicadas={unexp}')


def mpi_suite(k):
    from mpi4py import MPI
    import beta2_mpi_basico as b2
    import beta3_mpi_optimizado as b3
    comm = MPI.COMM_WORLD
    rank, p = comm.Get_rank(), comm.Get_size()
    cases = [('digits', None)] + [('aug', (5003, 997))]   # n no divisibles
    for name, sz in cases:
        Xtr = Xte = ytr = yte = ref = None
        if rank == 0:
            Xtr, Xte, ytr, yte = kc.load_split() if sz is None else kc.make_dataset(*sz)
            ref, _ = b1.run(Xtr, ytr, Xte, yte, k, 'original' if sz is None else 'loop')
        y2 = b2.knn_mpi_basic(comm, Xtr, ytr, Xte, k)
        res = {}
        for var in ['dag', 'replicate']:
            for cm in ['buffer', 'pickle']:
                res[(var, cm)] = b3.run_dag(comm, Xtr, ytr, Xte, k, 'gemm', cm)[0] if var == 'dag' \
                    else b3.run_replicate(comm, Xtr, ytr, Xte, k, 'gemm', cm)[0]
        if rank == 0:
            check(f'[MPI p={p}] {name} beta2 == beta1', np.array_equal(y2, ref))
            for key, y3 in res.items():
                check(f'[MPI p={p}] {name} beta3 {key} == beta1', np.array_equal(y3, ref))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mpi', action='store_true')
    ap.add_argument('--k', type=int, default=3)
    a = ap.parse_args()
    mpi_suite(a.k) if a.mpi else local_suite(a.k)
    if not a.mpi or __import__('mpi4py.MPI', fromlist=['x']).COMM_WORLD.Get_rank() == 0:
        print(f'\nRESULTADO: {"TODO OK" if not FAIL else "FALLOS: " + str(FAIL)}')
    sys.exit(1 if FAIL else 0)
