"""BETA 2 - Primera paralelizacion MPI (mpi4py, objetos Python) segun el DAG.

  rank 0 : particion de X_train / y_train en p bloques (n_train no divisible ok)
  comm.scatter : bloque (X_tr^(r), y_tr^(r), indices globales) a cada rank r
  comm.bcast   : X_test completo a todos
  pardo        : cada rank calcula distancias euclidianas (formula del codigo
                 base, bucle Python) y sus k vecinos locales por consulta
  comm.gather  : p listas de k candidatos (dist, etiqueta, idx) por consulta
  rank 0       : k minimos globales + mayoria simple (Counter)

Ejecutar:  mpirun -np 4 python src/beta2_mpi_basico.py [--n-train N --n-test M]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_core as kc  # noqa: E402


def local_candidates(Xtr_loc, ytr_loc, gid_loc, Xte, k):
    """Para cada consulta: lista de k tuplas (dist, gid, label) locales."""
    out = []
    for q in Xte:
        dist = np.array([kc.euclidean_distance(q, x) for x in Xtr_loc])
        order = np.lexsort((gid_loc, dist))[:k]          # (dist, gid)
        out.append([(float(dist[i]), int(gid_loc[i]), int(ytr_loc[i])) for i in order])
    return out


def knn_mpi_basic(comm, Xtr, ytr, Xte, k):
    rank, p = comm.Get_rank(), comm.Get_size()
    chunks = None
    if rank == 0:
        counts, displs = kc.block_counts(len(Xtr), p)
        chunks = [(Xtr[o:o + c], ytr[o:o + c], np.arange(o, o + c)) for c, o in zip(counts, displs)]
    Xtr_loc, ytr_loc, gid_loc = comm.scatter(chunks, root=0)
    Xte = comm.bcast(Xte, root=0)
    cand = local_candidates(Xtr_loc, ytr_loc, gid_loc, Xte, k)
    allc = comm.gather(cand, root=0)
    if rank != 0:
        return None
    y_pred = np.empty(len(Xte), dtype=np.int64)
    for j in range(len(Xte)):
        pool = sorted(c for r in range(p) for c in allc[r][j])[:k]  # orden (dist, gid)
        y_pred[j] = Counter(lab for _, _, lab in pool).most_common(1)[0][0]
    return y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=0)
    ap.add_argument('--n-test', type=int, default=0)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Xtr = Xte = ytr = yte = None
    if rank == 0:
        if a.n_train:
            Xtr, Xte, ytr, yte = kc.make_dataset(a.n_train, a.n_test, a.seed)
        else:
            Xtr, Xte, ytr, yte = kc.load_split()
    comm.Barrier()
    t0 = MPI.Wtime()
    y_pred = knn_mpi_basic(comm, Xtr, ytr, Xte, a.k)
    if rank == 0:
        acc = float(np.mean(y_pred == yte))
        t = MPI.Wtime() - t0
        print(json.dumps(dict(version='beta2', p=comm.Get_size(), n_train=len(Xtr),
                              n_test=len(Xte), k=a.k, T_total=t, accuracy=acc)))


if __name__ == '__main__':
    main()
