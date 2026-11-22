"""BETA 3 - Version MPI optimizada e instrumentada (experimentos en Khipu).

Mejoras respecto a beta2:
  * comunicacion por buffers contiguos NumPy (Scatterv / Bcast / Gather)
    o por objetos pickle (--comm pickle) para comparar
  * distancias vectorizadas por bloques (kernel gemm exacto o formula directa)
  * seleccion top-k con argpartition O(n) en lugar de argsort O(n log n)
  * los k candidatos (dist^2, etiqueta, idx) se empaquetan en UN mensaje
  * medicion con MPI.Wtime() y barreras: T_total, T_comp, T_comm, T_idle
  * salida CSV (una fila por repeticion)

Variantes de descomposicion (--variant):
  dag       : Scatter(X_tr, y_tr), Bcast(X_te), Gather(k candidatos)  <- enunciado
  replicate : Bcast(X_tr, y_tr), Scatter(X_te), Gather(y_pred)        <- parcial

Ejecutar:
  mpirun -np 8 python src/beta3_mpi_optimizado.py --n-train 92000 --n-test 3600 \
        --reps 5 --out resultados/strong.csv --tag strong
"""
import argparse
import csv
import os
import socket
import sys
import time

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_core as kc  # noqa: E402

FIELDS = ['tag', 'version', 'variant', 'comm_mode', 'kernel', 'rep', 'p', 'nodes', 'N',
          'n_train', 'n_test', 'd', 'k', 'T_total', 'T_comp', 'T_comp_local_max',
          'T_comp_local_min', 'T_vote', 'T_comm', 'T_scatter', 'T_bcast', 'T_gather',
          'T_idle', 'accuracy', 'total_flops', 'gflops_per_sec', 'gflops_per_sec_comp',
          'checksum', 'timestamp']


# --------------------------------------------------------------- variante DAG
def run_dag(comm, Xtr, ytr, Xte, k, kernel, comm_mode):
    rank, p = comm.Get_rank(), comm.Get_size()
    T = {}
    comm.Barrier()
    t0 = MPI.Wtime()

    # ---- Scatter de X_tr, y_tr (bloques n_tr/p, resto repartido)
    if comm_mode == 'pickle':
        chunks = None
        if rank == 0:
            counts, displs = kc.block_counts(len(Xtr), p)
            chunks = [(Xtr[o:o + c], ytr[o:o + c], o) for c, o in zip(counts, displs)]
        Xl, yl, off = comm.scatter(chunks, root=0)
        meta = None
    else:
        meta = comm.bcast((len(Xtr), Xtr.shape[1], len(Xte)) if rank == 0 else None, root=0)
        n_tr, d, n_te = meta
        counts, displs = kc.block_counts(n_tr, p)
        Xl = np.empty((counts[rank], d), dtype=np.float64)
        yl = np.empty(counts[rank], dtype=np.int64)
        sx = [Xtr, counts * d, displs * d, MPI.DOUBLE] if rank == 0 else None
        sy = [ytr.astype(np.int64), counts, displs, MPI.INT64_T] if rank == 0 else None
        comm.Scatterv(sx, Xl, root=0)
        comm.Scatterv(sy, yl, root=0)
        off = int(displs[rank])
    t1 = MPI.Wtime()

    # ---- Bcast de X_te
    if comm_mode == 'pickle':
        Xte = comm.bcast(Xte, root=0)
    else:
        if rank != 0:
            Xte = np.empty((meta[2], meta[1]), dtype=np.float64)
        comm.Bcast(Xte, root=0)
    t2 = MPI.Wtime()

    # ---- pardo: distancias + top-k local
    gid = np.arange(off, off + len(Xl), dtype=np.int64)
    d2, lab, gg = kc.local_topk(Xte, Xl, yl, gid, k, kernel=kernel)
    t3 = MPI.Wtime()
    comm.Barrier()                          # espera por desbalance -> T_idle
    t4 = MPI.Wtime()

    # ---- Gather de k candidatos por consulta (un solo mensaje empaquetado)
    pack = np.stack([d2, lab.astype(np.float64), gg.astype(np.float64)], axis=-1)
    if comm_mode == 'pickle':
        allp = comm.gather(pack, root=0)
    else:
        allp = np.empty((p,) + pack.shape, dtype=np.float64) if rank == 0 else None
        comm.Gather(np.ascontiguousarray(pack), allp, root=0)
    t5 = MPI.Wtime()

    # ---- rank 0: k minimos globales + mayoria simple
    y_pred = None
    if rank == 0:
        allp = np.asarray(allp)
        D2, L, G = kc.merge_topk([allp[r, :, :, 0] for r in range(p)],
                                 [allp[r, :, :, 1].astype(np.int64) for r in range(p)],
                                 [allp[r, :, :, 2].astype(np.int64) for r in range(p)], k)
        y_pred = kc.majority_vote(L)
    t6 = MPI.Wtime()
    T.update(T_scatter=t1 - t0, T_bcast=t2 - t1, T_local=t3 - t2, T_idle=t4 - t3,
             T_gather=t5 - t4, T_vote=t6 - t5, T_total=t6 - t0)
    return y_pred, T


# --------------------------------------------------------- variante replicate
def run_replicate(comm, Xtr, ytr, Xte, k, kernel, comm_mode):
    rank, p = comm.Get_rank(), comm.Get_size()
    T = {}
    comm.Barrier()
    t0 = MPI.Wtime()
    if comm_mode == 'pickle':
        Xtr, ytr = comm.bcast((Xtr, ytr) if rank == 0 else None, root=0)
    else:
        meta = comm.bcast((len(Xtr), Xtr.shape[1], len(Xte)) if rank == 0 else None, root=0)
        if rank != 0:
            Xtr = np.empty(meta[:2]); ytr = np.empty(meta[0], dtype=np.int64)
        else:
            ytr = ytr.astype(np.int64)
        comm.Bcast(Xtr, root=0); comm.Bcast(ytr, root=0)
    t1 = MPI.Wtime()
    if comm_mode == 'pickle':
        chunks = None
        if rank == 0:
            c, o = kc.block_counts(len(Xte), p)
            chunks = [Xte[a:a + b] for b, a in zip(c, o)]
        Xq = comm.scatter(chunks, root=0)
    else:
        c, o = kc.block_counts(meta[2], p)
        Xq = np.empty((c[rank], meta[1]))
        comm.Scatterv([Xte, c * meta[1], o * meta[1], MPI.DOUBLE] if rank == 0 else None, Xq, root=0)
    t2 = MPI.Wtime()
    gid = np.arange(len(Xtr), dtype=np.int64)
    _, lab, _ = kc.local_topk(Xq, Xtr, ytr, gid, k, kernel=kernel)
    yq = kc.majority_vote(lab)
    t3 = MPI.Wtime()
    comm.Barrier()
    t4 = MPI.Wtime()
    if comm_mode == 'pickle':
        parts = comm.gather(yq, root=0)
        y_pred = np.concatenate(parts) if rank == 0 else None
    else:
        y_pred = np.empty(meta[2], dtype=np.int64) if rank == 0 else None
        comm.Gatherv(yq.astype(np.int64), [y_pred, c, o, MPI.INT64_T] if rank == 0 else None, root=0)
    t5 = MPI.Wtime()
    # en esta variante: "scatter" del CSV = Scatter(X_te), "bcast" = Bcast(X_tr)
    T.update(T_bcast=t1 - t0, T_scatter=t2 - t1, T_local=t3 - t2, T_idle=t4 - t3,
             T_gather=t5 - t4, T_vote=0.0, T_total=t5 - t0)
    return y_pred, T


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=0, help='0 = load_digits original')
    ap.add_argument('--n-test', type=int, default=0)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--kernel', default='gemm', choices=['gemm', 'direct'])
    ap.add_argument('--comm', dest='comm_mode', default='buffer', choices=['buffer', 'pickle'])
    ap.add_argument('--variant', default='dag', choices=['dag', 'replicate'])
    ap.add_argument('--out', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--save-pred', default='', help='guarda y_pred (.npy) para validacion')
    a = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, p = comm.Get_rank(), comm.Get_size()
    nodes = len(set(comm.allgather(socket.gethostname())))
    Xtr = Xte = ytr = yte = None
    if rank == 0:                                   # generacion fuera del tiempo medido
        if a.n_train:
            Xtr, Xte, ytr, yte = kc.make_dataset(a.n_train, a.n_test, a.seed)
        else:
            Xtr, Xte, ytr, yte = kc.load_split()
        Xtr = np.ascontiguousarray(Xtr, dtype=np.float64)
        Xte = np.ascontiguousarray(Xte, dtype=np.float64)
    run = run_dag if a.variant == 'dag' else run_replicate

    for rep in range(-a.warmup, a.reps):
        y_pred, T = run(comm, Xtr, ytr, Xte, a.k, a.kernel, a.comm_mode)
        loc_max = comm.reduce(T['T_local'], op=MPI.MAX, root=0)
        loc_min = comm.reduce(T['T_local'], op=MPI.MIN, root=0)
        if rank != 0 or rep < 0:
            continue
        acc = float(np.mean(y_pred == yte))
        fl = kc.flops_parallel_region(len(Xtr), len(Xte), Xtr.shape[1])
        T_comm = T['T_scatter'] + T['T_bcast'] + T['T_gather']
        T_comp = loc_max + T['T_vote']
        row = dict(tag=a.tag, version='beta3', variant=a.variant, comm_mode=a.comm_mode,
                   kernel=a.kernel, rep=rep, p=p, nodes=nodes, N=len(Xtr) + len(Xte),
                   n_train=len(Xtr), n_test=len(Xte), d=Xtr.shape[1], k=a.k,
                   T_total=T['T_total'], T_comp=T_comp, T_comp_local_max=loc_max,
                   T_comp_local_min=loc_min, T_vote=T['T_vote'], T_comm=T_comm,
                   T_scatter=T['T_scatter'], T_bcast=T['T_bcast'], T_gather=T['T_gather'],
                   T_idle=T['T_idle'], accuracy=acc, total_flops=fl,
                   gflops_per_sec=fl / 1e9 / T['T_total'], gflops_per_sec_comp=fl / 1e9 / loc_max,
                   checksum=int(np.sum(y_pred * np.arange(1, len(y_pred) + 1)) % 1000003),
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))
        print(','.join(f'{k_}={row[k_]:.5g}' if isinstance(row[k_], float) else f'{k_}={row[k_]}'
                       for k_ in ['p', 'n_train', 'n_test', 'rep', 'T_total', 'T_comp',
                                  'T_comm', 'T_idle', 'accuracy']), flush=True)
        if a.out:
            new = not os.path.exists(a.out)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
        if a.save_pred and rep == 0:
            np.save(a.save_pred, y_pred)


if __name__ == '__main__':
    main()
