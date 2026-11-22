"""Diagnostico: ¿el computo local esta limitado por ancho de banda de memoria?

Cada rank ejecuta el MISMO trabajo local (escalabilidad debil pura, sin
comunicacion de datos): local_topk de n_te x n_loc. Si el tiempo por rank
crece con p, hay contencion de recursos compartidos del nodo (memoria/L3/
frecuencia), no de MPI. Ademas registra el core y socket (NUMA) de cada rank.

  mpiexec -bind-to core -n 8 python src/diag_memoria.py --tag core
"""
import argparse
import csv
import os
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_core as kc  # noqa: E402


def socket_of(cpu):
    try:
        with open(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id') as f:
            return int(f.read())
    except OSError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-loc', type=int, default=11496)
    ap.add_argument('--n-test', type=int, default=3600)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank, p = comm.Get_rank(), comm.Get_size()
    Xtr, Xte, ytr, _ = kc.make_dataset(a.n_loc, a.n_test)       # mismo dato en cada rank
    gid = np.arange(a.n_loc, dtype=np.int64)
    cpus = sorted(os.sched_getaffinity(0))
    sock = sorted({socket_of(c) for c in cpus})
    for rep in range(-1, a.reps):
        comm.Barrier()
        t0 = MPI.Wtime()
        D = kc.sqdist_gemm(Xte, Xtr)                               # solo GEMM (compute-bound)
        t1 = MPI.Wtime()
        kc.local_topk(Xte, Xtr, ytr, gid, 3)                       # top-k completo
        t2 = MPI.Wtime()
        del D
        rows = comm.gather((rank, cpus[0], sock[0], t1 - t0, t2 - t1), root=0)
        if rank == 0 and rep >= 0:
            t_g = [r[3] for r in rows]; t_k = [r[4] for r in rows]
            socks = [r[2] for r in rows]
            line = dict(tag=a.tag, p=p, rep=rep, n_loc=a.n_loc, n_test=a.n_test,
                        t_gemm_mean=np.mean(t_g), t_gemm_max=max(t_g),
                        t_topk_mean=np.mean(t_k), t_topk_max=max(t_k),
                        ranks_socket0=socks.count(0), ranks_socket1=socks.count(1),
                        cpus=';'.join(str(r[1]) for r in rows))
            print(line, flush=True)
            if a.out:
                new = not os.path.exists(a.out)
                with open(a.out, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(line))
                    if new:
                        w.writeheader()
                    w.writerow(line)


if __name__ == '__main__':
    main()
