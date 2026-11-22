"""Nucleo compartido del proyecto KNN (CS4052 2026-II).

Contiene:
  * carga/particion de load_digits identica a knn_digits_sec.py
  * generador reproducible de datos escalados (aumentacion controlada)
  * kernels de distancia euclidiana (formula directa y forma GEMM exacta)
  * top-k local con desempate determinista (distancia, indice global)
  * fusion de candidatos y voto por mayoria simple
  * particion por bloques con n no divisible entre p
  * conteo de FLOP de la region paralelizable

Convencion de desempate (usada en TODAS las versiones):
  los vecinos se ordenan por (distancia^2, indice global de entrenamiento).
  El voto replica collections.Counter.most_common(1): a igual numero de votos
  gana la etiqueta que aparece primero en la lista ordenada de vecinos.
"""
from collections import Counter

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

D_FEAT = 64
PIX_MAX = 16.0


# ---------------------------------------------------------------- datos
def load_split(test_size=0.2, random_state=42):
    """Particion exacta del codigo secuencial base."""
    digits = load_digits()
    return train_test_split(digits.data, digits.target,
                            test_size=test_size, random_state=random_state)


def _shift(imgs, dy, dx):
    """Desplaza imagenes 8x8 (dy, dx) pixeles con relleno 0 (fondo)."""
    out = np.zeros_like(imgs)
    h, w = imgs.shape[1:]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[:, yd, xd] = imgs[:, ys, xs]
    return out


def augment(X, y, n_out, seed, sigma=1.0, p_shift=0.0):
    """Genera n_out muestras derivadas de (X, y) de forma reproducible.

    Las primeras min(n_out, len(X)) son los datos originales sin cambios.
    Cada muestra adicional toma un original (recorrido estratificado ciclico),
    con probabilidad p_shift lo desplaza 1 pixel en una de 8 direcciones,
    suma ruido gaussiano N(0, sigma^2), redondea a entero y recorta a [0, 16].
    La etiqueta se conserva. Los valores siguen siendo enteros en [0,16]
    (mismo dominio que load_digits), lo que hace exacta la aritmetica float64.
    """
    rng = np.random.default_rng(seed)
    n0 = len(X)
    if n_out <= n0:
        return X[:n_out].copy(), y[:n_out].copy()
    extra = n_out - n0
    src = np.arange(extra) % n0
    rng.shuffle(src)
    imgs = X[src].reshape(-1, 8, 8).astype(np.float64)
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    do_shift = rng.random(extra) < p_shift
    which = rng.integers(0, 8, extra)
    for t, (dy, dx) in enumerate(dirs):
        sel = do_shift & (which == t)
        if sel.any():
            imgs[sel] = _shift(imgs[sel], dy, dx)
    imgs = imgs.reshape(extra, -1) + rng.normal(0.0, sigma, (extra, X.shape[1]))
    imgs = np.clip(np.rint(imgs), 0, PIX_MAX)
    return np.vstack([X, imgs]), np.concatenate([y, y[src]])


def make_dataset(n_train, n_test, seed=2026, sigma=1.0, p_shift=0.0):
    """Dataset escalado: train derivado SOLO de X_train original y test SOLO
    de X_test original (sin fuga de informacion entre particiones)."""
    Xtr, Xte, ytr, yte = load_split()
    Xtr2, ytr2 = augment(Xtr, ytr, n_train, seed, sigma, p_shift)
    Xte2, yte2 = augment(Xte, yte, n_test, seed + 1, sigma, p_shift)
    return Xtr2, Xte2, ytr2, yte2


# ---------------------------------------------------------------- particion
def block_counts(n, p):
    """Tamanos de bloque: los primeros n % p procesos reciben un elemento mas."""
    q, r = divmod(n, p)
    counts = np.array([q + 1 if i < r else q for i in range(p)], dtype=np.int64)
    displs = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return counts, displs


# ---------------------------------------------------------------- distancias
def euclidean_distance(a, b):
    """Formula del codigo base: sqrt(sum((a-b)^2)) -> 3d FLOP."""
    return np.sqrt(np.sum((a - b) ** 2))


def sqdist_direct(Xq, Xr, block_bytes=128 * 2**20):
    """||q - x||^2 por la formula directa (resta, cuadrado, suma), por bloques."""
    m, n = len(Xq), len(Xr)
    out = np.empty((m, n), dtype=np.float64)
    b = max(1, int(block_bytes // max(1, n * Xq.shape[1] * 8)))
    for i in range(0, m, b):
        diff = Xq[i:i + b, None, :] - Xr[None, :, :]
        out[i:i + b] = np.einsum('ijk,ijk->ij', diff, diff)
    return out


def sqdist_gemm(Xq, Xr):
    """||q||^2 + ||x||^2 - 2 q.x (BLAS). Exacto para datos enteros en [0,16]
    con d=64 (todas las cantidades son enteros < 2^53)."""
    qq = np.einsum('ij,ij->i', Xq, Xq)[:, None]
    rr = np.einsum('ij,ij->i', Xr, Xr)[None, :]
    D = qq + rr - 2.0 * (Xq @ Xr.T)
    np.maximum(D, 0.0, out=D)
    return D


# ---------------------------------------------------------------- top-k
def local_topk(Xq, Xr, yr, gidx, k, kernel='gemm', block_rows=None):
    """k vecinos locales por consulta.

    Devuelve (d2, lab, gid) de forma (m, k) ordenados por (d2, gid).
    Usa argpartition O(n) sobre una clave entera unica d2*G + gid.
    """
    m, n = len(Xq), len(Xr)
    kk = min(k, n)
    d2_out = np.full((m, k), np.inf)
    lab_out = np.full((m, k), -1, dtype=np.int64)
    gid_out = np.full((m, k), np.iinfo(np.int64).max, dtype=np.int64)
    if n == 0:
        return d2_out, lab_out, gid_out
    G = np.int64(int(gidx.max()) + 1)
    if block_rows is None:  # ~64 MB de matriz de distancias por bloque
        block_rows = max(1, int(8 * 2**20 // n))
    fn = sqdist_gemm if kernel == 'gemm' else sqdist_direct
    for i in range(0, m, block_rows):
        D = fn(Xq[i:i + block_rows], Xr)
        key = np.rint(D).astype(np.int64) * G + gidx[None, :]
        if kk < n:
            idx = np.argpartition(key, kk - 1, axis=1)[:, :kk]
        else:
            idx = np.broadcast_to(np.arange(n), (len(D), n))
        ksel = np.take_along_axis(key, idx, 1)
        order = np.argsort(ksel, axis=1)
        idx = np.take_along_axis(idx, order, 1)
        rows = slice(i, i + len(D))
        d2_out[rows, :kk] = np.take_along_axis(D, idx, 1)
        lab_out[rows, :kk] = yr[idx]
        gid_out[rows, :kk] = gidx[idx]
    return d2_out, lab_out, gid_out


def merge_topk(d2_list, lab_list, gid_list, k):
    """Fusiona p*k candidatos por consulta -> k minimos globales por (d2, gid)."""
    d2 = np.concatenate(d2_list, axis=1)
    lab = np.concatenate(lab_list, axis=1)
    gid = np.concatenate(gid_list, axis=1)
    order = _lexsort_rows(d2, gid)[:, :k]
    return (np.take_along_axis(d2, order, 1), np.take_along_axis(lab, order, 1),
            np.take_along_axis(gid, order, 1))


def _lexsort_rows(d2, gid):
    o1 = np.argsort(gid, axis=1, kind='stable')
    d2s = np.take_along_axis(d2, o1, 1)
    o2 = np.argsort(d2s, axis=1, kind='stable')
    return np.take_along_axis(o1, o2, 1)


def majority_vote(labels, n_classes=10):
    """Mayoria simple por fila, empate -> etiqueta que aparece primero
    (identico a Counter(labels).most_common(1))."""
    m, k = labels.shape
    onehot = labels[:, :, None] == np.arange(n_classes)[None, None, :]
    counts = onehot.sum(1)
    best = counts.max(1, keepdims=True)
    cand = counts == best                       # etiquetas empatadas
    first = np.where(onehot, np.arange(k)[None, :, None], k).min(1)
    first = np.where(cand, first, k + 1)
    return np.argmin(first, axis=1)


def vote_counter(labels):
    return np.array([Counter(row.tolist()).most_common(1)[0][0] for row in labels])


# ---------------------------------------------------------------- FLOP
def flops_distance(d=D_FEAT):
    """FLOP de la distancia euclidiana entre dos puntos en R^d:
    d restas + d cuadrados + (d-1) sumas + 1 raiz = 3d (todas pesan 1, U6.2)."""
    return 3 * d


def flops_parallel_region(n_train, n_test, d=D_FEAT):
    return flops_distance(d) * n_train * n_test
