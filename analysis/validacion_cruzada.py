"""Validacion fuera de muestra del modelo extendido (dejar una serie fuera).

  python analysis/validacion_cruzada.py resultados/khipu_<jobid>

Para cada serie fuerte (digits, strong_small, strong):
  1. ajusta c1' y c2 (NNLS ponderado por 1/T_comp, igual que analyze.fit_extended)
     usando SOLO las otras dos series;
  2. predice T_total de la serie excluida con
        T = c1' * (n_tr n_te d / p) * phi(p) + c2 * n_te k p + alpha M(p) + beta B(p);
  3. reporta R^2 y error relativo medio de esa prediccion.

alpha, beta y phi(p) se leen de <dir>/tablas/reporte.json (salida de analyze.py):
alpha y beta salen de T_comm y phi de la serie debil, asi que ninguno usa el
T_comp de la serie excluida. No importa matplotlib.
Escribe <dir>/tablas/validacion_cruzada.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

STRONG = ['digits', 'strong_small', 'strong']
D, K = 64, 3


def features(p, ntr, nte, P, PHI):
    """Mismos regresores que analyze.features / predict_ext."""
    p = np.asarray(p, float)
    lg = np.log2(p)
    phi = np.interp(np.log(p), np.log(PHI['p']), PHI['phi'])
    M = 2 * (2 * p - 1) + lg + (2 * p - 1)
    B = (2 * p - 1) / p * ntr * (D + 1) * 8 + lg * nte * D * 8 + (2 * p - 1) * 3 * K * nte * 8
    return ntr * nte * D / p * phi, nte * K * p, P['alpha'] * M + P['beta'] * B


def load_series(d):
    out = {}
    for tag in STRONG:
        x = pd.read_csv(os.path.join(d, f'{tag}.csv'))
        x = x[(x.tag == tag) & (x.variant == 'dag') & (x.comm_mode == 'buffer') & (x.kernel == 'gemm')]
        out[tag] = x.groupby('p').agg(T=('T_total', 'mean'), C=('T_comp', 'mean'),
                                      ntr=('n_train', 'first'), nte=('n_test', 'first')).reset_index()
    return out


def main(d):
    rep = json.load(open(os.path.join(d, 'tablas', 'reporte.json')))
    P, PHI = rep['parametros_ajustados'], rep['phi_contencion']
    S = load_series(d)
    rows = []
    for held in STRONG:
        tr = pd.concat([S[s] for s in STRONG if s != held])
        a, b, _ = features(tr.p, tr.ntr.values, tr.nte.values, P, PHI)
        w = 1 / tr.C.values
        (c1, c2), _ = nnls(np.c_[a, b] * w[:, None], tr.C.values * w)
        s = S[held]
        a, b, comm = features(s.p, s.ntr.values, s.nte.values, P, PHI)
        pred = c1 * a + c2 * b + comm
        y = s['T'].values
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        err = float(np.mean(np.abs(pred - y) / y))
        rows.append(dict(serie_excluida=held, c1e=c1, c2e=c2, R2_prediccion=r2, err_rel_medio=err))
        print(f'{held:13s} c1\'={c1:.3e}  R2={r2:.3f}  error medio={100 * err:.1f} %')
    pd.DataFrame(rows).to_csv(os.path.join(d, 'tablas', 'validacion_cruzada.csv'), index=False)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'resultados/khipu_64271')
