"""Figuras y numeros del informe final.

  python analysis/figs_informe.py resultados/khipu_64271 informe/figs

Lee los CSV de Khipu (resultados/khipu_64271 y resultados/khipu_ht_64272) y genera:
  tiempos_vs_p.pdf, speedup_eficiencia.pdf, gflops_vs_p.pdf, escalabilidad.pdf
e imprime las constantes del modelo normalizado y p* = sqrt(A/B).
Requiere numpy, pandas, scipy y matplotlib.

Modelo teorico (U6.1 para la comunicacion):
  T_p = A/p + B p + C log2 p + D
  A = c1 n_tr n_te d                     (distancias, se reparten entre p)
  B = c2 n_te k + 6 alpha + 48 k n_te beta  (fusion en P0 + Gather + arranques del Scatter)
  C = alpha + 8 n_te d beta              (Bcast de X_te)
  D = 16 n_tr (d+1) beta                 (bytes del Scatter de X_tr)
  c1: con la corrida secuencial (p=1) de las tres series.
  c2: con el tiempo medido de la fusion (T_vote).
  alpha, beta: minimos cuadrados (>=0) sobre T_comm medido (p>1).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

D_, K_ = 64, 3
W_IN = 6.3
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 9.5, 'legend.fontsize': 8.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .3, 'grid.linestyle': ':',
    'lines.linewidth': 1.4, 'lines.markersize': 4.5,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03, 'pdf.fonttype': 42,
})
SER = {'digits': dict(c='#0072B2', m='o', lab='load_digits ($n_{tr}=1437$)'),
       'strong_small': dict(c='#E69F00', m='s', lab='Mediano ($n_{tr}=22\\,992$)'),
       'strong': dict(c='#009E73', m='^', lab='Grande ($n_{tr}=91\\,968$)')}
PS = [1, 2, 4, 6, 8, 12, 16]


def logx(ax, ticks=PS):
    ax.set_xscale('log', base=2)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mt.FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(mt.NullLocator())


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.pdf'))
    plt.close(fig)


def load(d):
    """Lee <d>/*.csv y las series de hyper-threading hermanas (khipu_ht_*/)."""
    fs = glob.glob(os.path.join(d, '*.csv'))
    fs += glob.glob(os.path.join(os.path.dirname(os.path.abspath(d)), 'khipu_ht_*', '*.csv'))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def agg(df):
    """Media y desviacion de las repeticiones de cada configuracion."""
    keys = ['tag', 'variant', 'comm_mode', 'kernel', 'p', 'n_train', 'n_test', 'd', 'k']
    m = ['T_total', 'T_comp', 'T_comm', 'T_vote', 'T_comp_local_max', 'accuracy', 'gflops_per_sec']
    g = df.groupby(keys)[m].agg(['mean', 'std']).reset_index()
    g.columns = ['_'.join(c).rstrip('_') for c in g.columns]
    return g.sort_values(['tag', 'p'])


def fit_comm(s):
    """alpha, beta >= 0 por minimos cuadrados (relativos) de T_comm, formulas de U6.1."""
    s = s[s.p > 1]
    p = s.p.values.astype(float); lg = np.log2(p)
    ntr = s.n_train.values.astype(float); nte = s.n_test.values.astype(float)
    M = 2 * (2 * p - 1) + lg + (2 * p - 1)                                    # arranques
    B = (2 * p - 1) / p * ntr * (D_ + 1) * 8 + lg * nte * D_ * 8 + (2 * p - 1) * 3 * K_ * nte * 8  # bytes
    y = s.T_comm_mean.values; w = 1 / y
    ab, _ = nnls(np.c_[M, B] * w[:, None], y * w)
    return ab


def fit(base):
    s = base[base.tag.isin(SER)]
    # c1: corrida secuencial (p=1) de las tres series, minimos cuadrados relativos
    s1 = s[s.p == 1]
    x = s1.n_train.astype(float) * s1.n_test * D_
    y = s1.T_comp_local_max_mean.values
    w = 1 / y
    c1 = np.sum(w * w * x * y) / np.sum(w * w * x * x)
    # c2: tiempo de la fusion en P0, T_vote ~ c2 * n_te k p
    sv = s[s.p > 1]
    xv = sv.n_test.astype(float) * K_ * sv.p
    c2 = np.sum(xv * sv.T_vote_mean) / np.sum(xv * xv)
    # alpha, beta: comunicacion
    a, b = fit_comm(s)
    return dict(c1=c1, c2=c2, alpha=a, beta=b)


def coefs(P, ntr, nte):
    ntr, nte = float(ntr), float(nte)
    a, b = P['alpha'], P['beta']
    return dict(A=P['c1'] * ntr * nte * D_,
                B=P['c2'] * nte * K_ + 6 * a + 48 * K_ * nte * b,
                C=a + 8 * nte * D_ * b,
                D=16 * ntr * (D_ + 1) * b)


def model(P, p, ntr, nte):
    p = np.asarray(p, float)
    c = coefs(P, ntr, nte)
    comp = c['A'] / p + P['c2'] * float(nte) * K_ * p
    tot = c['A'] / p + c['B'] * p + c['C'] * np.log2(p) + c['D']
    return comp, tot - comp, tot


def main(d, out):
    os.makedirs(out, exist_ok=True)
    g = agg(load(d))
    base = g[(g.variant == 'dag') & (g.comm_mode == 'buffer') & (g.kernel == 'gemm')]
    P = fit(base)
    S_ = lambda t: base[base.tag == t].sort_values('p')  # noqa: E731
    log = dict(P={k: float(v) for k, v in P.items()}, gflops_core=3 / P['c1'] / 1e9)

    # ------------------------------------------------ tiempos vs p (3 paneles)
    fig, axs = plt.subplots(1, 3, figsize=(W_IN, 2.45))
    for ax, tag in zip(axs, SER):
        s = S_(tag); ntr, nte = s.n_train.iloc[0], s.n_test.iloc[0]
        sc, un = (1e3, 'ms') if tag == 'digits' else (1.0, 's')
        ax.plot(s.p, s.T_total_mean * sc, 'o-', color='black', label='Total (medido)')
        ax.plot(s.p, s.T_comp_mean * sc, 's-', color='#0072B2', label='Cómputo (medido)')
        ax.plot(s.p, s.T_comm_mean * sc, '^-', color='#D55E00', label='Comunicación (medido)')
        pp = np.geomspace(1, 16, 200)
        _, cm, tt = model(P, pp, ntr, nte)
        ax.plot(pp, tt * sc, '--', color='black', lw=1.1, label='Total (teórico)')
        ax.plot(pp, cm * sc, '--', color='#D55E00', lw=1.1, label='Comunicación (teórico)')
        ax.set_yscale('log'); logx(ax)
        ax.set_title(SER[tag]['lab'].replace(' (', '\n('), fontsize=9.5)
        ax.set_xlabel('$p$ (procesos)'); ax.set_ylabel(f'Tiempo [{un}]')
        _, _, tm = model(P, s.p.values, ntr, nte)
        log[f'ratio_{tag}'] = np.round(s.T_total_mean.values / tm, 2).tolist()
        c = coefs(P, ntr, nte)
        log[f'coefs_{tag}'] = {k: float(v) for k, v in c.items()}
        log[f'pstar_{tag}'] = float(np.sqrt(c['A'] / c['B']))
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3)
    fig.tight_layout(w_pad=1.0)
    save(fig, out, 'tiempos_vs_p')

    # ------------------------------------------------ speedup y eficiencia
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_IN, 2.6))
    for tag, st in SER.items():
        s = S_(tag)
        S = s.T_total_mean.iloc[0] / s.T_total_mean.values
        a1.plot(s.p, S, marker=st['m'], color=st['c'], label=st['lab'])
        a2.plot(s.p, S / s.p.values, marker=st['m'], color=st['c'])
    a1.plot([1, 16], [1, 16], ':', color='gray', label='Ideal ($S=p$)')
    a2.axhline(1, ls=':', color='gray')
    for ax in (a1, a2):
        logx(ax); ax.set_xlabel('$p$ (procesos)')
    a1.set_yscale('log', base=2); a1.set_yticks([1, 2, 4, 8, 16])
    a1.yaxis.set_major_formatter(mt.FixedFormatter(['1', '2', '4', '8', '16']))
    a1.yaxis.set_minor_locator(mt.NullLocator())
    a1.set_ylabel('Speedup $S=T_1/T_p$'); a1.set_title('Speedup')
    a2.set_ylim(0, 1.08); a2.set_ylabel('Eficiencia $E=S/p$'); a2.set_title('Eficiencia')
    h, l_ = a1.get_legend_handles_labels()
    fig.legend(h, l_, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4)
    fig.tight_layout(w_pad=2.0)
    save(fig, out, 'speedup_eficiencia')

    # ------------------------------------------------ GFLOP/s vs p
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for tag, st in SER.items():
        s = S_(tag)
        ax.plot(s.p, s.gflops_per_sec_mean, marker=st['m'], color=st['c'], label=st['lab'])
    logx(ax)
    ax.set_xlabel('$p$ (procesos)'); ax.set_ylabel('GFLOP/s')
    ax.set_ylim(0, 65)
    ax.legend(loc='upper left')
    fig.tight_layout()
    save(fig, out, 'gflops_vs_p')

    # ------------------------------------------------ escalabilidad con tamano variable
    w = S_('weak')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_IN, 2.7))
    a1.plot(w.p, w.T_total_mean, 'o-', color='black', label='Total')
    a1.plot(w.p, w.T_comp_local_max_mean, 's-', color='#0072B2', label='Cómputo')
    a1.plot(w.p, w.T_comm_mean, '^-', color='#D55E00', label='Comunicación')
    a1.axhline(w.T_total_mean.iloc[0], ls=':', color='gray', label='Ideal (constante)')
    logx(a1); a1.set_ylim(0, 3.3)
    a1.set_xlabel('$p$ (procesos), con $n_{tr}=11\\,496\\cdot p$'); a1.set_ylabel('Tiempo [s]')
    a1.set_title('(a) Mismo trabajo por proceso')
    a1.legend(loc='upper left', fontsize=8, ncol=2)
    allb = base[base.tag.isin(list(SER) + ['weak'])].copy()
    allb['F'] = 3.0 * D_ * allb.n_train.astype(float) * allb.n_test
    slopes = {}
    for p0, c, mk in [(1, 'black', 'o'), (4, '#0072B2', 's'), (16, '#D55E00', '^')]:
        s = allb[allb.p == p0].drop_duplicates(['n_train', 'n_test']).sort_values('F')
        big = s[s.F >= 1e10]
        sl = np.polyfit(np.log(big.F), np.log(big.T_total_mean), 1)[0]
        slopes[p0] = float(sl)
        a2.plot(s.F, s.T_total_mean, marker=mk, color=c, label=f'$p={p0}$ (pendiente {sl:.2f})')
    a2.set_xscale('log'); a2.set_yscale('log')
    a2.set_xlabel('FLOP $=3d\\,n_{tr}\\,n_{te}$'); a2.set_ylabel('$T_p$ [s]')
    a2.set_title('(b) Tiempo vs tamaño del problema')
    a2.legend(loc='upper left', fontsize=8)
    fig.tight_layout(w_pad=2.0)
    save(fig, out, 'escalabilidad')
    log['slopes'] = slopes
    log['weak'] = dict(p=w.p.tolist(), T=np.round(w.T_total_mean.values, 3).tolist(),
                       comp=np.round(w.T_comp_local_max_mean.values, 3).tolist(),
                       comm=np.round(w.T_comm_mean.values, 4).tolist(),
                       E=np.round(w.T_total_mean.iloc[0] / w.T_total_mean.values, 2).tolist())
    print(json.dumps(log, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
