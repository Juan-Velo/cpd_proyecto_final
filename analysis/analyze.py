"""Fase 4 - Analisis de resultados REALES de Khipu (resultados/khipu_<jobid>/*.csv).

  python analysis/analyze.py resultados/khipu_<jobid>

Genera en <dir>/figs/ (PDF+PNG) y <dir>/tablas/:
  tiempos (T_total, T_comp, T_comm) vs p y vs N, accuracy, speedup/eficiencia,
  GFLOP/s vs p, escalabilidad debil, ajuste teorico normalizado y p* teorico vs observado.

Modelo teorico (docs/FASE1_DISENO.md, formulas de U6.1):
  T_comp(p) = c1 * n_tr n_te d / p  +  c2 * n_te k p
  T_comm(p) = alpha * M(p) + beta * B(p)
     M(p) = 2(2p-1) [Scatter X,y]  + log2 p [Bcast] + (2p-1) [Gather]      (nro. de arranques)
     B(p) = (2p-1)/p * n_tr (d+1) * 8 + log2 p * n_te d * 8 + (2p-1) * 3k n_te * 8   (bytes)
  c1, c2, alpha, beta >= 0 se ajustan por minimos cuadrados no negativos (NNLS)
  sobre las medias de las series de escalabilidad fuerte; la serie debil se usa
  como validacion fuera de muestra del modelo FLOP (R2_fuera_de_muestra). El
  modelo extendido no puede validarse con ella (phi sale de esa serie): su
  validacion fuera de muestra esta en analysis/validacion_cruzada.py.
  p* : minimo de T_p(p) = A/p + B p + C log2 p + D  (dT/dp = 0).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': .3, 'legend.fontsize': 8})
COL = {'T_total': '#1f4e79', 'T_comp': '#2e8b57', 'T_comm': '#c0392b', 'T_idle': '#8e8e8e',
       'model': '#e67e22'}
STRONG = ['digits', 'strong_small', 'strong']
LABEL = {'digits': r'load_digits ($n_{tr}$=1437, $n_{te}$=360)',
         'strong_small': r'$n_{tr}$=22 992, $n_{te}$=3 600',
         'strong': r'$n_{tr}$=91 968, $n_{te}$=3 600',
         'weak': r'débil: $n_{tr}/p$=11 496, $n_{te}$=3 600'}


def load(d):
    """Lee <d>/*.csv y, si existen, las series HT hermanas (khipu_ht_*/)."""
    fs = glob.glob(os.path.join(d, '*.csv'))
    fs += glob.glob(os.path.join(os.path.dirname(os.path.abspath(d)), 'khipu_ht_*', '*.csv'))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def agg(df):
    keys = ['tag', 'variant', 'comm_mode', 'kernel', 'p', 'n_train', 'n_test', 'd', 'k']
    m = ['T_total', 'T_comp', 'T_comm', 'T_idle', 'T_scatter', 'T_bcast', 'T_gather', 'T_vote',
         'T_comp_local_max', 'T_comp_local_min', 'accuracy', 'gflops_per_sec', 'gflops_per_sec_comp']
    g = df.groupby(keys)[m].agg(['mean', 'std', 'count']).reset_index()
    g.columns = ['_'.join(c).rstrip('_') for c in g.columns]
    g['total_flops'] = 3 * g.d * g.n_train * g.n_test
    return g.sort_values(['tag', 'p'])


# ------------------------------------------------------------- modelo
def features(p, ntr, nte, d, k):
    p = np.asarray(p, float)
    lg = np.log2(p)
    comp1 = ntr * nte * d / p
    comp2 = nte * k * p
    M = 2 * (2 * p - 1) + lg + (2 * p - 1)
    B = (2 * p - 1) / p * ntr * (d + 1) * 8 + lg * nte * d * 8 + (2 * p - 1) * 3 * k * nte * 8
    return comp1, comp2, M, B


def fit_model(g):
    s = g[g.tag.isin(STRONG) & (g.variant == 'dag') & (g.comm_mode == 'buffer') & (g.kernel == 'gemm')]
    c1f, c2f, M, B = features(s.p, s.n_train, s.n_test, s.d, s.k)
    # computo: T_comp ~ c1*W/p + c2*n_te k p   (ponderado relativo)
    w = 1 / s.T_comp_mean.values
    c, _ = nnls(np.c_[c1f, c2f] * w[:, None], s.T_comp_mean.values * w)
    # comunicacion: T_comm ~ alpha*M + beta*B
    s2 = s[s.p > 1]
    _, _, M2, B2 = features(s2.p, s2.n_train, s2.n_test, s2.d, s2.k)
    w2 = 1 / s2.T_comm_mean.values
    ab, _ = nnls(np.c_[M2, B2] * w2[:, None], s2.T_comm_mean.values * w2)
    return dict(c1=c[0], c2=c[1], alpha=ab[0], beta=ab[1])


def predict(P, p, ntr, nte, d=64, k=3):
    c1f, c2f, M, B = features(p, ntr, nte, d, k)
    comp = P['c1'] * c1f + P['c2'] * c2f
    comm = P['alpha'] * M + P['beta'] * B
    return comp, comm, comp + comm


def p_star(P, ntr, nte, d=64, k=3):
    """dT/dp = 0 para T = A/p + B p + C log2 p + D."""
    A = P['c1'] * ntr * nte * d
    Bc = P['c2'] * nte * k + P['alpha'] * 3 * 2 + P['beta'] * 2 * 3 * k * nte * 8
    C = P['alpha'] + P['beta'] * nte * d * 8
    Cn = C / np.log(2)
    pst = (-Cn + np.sqrt(Cn ** 2 + 4 * A * Bc)) / (2 * Bc)
    return pst, A, Bc, C


def contention_factor(g):
    """phi(p) = T_comp_local_max(p) / T_comp_local_max(1) en la serie debil
    (trabajo por proceso constante => cualquier aumento es contencion del nodo)."""
    w = g[(g.tag == 'weak') & (g.variant == 'dag') & (g.comm_mode == 'buffer')].sort_values('p')
    if w.empty:
        return None
    ph = w.T_comp_local_max_mean.values / w.T_comp_local_max_mean.values[0]
    return dict(p=w.p.values.astype(float), phi=ph)


def phi_at(PHI, p):
    return np.interp(np.log(np.asarray(p, float)), np.log(PHI['p']), PHI['phi'])


def fit_extended(g, P, PHI):
    """Modelo extendido: T_comp = c1' * (n_tr n_te d / p) * phi(p) + c2 n_te k p ; c1' por NNLS."""
    s = g[g.tag.isin(STRONG) & (g.variant == 'dag') & (g.comm_mode == 'buffer') & (g.kernel == 'gemm')]
    c1f, c2f, _, _ = features(s.p, s.n_train, s.n_test, s.d, s.k)
    w = 1 / s.T_comp_mean.values
    c, _ = nnls(np.c_[c1f * phi_at(PHI, s.p), c2f] * w[:, None], s.T_comp_mean.values * w)
    return dict(P, c1e=c[0], c2e=c[1])


def predict_ext(P, PHI, p, ntr, nte, d=64, k=3):
    c1f, c2f, M, B = features(p, ntr, nte, d, k)
    comp = P['c1e'] * c1f * phi_at(PHI, p) + P['c2e'] * c2f
    comm = P['alpha'] * M + P['beta'] * B
    return comp, comm, comp + comm


def r2(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


# ------------------------------------------------------------- figuras
def save(fig, out, name):
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'))
    plt.close(fig)


def main(d):
    out = os.path.join(d, 'figs'); tab = os.path.join(d, 'tablas')
    os.makedirs(out, exist_ok=True); os.makedirs(tab, exist_ok=True)
    df = load(d)
    g = agg(df)
    g.to_csv(os.path.join(tab, 'resumen_por_configuracion.csv'), index=False)
    base = g[(g.variant == 'dag') & (g.comm_mode == 'buffer') & (g.kernel == 'gemm')]

    # referencia secuencial beta1 (mejor secuencial)
    Ts = {}
    fj = os.path.join(d, 'beta1_seq.jsonl')
    if os.path.exists(fj):
        for line in open(fj):
            r = json.loads(line)
            if r['mode'] == 'vector':
                Ts[(r['n_train'], r['n_test'])] = min(r['T_total'], Ts.get((r['n_train'], r['n_test']), 1e99))

    P = fit_model(g)
    PHI = contention_factor(g)
    if PHI is not None:
        P = fit_extended(g, P, PHI)
    report = {'parametros_ajustados': P, 'series': {},
              'phi_contencion': None if PHI is None else dict(p=PHI['p'].tolist(), phi=PHI['phi'].tolist())}

    # ---- (1) tiempos vs p + modelo, por serie fuerte
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, tag in zip(axs, STRONG):
        s = base[base.tag == tag]
        if s.empty:
            continue
        ntr, nte = s.n_train.iloc[0], s.n_test.iloc[0]
        for m in ['T_total', 'T_comp', 'T_comm']:
            ax.errorbar(s.p, s[f'{m}_mean'], yerr=s[f'{m}_std'], marker='o', ms=4, capsize=2,
                        color=COL[m], label=m.replace('_', r'$_{') + '}$' if False else m)
        pp = np.linspace(1, s.p.max(), 300)
        cm, cc, ct = predict(P, pp, ntr, nte)
        ax.plot(pp, ct, '--', color=COL['model'], label='modelo $T_p$ normalizado')
        ax.plot(pp, cc, ':', color=COL['T_comm'], alpha=.7, label='modelo $T_{comm}$')
        if PHI is not None:
            _, _, cte = predict_ext(P, PHI, pp, ntr, nte)
            ax.plot(pp, cte, '-', color='#7d3c98', lw=1.2, alpha=.8, label=r'modelo extendido $\cdot\,\varphi(p)$')
        ps, *_ = p_star(P, ntr, nte)
        pobs = int(s.loc[s.T_total_mean.idxmin(), 'p'])
        ax.text(.03, .04, f'$p^*_{{teo}}$ = {ps:.1f}   $p^*_{{obs}}$ = {pobs}', transform=ax.transAxes,
                fontsize=8, bbox=dict(fc='white', ec='0.7', alpha=.9))
        ax.set(title=LABEL[tag], xlabel='p (procesos)', ylabel='tiempo [s]', yscale='log', xscale='log')
        ax.set_xticks(sorted(s.p.unique())); ax.set_xticklabels(sorted(s.p.unique()))
        _, _, tp_model = predict(P, s.p.values, ntr, nte)
        rec = dict(n_train=int(ntr), n_test=int(nte), p_star_teorico=float(ps),
                   p_star_observado=pobs, R2_T_total=float(r2(s.T_total_mean, tp_model)),
                   err_rel_medio=float(np.mean(np.abs(tp_model - s.T_total_mean) / s.T_total_mean)))
        if PHI is not None:
            _, _, te = predict_ext(P, PHI, s.p.values, ntr, nte)
            rec.update(R2_T_total_ext=float(r2(s.T_total_mean, te)),
                       err_rel_medio_ext=float(np.mean(np.abs(te - s.T_total_mean) / s.T_total_mean)),
                       p_star_ext_en_rango=int(s.p.values[np.argmin(te)]))
        report['series'][tag] = rec
    axs[0].legend()
    save(fig, out, 'tiempos_vs_p')

    # ---- (2) desglose apilado de la serie grande
    for tag in STRONG:
        s = base[base.tag == tag]
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        x = np.arange(len(s))
        bot = np.zeros(len(s))
        for m, lab in [('T_comp_local_max', 'cómputo local (máx.)'), ('T_vote', 'fusión + voto (rank 0)'),
                       ('T_scatter', 'Scatter $X_{tr}$'), ('T_bcast', 'Bcast $X_{te}$'),
                       ('T_gather', 'Gather candidatos'), ('T_idle', 'espera (desbalance)')]:
            v = s[f'{m}_mean'].values
            ax.bar(x, v / s.T_total_mean.values * 100, bottom=bot, label=lab)
            bot += v / s.T_total_mean.values * 100
        ax.set(xticks=x, xticklabels=s.p, xlabel='p', ylabel='% de $T_{total}$', title=LABEL[tag], ylim=(0, 105))
        ax.legend(ncol=2, loc='lower left')
        save(fig, out, f'desglose_{tag}')

    # ---- (3) speedup y eficiencia
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for tag in STRONG:
        s = base[base.tag == tag]
        if s.empty:
            continue
        T1 = s[s.p == 1].T_total_mean.iloc[0]
        S = T1 / s.T_total_mean
        a1.plot(s.p, S, 'o-', label=LABEL[tag]); a2.plot(s.p, S / s.p, 'o-', label=LABEL[tag])
        h = g[(g.tag == tag + '_ht')]
        if len(h):  # p > 16 con hyper-threading (2 procesos por core fisico)
            c_ = a1.lines[-1].get_color()
            a1.plot(h.p, T1 / h.T_total_mean, 's--', color=c_, mfc='none')
            a2.plot(h.p, T1 / h.T_total_mean / h.p, 's--', color=c_, mfc='none')
            report['series'][tag]['ht'] = [dict(p=int(p_), T=float(t), S=float(T1 / t))
                                           for p_, t in zip(h.p, h.T_total_mean)]
        report['series'][tag].update(T1=float(T1), S_max=float(S.max()),
                                     p_S_max=int(s.p.iloc[int(np.argmax(S))]),
                                     tabla=[dict(p=int(p_), T=float(t), S=float(T1 / t), E=float(T1 / t / p_))
                                            for p_, t in zip(s.p, s.T_total_mean)])
        key = (int(s.n_train.iloc[0]), int(s.n_test.iloc[0]))
        if key in Ts:
            report['series'][tag]['Ts_beta1'] = Ts[key]
    pm = g.p.max()
    a1.plot([1, pm], [1, pm], 'k--', lw=.8, label='ideal $S=p$')
    if g.tag.str.endswith('_ht').any():
        a1.axvspan(16, pm, color='grey', alpha=.12, label='p>16: hyper-threading (□)')
        a2.axvspan(16, pm, color='grey', alpha=.12)
    a1.set(xlabel='p', ylabel='$S(p)=T_1/T_p$', title='Speedup (escalabilidad fuerte)')
    a2.axhline(1, color='k', ls='--', lw=.8)
    a2.set(xlabel='p', ylabel='$E(p)=S/p$', title='Eficiencia', ylim=(0, 1.15))
    a1.legend(); save(fig, out, 'speedup_eficiencia')

    # ---- (4) GFLOP/s vs p
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for tag in STRONG:
        s = base[base.tag == tag]
        if s.empty:
            continue
        ax.errorbar(s.p, s.gflops_per_sec_mean, yerr=s.gflops_per_sec_std, marker='o', capsize=2,
                    label=LABEL[tag] + ' ($T_{total}$)')
    dr = g[g.kernel == 'direct']
    if not dr.empty:
        ax.errorbar(dr.p, dr.gflops_per_sec_mean, yerr=dr.gflops_per_sec_std, marker='s', capsize=2,
                    label='fórmula directa, $n_{tr}$=11 496, $n_{te}$=1 800')
    ax.set(xlabel='p', ylabel='GFLOP/s  (FLOP $=3d\\,n_{tr}n_{te}$)', title='Rendimiento FLOP/s vs p',
           xscale='log', yscale='log')
    ax.legend(); save(fig, out, 'gflops_vs_p')

    # ---- (5) escalabilidad debil
    w = base[base.tag == 'weak']
    if not w.empty:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for m in ['T_total', 'T_comp', 'T_comm']:
            a1.errorbar(w.p, w[f'{m}_mean'], yerr=w[f'{m}_std'], marker='o', capsize=2, color=COL[m], label=m)
        _, cc, ct = predict(P, w.p.values, w.n_train.values, w.n_test.values)
        a1.plot(w.p, ct, '--', color=COL['model'], label='modelo (fuera de muestra)')
        a1.set(xlabel='p  ($n_{tr}=11\\,496\\,p$)', ylabel='tiempo [s]', title=LABEL['weak'], xscale='log')
        a1.legend()
        T1 = w[w.p == 1].T_total_mean.iloc[0]
        Ew = T1 / w.T_total_mean
        a2.plot(w.p, Ew, 'o-', label='$E_{débil}=T_1/T_p$')
        a2.plot(w.p, w.gflops_per_sec_mean / w.gflops_per_sec_mean.iloc[0] / w.p, 's--',
                label='GFLOP/s / (p·GFLOP/s$_1$)')
        a2.axhline(1, color='k', ls='--', lw=.8); a2.set(xlabel='p', ylabel='eficiencia', ylim=(0, 1.15),
                                                         title='Eficiencia débil (Gustafson)')
        a2.legend(); save(fig, out, 'escalabilidad_debil')
        report['series']['weak'] = dict(tabla=[dict(p=int(p_), n_train=int(n), T=float(t), E=float(T1 / t),
                                                    T_comm=float(c), T_comp=float(cp))
                                               for p_, n, t, c, cp in zip(w.p, w.n_train, w.T_total_mean,
                                                                          w.T_comm_mean, w.T_comp_mean)],
                                        R2_fuera_de_muestra=float(r2(w.T_total_mean, ct)))

    # ---- (6) tiempos vs N (todas las series dag, p fijos)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    allb = base[base.tag.isin(STRONG + ['weak'])].copy()
    allb['W'] = allb.n_train * allb.n_test
    for p_ in [1, 4, 16, 32]:
        s = allb[allb.p == p_].sort_values('W')
        if len(s):
            ax.plot(s.W * 3 * 64, s.T_total_mean, 'o-', label=f'p={p_}')
    ax.set(xscale='log', yscale='log', xlabel='trabajo FLOP $=3d\\,n_{tr}n_{te}$', ylabel='$T_{total}$ [s]',
           title='Tiempo vs tamaño del problema')
    ax.legend(); save(fig, out, 'tiempo_vs_N')

    # ---- (7) accuracy
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for tag in STRONG + ['weak']:
        s = base[base.tag == tag]
        if len(s):
            a1.plot(s.p, s.accuracy_mean, 'o-', label=LABEL[tag])
    a1.set(xlabel='p', ylabel='accuracy', title='Accuracy vs p (invariante)')
    a1.legend()
    s = base[base.p == 1].copy(); s['ntr'] = s.n_train
    s = pd.concat([s, base[base.tag == 'weak']]).drop_duplicates(['n_train', 'n_test']).sort_values('n_train')
    a2.plot(s.n_train, s.accuracy_mean, 'o-')
    a2.set(xscale='log', xlabel='$n_{tr}$', ylabel='accuracy', title='Accuracy vs $n_{tr}$')
    save(fig, out, 'accuracy')

    # ---- (8) comparacion de variantes
    c = g[g.tag.str.startswith('cmp_') | ((g.tag == 'strong') & g.p.isin([1, 4, 16, 32]))]
    if c.tag.str.startswith('cmp_').any():
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for tag, lab in [('strong', 'DAG: Scatter $X_{tr}$ + Bcast $X_{te}$ (buffers)'),
                         ('cmp_pickle', 'DAG con objetos pickle'),
                         ('cmp_replicate', 'Alternativa: Bcast $X_{tr}$ + Scatter $X_{te}$')]:
            s = c[c.tag == tag]
            ax.errorbar(s.p, s.T_total_mean, yerr=s.T_total_std, marker='o', capsize=2, label=lab)
            s2 = s.copy()
            ax.plot(s2.p, s2.T_comm_mean, ':', marker='x', color=ax.lines[-1].get_color(), alpha=.7)
        ax.set(xscale='log', yscale='log', xlabel='p', ylabel='tiempo [s]  (— total, ··· comm)',
               title='Comparación de descomposiciones ($n_{tr}$=91 968)')
        ax.legend(); save(fig, out, 'comparacion_variantes')
        report['comparacion'] = c[['tag', 'p', 'T_total_mean', 'T_comm_mean', 'T_comp_mean']].to_dict('records')

    # ---- (9) validacion por colectiva (U6.1): cada termino con su forma teorica
    s = base[base.tag.isin(['digits', 'strong_small', 'strong', 'weak']) & (base.p > 1)]
    p_ = s.p.values.astype(float); lg = np.log2(p_); ntr = s.n_train.values; nte = s.n_test.values
    forms = {'T_bcast': (r'Bcast $X_{te}$: $\log_2p\,(\alpha+m\beta)$', np.c_[lg, lg * nte * 64 * 8]),
             'T_scatter': (r'Scatter $X_{tr},y_{tr}$: $(2p-1)(\alpha+\frac{m}{p}\beta)$',
                           np.c_[2 * (2 * p_ - 1), (2 * p_ - 1) / p_ * ntr * 65 * 8]),
             'T_gather': (r'Gather $3k\,n_{te}$: $(2p-1)(\alpha+m\beta)$', np.c_[2 * p_ - 1, (2 * p_ - 1) * 9 * nte * 8])}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    report['colectivas'] = {}
    for ax, (m, (lab, X)) in zip(axs, forms.items()):
        y = s[f'{m}_mean'].values
        wv = 1 / y
        cf, _ = nnls(X * wv[:, None], y * wv)
        yh = X @ cf
        report['colectivas'][m] = dict(alpha=float(cf[0]), beta=float(cf[1]), R2=float(r2(y, yh)))
        for tag in ['digits', 'strong_small', 'strong', 'weak']:
            sel = (s.tag == tag).values
            if sel.any():
                l_ = ax.plot(p_[sel], y[sel] * 1e3, 'o', ms=4, label=tag)[0]
                o = np.argsort(p_[sel])
                ax.plot(p_[sel][o], yh[sel][o] * 1e3, '--', color=l_.get_color(), lw=1)
        ax.set(xscale='log', yscale='log', xlabel='p', ylabel='tiempo [ms]',
               title=f'{lab}\n$R^2$={r2(y, yh):.2f}')
        ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels([2, 4, 8, 16])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].legend(fontsize=7)
    save(fig, out, 'validacion_colectivas')

    # ---- (10) metricas de U2.1/U6.2: dT/dp, costo, overhead T0, Amdahl f_s, energia relativa
    rows = []
    for tag in STRONG:
        s = base[base.tag == tag].sort_values('p')
        T1 = s.T_total_mean.iloc[0]
        prevp, prevT = None, None
        for p0, T in zip(s.p, s.T_total_mean):
            S = T1 / T
            fs = (1 / S - 1 / p0) / (1 - 1 / p0) if p0 > 1 else np.nan
            rows.append(dict(serie=tag, p=int(p0), T=T, S=S, E=S / p0, costo_pTp=p0 * T,
                             T0_overhead=p0 * T - T1, T0_sobre_W=(p0 * T - T1) / T1,
                             dT_dp=(T - prevT) / (p0 - prevp) if prevp else np.nan,
                             fs_amdahl=fs, energia_rel=p0 * T / T1))
            prevp, prevT = p0, T
    w = base[base.tag == 'weak'].sort_values('p')
    if len(w):
        T1 = w.T_total_mean.iloc[0]
        for p0, T in zip(w.p, w.T_total_mean):
            E = T1 / T
            rows.append(dict(serie='weak', p=int(p0), T=T, E=E, S=p0 * E,  # speedup escalado (Gustafson)
                             costo_pTp=p0 * T, T0_overhead=p0 * T - p0 * T1,
                             T0_sobre_W=(p0 * T - p0 * T1) / (p0 * T1), energia_rel=T / T1))
    m62 = pd.DataFrame(rows)
    m62.to_csv(os.path.join(tab, 'metricas_U62.csv'), index=False)
    report['metricas_U62'] = m62.round(5).to_dict('records')

    with open(os.path.join(tab, 'reporte.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main(sys.argv[1])
