"""Figuras de calidad de publicacion para el informe (IEEEtran, 2 columnas).

  python analysis/figs_paper.py resultados/khipu_64271 informe/figs

Lee SOLO los CSV reales de Khipu (via analysis/analyze.py: mismas funciones de
agregacion y ajuste) y exporta PDF vectoriales al tamano exacto de insercion:
1 columna = 3.45 in, 2 columnas = 7.10 in.  Ningun valor se modifica.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

COL1, COL2 = 4.6, 6.3      # informe a 1 columna (11 pt, 	extwidth = 6.3 in)
HS2, HS1 = 1.18, 1.05      # escala de alturas para el ancho nuevo
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'axes.grid': True, 'grid.alpha': .22, 'grid.linestyle': ':', 'grid.linewidth': .6,
    'lines.linewidth': 1.2, 'lines.markersize': 3.8, 'errorbar.capsize': 1.8,
    'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.columnspacing': 1.2,
    'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .02, 'pdf.fonttype': 42,
})
# Okabe-Ito (apto daltonismo); misma identidad visual por serie en TODAS las figuras
SER = {'digits': dict(c='#0072B2', m='o', lab=r'digits ($n_{tr}=1437$)'),
       'strong_small': dict(c='#E69F00', m='s', lab=r'Fuerte 23k ($n_{tr}=22\,992$)'),
       'strong': dict(c='#009E73', m='^', lab=r'Fuerte 92k ($n_{tr}=91\,968$)'),
       'weak': dict(c='#CC79A7', m='D', lab=r'Débil ($n_{tr}=11\,496\,p$)')}
MET = {'T_total': dict(c='#000000', m='o', lab=r'$T_{\mathrm{total}}$'),
       'T_comp': dict(c='#56B4E9', m='s', lab=r'$T_{\mathrm{comp}}$'),
       'T_comm': dict(c='#D55E00', m='^', lab=r'$T_{\mathrm{comm}}$')}
MODEL_C, EXT_C = '#7F7F7F', '#882255'
PS = [1, 2, 4, 6, 8, 12, 16]


def logx(ax, ticks=PS):
    ax.set_xscale('log', base=2)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mt.FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(mt.NullLocator())


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.pdf'))
    fig.savefig(os.path.join(out, name + '.png'), dpi=200)
    plt.close(fig)


def top_legend(fig, handles, labels, ncol, y=1.0):
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, y), ncol=ncol,
               handletextpad=.4, borderaxespad=0)


def main(d, out):
    os.makedirs(out, exist_ok=True)
    g = A.agg(A.load(d))
    base = g[(g.variant == 'dag') & (g.comm_mode == 'buffer') & (g.kernel == 'gemm')]
    P = A.fit_model(g)
    PHI = A.contention_factor(g)
    P = A.fit_extended(g, P, PHI)
    S_ = lambda t: base[base.tag == t].sort_values('p')  # noqa: E731
    log = {}

    # ---------------- Fig. tiempos vs p (3 paneles, 2 columnas)
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.15 * HS2))
    for ax, tag in zip(axs, ['digits', 'strong_small', 'strong']):
        s = S_(tag); ntr, nte = s.n_train.iloc[0], s.n_test.iloc[0]
        sc = 1e3 if tag == 'digits' else 1.0
        for m, st in MET.items():
            ax.errorbar(s.p, s[f'{m}_mean'] * sc, yerr=s[f'{m}_std'] * sc, color=st['c'], marker=st['m'],
                        lw=1, ms=3.2, elinewidth=.7, label=st['lab'])
        pp = np.geomspace(1, 16, 200)
        _, _, ct = A.predict(P, pp, ntr, nte)
        _, _, ce = A.predict_ext(P, PHI, pp, ntr, nte)
        ax.plot(pp, ct * sc, '--', color=MODEL_C, lw=1, label=r'Modelo FLOP + $\alpha$-$\beta$')
        ax.plot(pp, ce * sc, '-', color=EXT_C, lw=1, alpha=.85, label='Modelo extendido ($\\varphi(p)$)')
        ps, *_ = A.p_star(P, ntr, nte)
        pobs = int(s.loc[s.T_total_mean.idxmin(), 'p'])
        _, _, te = A.predict_ext(P, PHI, s.p.values, ntr, nte)
        r2e = A.r2(s.T_total_mean, te)
        if tag == 'digits':
            ax.set_ylim(5e-3, 25)
        ax.text(.04, .04 if tag == 'digits' else .50, f'$p^*_{{\\mathrm{{teo}}}}={ps:.1f}$\n$p^*_{{\\mathrm{{obs}}}}={pobs}$\n$R^2_{{\\mathrm{{ext}}}}={r2e:.3f}$',
                transform=ax.transAxes, ha='left', va='bottom' if tag == 'digits' else 'center',
                fontsize=7.3, linespacing=1.3,
                bbox=dict(fc='white', ec='#BBBBBB', lw=.5, pad=2.2))
        ax.set_yscale('log'); logx(ax)
        ax.set_title(SER[tag]['lab'].replace('digits', 'load_digits'), pad=3)
        ax.set_xlabel('$p$ (procesos)')
        ax.set_ylabel('Tiempo [ms]' if tag == 'digits' else 'Tiempo [s]')
        log[tag] = dict(p_star_teo=ps, p_obs=pobs, R2_ext=r2e)
    h, l_ = axs[0].get_legend_handles_labels()
    top_legend(fig, h, l_, 5, 1.0)
    fig.tight_layout(w_pad=1.2)
    save(fig, out, 'tiempos_vs_p')

    # ---------------- Fig. desglose (3 paneles)
    comps = [('T_comp_local_max', 'Cómputo local (máx.)', '#4477AA'), ('T_vote', 'Fusión + voto', '#66CCEE'),
             ('T_scatter', 'Scatter $X_{tr}$', '#228833'), ('T_bcast', 'Bcast $X_{te}$', '#CCBB44'),
             ('T_gather', 'Gather candidatos', '#EE6677'), ('T_idle', 'Espera (desbalance)', '#BBBBBB')]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.0 * HS2), sharey=True)
    for ax, tag in zip(axs, ['digits', 'strong_small', 'strong']):
        s = S_(tag); x = np.arange(len(s)); bot = np.zeros(len(s))
        for m, lab, c in comps:
            v = s[f'{m}_mean'].values / s.T_total_mean.values * 100
            ax.bar(x, v, bottom=bot, color=c, width=.72, label=lab, edgecolor='white', linewidth=.3)
            bot += v
        ax.set_xticks(x); ax.set_xticklabels(s.p); ax.set_xlabel('$p$')
        ax.set_title(SER[tag]['lab'].replace('digits', 'load_digits'), pad=3)
        ax.set_ylim(0, 100); ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('% de $T_{\\mathrm{total}}$')
    h, l_ = axs[0].get_legend_handles_labels()
    top_legend(fig, h, l_, 6, 1.0)
    fig.tight_layout(w_pad=.8)
    save(fig, out, 'desglose')

    # ---------------- Fig. validacion por colectiva
    s = base[base.tag.isin(SER) & (base.p > 1)]
    p_ = s.p.values.astype(float); lg = np.log2(p_); ntr = s.n_train.values; nte = s.n_test.values
    forms = [('T_bcast', 'Bcast $X_{te}$', r'$\log_2 p\,(\alpha+m\beta)$', np.c_[lg, lg * nte * 64 * 8]),
             ('T_scatter', 'Scatter $X_{tr},y_{tr}$', r'$(2p-1)(\alpha+\frac{m}{p}\beta)$',
              np.c_[2 * (2 * p_ - 1), (2 * p_ - 1) / p_ * ntr * 65 * 8]),
             ('T_gather', 'Gather de candidatos', r'$(2p-1)(\alpha+m\beta)$',
              np.c_[2 * p_ - 1, (2 * p_ - 1) * 9 * nte * 8])]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.15 * HS2))
    for ax, (m, tit, eq, X) in zip(axs, forms):
        y = s[f'{m}_mean'].values; wv = 1 / y
        cf, _ = nnls(X * wv[:, None], y * wv); yh = X @ cf
        for tag, st in SER.items():
            sel = (s.tag == tag).values
            o = np.argsort(p_[sel])
            ax.plot(p_[sel][o], y[sel][o] * 1e3, st['m'], color=st['c'], ms=3.5, label=st['lab'])
            # Bcast y Gather solo dependen de n_te: las tres series con n_te=3600 comparten
            # exactamente la misma curva teorica -> una sola linea gris (evita 3 lineas superpuestas)
            if m != 'T_scatter' and tag != 'digits':
                if tag == 'strong':
                    ax.plot(p_[sel][o], yh[sel][o] * 1e3, '-', color='#555555', lw=.9, alpha=.8)
                continue
            ax.plot(p_[sel][o], yh[sel][o] * 1e3, '-', color=st['c'], lw=.9, alpha=.75)
        ax.set_yscale('log'); logx(ax, [2, 4, 8, 16])
        ax.set_title(f'{tit}\n{eq}', pad=3, fontsize=8.8, linespacing=1.4)
        ax.set_xlabel('$p$ (procesos)'); ax.set_ylabel('Tiempo [ms]')
        ypos = .42 if m == 'T_scatter' else .08
        ax.text(.97, ypos, f'$R^2={A.r2(y, yh):.2f}$', transform=ax.transAxes, ha='right', va='center',
                fontsize=7.5, bbox=dict(fc='white', ec='none', pad=1))
        log[m] = A.r2(y, yh)
    h, l_ = axs[0].get_legend_handles_labels()
    top_legend(fig, h, l_, 4, 1.0)
    fig.tight_layout(w_pad=1.0)
    save(fig, out, 'validacion_colectivas')

    # ---------------- Fig. speedup y eficiencia
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.35 * HS2))
    for tag in ['digits', 'strong_small', 'strong']:
        st = SER[tag]; s = S_(tag)
        T1, s1 = s.T_total_mean.iloc[0], s.T_total_std.iloc[0]
        S = T1 / s.T_total_mean.values
        sS = S * np.sqrt((s1 / T1) ** 2 + (s.T_total_std.values / s.T_total_mean.values) ** 2)
        sS[s.p.values == 1] = 0.0   # S(1) = T1/T1 = 1 exacto: sin incertidumbre en p=1
        for ax, v, sv in [(a1, S, sS), (a2, S / s.p.values, sS / s.p.values)]:
            ax.plot(s.p, v, marker=st['m'], color=st['c'], label=st['lab'])
            ax.fill_between(s.p, v - sv, v + sv, color=st['c'], alpha=.18, lw=0)
        h = g[g.tag == tag + '_ht'].sort_values('p')
        if len(h):
            Sh = T1 / h.T_total_mean.values
            a1.plot(h.p, Sh, ls='--', marker=st['m'], mfc='white', color=st['c'], lw=.9)
            a2.plot(h.p, Sh / h.p.values, ls='--', marker=st['m'], mfc='white', color=st['c'], lw=.9)
    for ax in (a1, a2):
        ax.axvspan(16, 32, color='#999999', alpha=.10, lw=0)
        logx(ax, [1, 2, 4, 8, 16, 24, 32])
        ax.set_xlabel('$p$ (procesos)')
    a2.text(21.5, .62, 'hyper-threading\n(2 procesos/núcleo)', rotation=90, fontsize=6.8, color='#555555',
            ha='center', va='center')
    a1.plot([1, 32], [1, 32], ':', color='#555555', lw=.8, label='Ideal $S=p$')
    a1.set_yscale('log', base=2); a1.set_yticks([1, 2, 4, 8, 16, 32])
    a1.yaxis.set_major_formatter(mt.FixedFormatter(['1', '2', '4', '8', '16', '32']))
    a1.yaxis.set_minor_locator(mt.NullLocator())
    a1.set_ylabel('Speedup $S(p)=T_1/T_p$')
    a2.axhline(1, ls=':', color='#555555', lw=.8)
    a2.set_ylim(0, 1.08); a2.set_ylabel('Eficiencia $E(p)=S/p$')
    h, l_ = a1.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], ls='--', marker='o', mfc='white', color='#555555', lw=.9))
    l_.append('HT: 2 procesos/núcleo')
    from matplotlib.patches import Patch
    h.append(Patch(color='#555555', alpha=.25, lw=0))
    l_.append(r'banda: $\pm\sigma$ (5 rep.)')
    top_legend(fig, h, l_, 3, 1.0)
    fig.tight_layout(w_pad=2.0)
    save(fig, out, 'speedup_eficiencia')

    # ---------------- Fig. GFLOP/s (1 columna)
    fig, ax = plt.subplots(figsize=(COL1, 2.45 * HS1))
    for tag in ['digits', 'strong_small', 'strong']:
        st = SER[tag]; s = S_(tag)
        ax.errorbar(s.p, s.gflops_per_sec_mean, yerr=s.gflops_per_sec_std, color=st['c'], marker=st['m'],
                    elinewidth=.7, label=st['lab'].split(' (')[0])
    dr = g[g.kernel == 'direct'].sort_values('p')
    ax.errorbar(dr.p, dr.gflops_per_sec_mean, yerr=dr.gflops_per_sec_std, color='#555555', marker='x',
                ls='--', elinewidth=.7, label='Fórmula directa')
    best = S_('strong').gflops_per_sec_mean.max()
    ax.annotate(f'{best:.1f} GFLOP/s', xy=(16, best), xytext=(5.2, best * 1.35), fontsize=7.3,
                arrowprops=dict(arrowstyle='-', lw=.6, color='#009E73'), color='#009E73')
    ax.set_yscale('log'); logx(ax)
    ax.set_ylim(.5, 120)
    ax.set_xlabel('$p$ (procesos)'); ax.set_ylabel('Rendimiento [GFLOP/s]')
    ax.set_title(r'FLOP $=3d\,n_{tr}\,n_{te}$,  $T=T_{\mathrm{total}}$', fontsize=8.5, pad=3)
    ax.legend(loc='lower right', ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, out, 'gflops_vs_p')

    # ---------------- Fig. escalabilidad debil (2 paneles)
    w = S_('weak')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.0 * HS2), gridspec_kw=dict(wspace=.38))
    for m, st in MET.items():
        a1.errorbar(w.p, w[f'{m}_mean'], yerr=w[f'{m}_std'], color=st['c'], marker=st['m'], elinewidth=.7,
                    label=st['lab'])
    _, _, ce = A.predict_ext(P, PHI, w.p.values, w.n_train.values, w.n_test.values)
    a1.set_yscale('log'); logx(a1)
    a1.set_xlabel('$p$  ($n_{tr}=11\\,496\\,p$, $n_{te}=3600$)'); a1.set_ylabel('Tiempo [s]')
    a1.set_title('Tiempos (trabajo por proceso constante)', pad=3)
    a1.legend(loc='center left', bbox_to_anchor=(.02, .45), ncol=1)
    T1 = w.T_total_mean.iloc[0]
    a2.plot(w.p, T1 / w.T_total_mean, marker='D', color=SER['weak']['c'], label='$E_{\\mathrm{débil}}=T_1/T_p$')
    a2.plot(w.p, 1 / (PHI['phi']), ls='--', marker='o', mfc='white', color='#555555', lw=.9,
            label='$1/\\varphi(p)$ (cómputo)')
    a2.axhline(1, ls=':', color='#555555', lw=.8)
    logx(a2); a2.set_ylim(0, 1.08)
    a2.set_xlabel('$p$'); a2.set_ylabel('Eficiencia débil')
    a2.set_title('Eficiencia (Gustafson)', pad=3)
    a2.legend(loc='lower left')
    fig.tight_layout(w_pad=2.2)
    save(fig, out, 'escalabilidad_debil')

    # ---------------- Fig. tiempo vs trabajo (1 columna) con pendientes calculadas
    allb = base[base.tag.isin(SER)].copy()
    allb['F'] = 3 * 64 * allb.n_train * allb.n_test
    fig, ax = plt.subplots(figsize=(COL1, 2.45 * HS1))
    slopes = {}
    for p0, c, mk in [(1, '#000000', 'o'), (4, '#0072B2', 's'), (16, '#D55E00', '^')]:
        s = allb[allb.p == p0].drop_duplicates(['n_train', 'n_test']).sort_values('F')
        big = s[s.F >= 1e10]
        sl = np.polyfit(np.log(big.F), np.log(big.T_total_mean), 1)[0]
        slopes[p0] = sl
        ax.plot(s.F, s.T_total_mean, marker=mk, color=c, lw=1, label=f'$p={p0}$  (pendiente $m={sl:.2f}$)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Trabajo [FLOP] $=3d\\,n_{tr}n_{te}$'); ax.set_ylabel('$T_{\\mathrm{total}}$ [s]')
    ax.legend(loc='upper left')
    fig.tight_layout()
    save(fig, out, 'tiempo_vs_N')
    log['slopes_F_ge_1e10'] = slopes

    # ---------------- Fig. exactitud (2 paneles)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 1.85 * HS2), gridspec_kw=dict(wspace=.32))
    for tag in ['digits', 'strong_small', 'strong']:   # mismo conjunto de datos para todo p
        st = SER[tag]; s = S_(tag)
        a1.plot(s.p, s.accuracy_mean, marker=st['m'], color=st['c'], label=st['lab'])
    logx(a1); a1.set_ylim(.975, .990)
    a1.set_xlabel('$p$'); a1.set_ylabel('Exactitud'); a1.set_title('Invariancia respecto a $p$', pad=3)
    h = base[(base.n_test == 3600)].drop_duplicates('n_train').sort_values('n_train')
    a2.plot(h.n_train, h.accuracy_mean, marker='D', color=SER['weak']['c'], label='$n_{te}=3600$ (aumentado)')
    dg = S_('digits').accuracy_mean.iloc[0]
    a2.axhline(dg, ls='--', color=SER['digits']['c'], lw=.9)
    a2.plot([1437], [dg], marker='o', color=SER['digits']['c'], ls='none', label=f'load_digits, $n_{{te}}=360$ ({dg:.4f})')
    a2.set_xscale('log'); a2.set_ylim(.975, .990)
    # una corrida por n_tr (p=1 cuando existe; si no, la serie debil): equivalente, la exactitud no depende de p
    a2.set_xlabel('$n_{tr}$'); a2.set_ylabel('Exactitud'); a2.set_title('Exactitud vs $n_{tr}$', pad=3)
    a2.legend(loc='lower right')
    h1, l1 = a1.get_legend_handles_labels()
    top_legend(fig, h1, l1, 3, 1.0)
    fig.tight_layout(w_pad=2.0)
    save(fig, out, 'accuracy')
    log['acc_n_te_3600'] = dict(zip(h.n_train.astype(int).tolist(), h.accuracy_mean.round(5).tolist()))

    # ---------------- Fig. comparacion de variantes (2 paneles)
    var = [('strong', 'DAG, buffers NumPy (final)', '#009E73', '^', '-'),
           ('cmp_pickle', 'DAG, objetos pickle', '#E69F00', 's', '--'),
           ('cmp_replicate', 'Alternativa: Bcast $X_{tr}$ + Scatter $X_{te}$', '#D55E00', 'o', '-.')]
    c = g[g.tag.isin([v[0] for v in var])]
    c = c[c.p.isin([1, 4, 8, 16])] if True else c
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 1.9 * HS2), gridspec_kw=dict(wspace=.3))
    for tag, lab, col, mk, ls in var:
        s = c[(c.tag == tag) & (c.variant.isin(['dag', 'replicate'])) & (c.kernel == 'gemm')]
        if tag == 'strong':
            s = s[(s.variant == 'dag') & (s.comm_mode == 'buffer')]
        s = s.sort_values('p')
        a1.errorbar(s.p, s.T_total_mean, yerr=s.T_total_std, color=col, marker=mk, ls=ls, elinewidth=.7, label=lab)
        a2.errorbar(s.p, s.T_comm_mean * 1e3, yerr=s.T_comm_std * 1e3, color=col, marker=mk, ls=ls, elinewidth=.7)
    for ax in (a1, a2):
        logx(ax, [1, 4, 8, 16]); ax.set_xlabel('$p$')
    a1.set_yscale('log'); a1.set_yticks([1, 2, 4, 7])
    a1.yaxis.set_major_formatter(mt.FixedFormatter(['1', '2', '4', '7']))
    a1.yaxis.set_minor_formatter(mt.NullFormatter()); a1.set_ylabel('$T_{\\mathrm{total}}$ [s]'); a1.set_title('Tiempo total', pad=3)
    a2.set_ylabel('$T_{\\mathrm{comm}}$ [ms]'); a2.set_title('Tiempo de comunicación', pad=3)
    a2.set_ylim(0, 65)
    st16 = c[(c.tag == 'strong') & (c.p == 16) & (c.comm_mode == 'buffer')].T_comm_mean.iloc[0] * 1e3
    rp16 = c[(c.tag == 'cmp_replicate') & (c.p == 16)].T_comm_mean.iloc[0] * 1e3
    a2.annotate(f'{rp16 / st16:.1f}$\\times$ menos', xy=(16, st16), xytext=(5.5, 25), fontsize=7.3,
                arrowprops=dict(arrowstyle='->', lw=.6, color='#333333'))
    h, l_ = a1.get_legend_handles_labels()
    top_legend(fig, h, l_, 3, 1.0)
    fig.tight_layout(w_pad=2.0)
    save(fig, out, 'comparacion_variantes')
    log['comm16_ms'] = dict(final=st16, replicate=rp16)

    print(json.dumps(log, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
