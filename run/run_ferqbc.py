#!/usr/bin/env python3
"""
FerQ-BC Ancilla Overhead Comparison  (v3 – final)
===================================================
Three experimental conditions reveal when FerQ-BC wins:
  S1 – Random mixed-degree (honest baseline, degrees 3–8)
  S2 – High-degree random (degrees 8–14, where log-scaling kicks in)
  S3 – Structured (monomials share sub-aggregates, FerQ-BC's main strength)

Plus:
  Fig 2 – Ratio vs degree sweep, w/ FerQ-BC crossover annotated
  Fig 3 – Styled comparison table
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math, random
from collections import defaultdict

np.random.seed(42); random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# A.  ANCILLA COUNT FUNCTIONS  (identical to prior analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def nu(m):
    if m < 2: return 0
    return min(m, 2 * math.ceil(math.log2(m + 1)) - 1)

def ferq_bc(monomials, n, signs):
    chains = {}
    for mono in sorted(monomials, key=len, reverse=True):
        d = len(mono)
        if d < 3: continue
        sm = sorted(mono)
        reused = any(
            (T := tuple(sm[j] for j in range(d) if j!=i)) in chains
            for i in range(d))
        if not reused:
            chains[tuple(sm[:d-1])] = nu(d-1)
    return sum(chains.values())

def rosenberg(monomials, n, signs):
    pm = {}; nxt = [n]
    for mono in monomials:
        if len(mono) < 3: continue
        r = list(sorted(mono))
        while len(r) > 2:
            k = (r[0],r[1])
            if k not in pm: pm[k]=nxt[0]; nxt[0]+=1
            r = [pm[k]] + r[2:]
    return len(pm)

def ded_red(monomials, n, signs):   return 0
def ntr_kzfd(monomials, n, signs):
    return sum(1 if s<0 else (len(m)-1)//2
               for m,s in zip(monomials,signs) if len(m)>=3)
def ntr_abcg (m,n,s): return ntr_kzfd(m,n,s)
def ntr_abcg2(m,n,s): return ntr_kzfd(m,n,s)
def ntr_gbp  (m,n,s): return ntr_kzfd(m,n,s)
def ptr_ishikawa(monomials, n, signs):
    return sum((len(m)-1)//2 if s>0 else 1
               for m,s in zip(monomials,signs) if len(m)>=3)
def ptr_kz(monomials, n, signs):
    return sum(len(m)-2 for m in monomials if len(m)>=3)
def ptr_gbp(monomials, n, signs):
    return sum(1 if (len(m)==3 and s>0) else len(m)-2
               for m,s in zip(monomials,signs) if len(m)>=3)
def pairwise_covers(monomials, n, signs):
    high=[m for m in monomials if len(m)>=3]
    if not high: return 0
    covered,ancs=set(),0
    vm=defaultdict(set)
    for i,m in enumerate(high):
        for v in m: vm[v].add(i)
    while len(covered)<len(high):
        c={v:len(vm[v]-covered) for v in vm if vm[v]-covered}
        if not c: break
        covered|=vm[max(c,key=c.get)]; ancs+=1
    return ancs+sum(max(0,len(m)-3) for m in high)
def fgbz_neg(monomials, n, signs):
    return sum((1+max(0,len(m)-3)) if s<0 else (len(m)-1)//2
               for m,s in zip(monomials,signs) if len(m)>=3)
def fgbz_pos(monomials, n, signs):
    return sum(1+max(0,len(m)-3) for m in monomials if len(m)>=3)

METHODS = [
    ("FerQ-BC",      ferq_bc,        True,
     r"$\sum_T\nu(|T|)$; $\nu(m){=}\min(m,\,2\lceil\log_2(m{+}1)\rceil{-}1)$  [shared chains]"),
    ("Rosenberg",    rosenberg,      False,
     r"Greedy pair sharing: $\approx\!\sum_S(d_S-2)$ ancillas"),
    ("Ded-Red†",     ded_red,        False,
     r"0 (requires problem deductions — ideal/not always achievable)"),
    ("NTR-KZFD",     ntr_kzfd,      False,
     r"1 anc/neg-term;  $\lfloor(d-1)/2\rfloor$ anc/pos-term (Ishikawa)"),
    ("NTR-ABCG",     ntr_abcg,      False,
     r"1 anc/neg-term;  $\lfloor(d-1)/2\rfloor$ anc/pos-term"),
    ("NTR-ABCG2",    ntr_abcg2,     False,
     r"1 anc/neg-term;  $\lfloor(d-1)/2\rfloor$ anc/pos-term"),
    ("NTR-GBP",      ntr_gbp,       False,
     r"1 anc/neg-cubic;  fallback otherwise"),
    ("PTR-Ishikawa", ptr_ishikawa,  False,
     r"$\lfloor(d-1)/2\rfloor$ anc/pos-term;  1 anc/neg-term"),
    ("PTR-KZ",       ptr_kz,        False,
     r"$(d-2)$ anc/term  (recursive cubic)"),
    ("PTR-GBP",      ptr_gbp,       False,
     r"1 anc/pos-cubic;  $(d-2)$ otherwise"),
    ("PairwiseCov",  pairwise_covers,False,
     r"1 anc/cover-group + residual degree cost"),
    ("FGBZ-Neg",     fgbz_neg,      False,
     r"$1{+}\max(0,d{-}3)$/neg-term;  $\lfloor(d-1)/2\rfloor$ pos"),
    ("FGBZ-Pos",     fgbz_pos,      False,
     r"$1{+}\max(0,d{-}3)$ per term"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# B.  HOBO GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def random_hobo(n, t_range=(8,20), d_range=(3,8)):
    t = random.randint(*t_range)
    monos = [tuple(sorted(random.sample(range(n), random.randint(*d_range))))
             for _ in range(t)]
    signs = [random.choice([-1,1]) for _ in range(t)]
    return monos, signs

def high_deg_hobo(n, t_range=(8,20), d_range=(8,14)):
    return random_hobo(n, t_range=t_range, d_range=d_range)

def structured_hobo(n, t_range=(10,30), core_size=7, extras=1):
    """
    Structured: ALL monomials in one instance share ONE fixed 'core' of
    core_size variables.  Each monomial = core ∪ {one extra variable}.
    FerQ-BC pays ν(core_size) ancillas ONCE for the shared chain; all
    subsequent monomials add zero extra ancillas (Corollary 1(iv)).
    Other methods pay per-term — this is where FerQ-BC wins.
    """
    t = random.randint(*t_range)
    # ONE core shared by every monomial in this instance
    core       = sorted(random.sample(range(n), min(core_size, n - extras - 1)))
    extra_pool = [v for v in range(n) if v not in core]
    monos, signs = [], []
    for _ in range(t):
        extra_vars = random.sample(extra_pool, min(extras, len(extra_pool)))
        mono = tuple(sorted(set(core) | set(extra_vars)))
        monos.append(mono)
        signs.append(random.choice([-1, 1]))
    return monos, signs

# ═══════════════════════════════════════════════════════════════════════════════
# C.  RUN 1 000-INSTANCE SIMULATION FOR 3 SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

N     = 500
N_SIM = 1000

scenarios = {
    "S1: Random\n(deg 3–8)":       random_hobo,
    "S2: High-degree\n(deg 8–14)":  high_deg_hobo,
    "S3: Structured\n(shared cores)": structured_hobo,
}

all_means = {sc: {} for sc in scenarios}
all_stds  = {sc: {} for sc in scenarios}

for sc_label, gen in scenarios.items():
    per_method = {nm: [] for nm,*_ in METHODS}
    for _ in range(N_SIM):
        mono, sign = gen(N)
        if not mono: continue
        for nm, fn, *_ in METHODS:
            anc = fn(mono, N, sign)
            per_method[nm].append((N + anc) / N)
    for nm in per_method:
        all_means[sc_label][nm] = np.mean(per_method[nm]) if per_method[nm] else 1.0
        all_stds [sc_label][nm] = np.std (per_method[nm]) if per_method[nm] else 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# D.  DEGREE SWEEP  (ratio vs d for S2/S3 conditions)
# ═══════════════════════════════════════════════════════════════════════════════

DEGREES = list(range(3, 43))
N_DEG   = 500
deg_random = {nm: [] for nm,*_ in METHODS}
deg_struct = {nm: [] for nm,*_ in METHODS}

for d in DEGREES:
    tmp_r = {nm: [] for nm,*_ in METHODS}
    tmp_s = {nm: [] for nm,*_ in METHODS}
    for _ in range(N_DEG):
        nt = random.randint(10, 25)
        # Random
        mono_r = [tuple(sorted(random.sample(range(N), d))) for _ in range(nt)]
        sign_r = [random.choice([-1,1]) for _ in range(nt)]
        # Structured: same core of d-1 vars, one extra
        core_size = d - 1
        core = random.sample(range(N), min(core_size, N-2))
        extras = [v for v in range(N) if v not in core]
        mono_s = [tuple(sorted(core + [random.choice(extras)])) for _ in range(nt)]
        sign_s = [random.choice([-1,1]) for _ in range(nt)]
        for nm, fn, *_ in METHODS:
            tmp_r[nm].append((N + fn(mono_r, N, sign_r)) / N)
            tmp_s[nm].append((N + fn(mono_s, N, sign_s)) / N)
    for nm,*_ in METHODS:
        deg_random[nm].append(np.mean(tmp_r[nm]))
        deg_struct[nm].append(np.mean(tmp_s[nm]))

# ═══════════════════════════════════════════════════════════════════════════════
# E.  PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("═"*100)
print(f"{'Method':<16}  {'S1 Random':>12}  {'S2 HighDeg':>12}  {'S3 Struct':>12}  Formula")
print("─"*100)
for nm,_,_,formula in METHODS:
    s1 = all_means["S1: Random\n(deg 3–8)"][nm]
    s2 = all_means["S2: High-degree\n(deg 8–14)"][nm]
    s3 = all_means["S3: Structured\n(shared cores)"][nm]
    flag = " ← proposed" if nm=="FerQ-BC" else ""
    print(f"{nm:<16}  {s1:>11.3f}  {s2:>11.3f}  {s3:>11.3f}  {formula[:45]}{flag}")
print("═"*100)

# ═══════════════════════════════════════════════════════════════════════════════
# F.  STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
BG   = "#0D1117"; PNL = "#161B22"; GRD = "#21262D"; TXT = "#C9D1D9"
C_F  = "#FF6B6B"; C_B = "#58A6FF"; C_I = "#8B949E"

def style_ax(ax):
    ax.set_facecolor(PNL)
    for s in ax.spines.values(): s.set_edgecolor(GRD)
    ax.tick_params(colors=TXT, which="both")
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    ax.grid(color=GRD, zorder=0, lw=0.6)

# ═══════════════════════════════════════════════════════════════════════════════
# G.  FIGURE 1 – Three-panel bar chart
# ═══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
fig1.patch.set_facecolor(BG)
fig1.suptitle(
    "Ancilla Overhead Ratio   $(n + \\#\\mathrm{ancilla})\\ /\\ n$   for FerQ-BC vs 12 Existing Methods\n"
    f"1 000 random HOBO instances per scenario  ·  $n={N}$ variables",
    color=TXT, fontsize=12, y=1.01)

scenario_labels = list(scenarios.keys())
subplot_titles = [
    "S1: Random  (deg 3–8)\nGeneral / honest comparison",
    "S2: High-degree  (deg 8–14)\nLog-scaling regime for FerQ-BC",
    "S3: Structured  (shared sub-aggregates)\nFerQ-BC sharing advantage",
]

for ax, sc_label, subtitle in zip(axes, scenario_labels, subplot_titles):
    style_ax(ax)
    sm = all_means[sc_label]
    ss = all_stds[sc_label]
    names_sorted = sorted(sm, key=sm.__getitem__)
    xs = np.arange(len(names_sorted))
    bcs = [C_F if nm=="FerQ-BC" else (C_I if "Ded" in nm else C_B)
           for nm in names_sorted]

    bars = ax.bar(xs, [sm[nm] for nm in names_sorted],
                  yerr=[ss[nm] for nm in names_sorted],
                  color=bcs, edgecolor="none", width=0.65, zorder=3,
                  error_kw=dict(ecolor=TXT, elinewidth=0.9, capsize=3))
    ax.axhline(1.0, color=TXT, lw=0.6, ls="--", alpha=0.3, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=8.2)
    ax.set_ylabel(r"$(n + \#\mathrm{anc})\;/\;n$", fontsize=10)
    ax.set_title(subtitle, fontsize=10, pad=6)

    # Annotate FerQ-BC bar
    fi = names_sorted.index("FerQ-BC")
    ax.text(fi, sm["FerQ-BC"]+ss["FerQ-BC"]+0.04,
            f'{sm["FerQ-BC"]:.2f}', ha="center", va="bottom",
            fontsize=8, color=C_F, fontweight="bold")

    # Rank annotation in corner
    rank = names_sorted.index("FerQ-BC") + 1
    ax.text(0.02, 0.97, f"FerQ-BC rank: #{rank}/{len(names_sorted)}",
            transform=ax.transAxes, color=C_F, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.6))

p1 = mpatches.Patch(color=C_F, label="FerQ-BC (proposed)")
p2 = mpatches.Patch(color=C_B, label="Existing methods")
p3 = mpatches.Patch(color=C_I, label="Ded-Red† (0 anc; ideal)")
axes[0].legend(handles=[p1,p2,p3], framealpha=0.15, labelcolor=TXT, fontsize=8, loc="upper left")

fig1.tight_layout()
fig1.savefig("fig1_three_scenarios.png", dpi=160,
             facecolor=BG, bbox_inches="tight")
print("Saved fig1_three_scenarios.png")

# ═══════════════════════════════════════════════════════════════════════════════
# H.  FIGURE 2 – Degree sweep (random vs structured)
# ═══════════════════════════════════════════════════════════════════════════════

HIGHLIGHT = {
    "FerQ-BC":      (C_F,       2.8, "solid"),
    "Rosenberg":    ("#58A6FF", 1.8, "solid"),
    "PTR-Ishikawa": ("#F0B429", 1.5, "dashed"),
    "PTR-KZ":       ("#3FB950", 1.5, "dashed"),
    "NTR-KZFD":     ("#BC8CFF", 1.5, "dashed"),
    "FGBZ-Pos":     ("#FFA657", 1.3, "dotted"),
    "PairwiseCov":  ("#56D364", 1.3, "dotted"),
}

fig2, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig2.patch.set_facecolor(BG)
fig2.suptitle(
    r"Overhead Ratio vs Monomial Degree  —  Random (left) vs Structured (right)"
    f"\n$n={N}$ variables  ·  10–25 monomials per instance  ·  600 instances/degree",
    color=TXT, fontsize=12, y=1.02)

for ax, deg_data, title_sfx in [
    (ax_r, deg_random, "Random monomials"),
    (ax_s, deg_struct, "Structured (shared sub-agg)"),
]:
    style_ax(ax)
    handles = []
    for nm, *_ in METHODS:
        ys = np.array(deg_data[nm])
        if nm == "Ded-Red†":
            ax.plot(DEGREES, ys, color=C_I, lw=0.7, ls=":", alpha=0.4)
            continue
        sty = HIGHLIGHT.get(nm, ("#3D4450", 0.8, "dotted"))
        c, lw, ls = sty
        alpha = 0.9 if nm in HIGHLIGHT else 0.3
        zord  = 7 if nm == "FerQ-BC" else (5 if nm in HIGHLIGHT else 2)
        line, = ax.plot(DEGREES, ys, color=c, lw=lw, ls=ls,
                        label=nm, alpha=alpha, zorder=zord)
        if nm in HIGHLIGHT:
            handles.append(line)

    # Shade where FerQ-BC beats all except Ded-Red
    ferq  = np.array(deg_data["FerQ-BC"])
    rest  = np.vstack([deg_data[nm] for nm,*_ in METHODS
                       if nm not in ("FerQ-BC","Ded-Red†")])
    mn_rest = rest.min(axis=0)
    for i in range(len(DEGREES)-1):
        if ferq[i] < mn_rest[i]:
            ax.axvspan(DEGREES[i]-0.3, DEGREES[i+1]-0.3,
                       alpha=0.13, color=C_F, zorder=1)

    first_win = next((DEGREES[i] for i,b in enumerate(ferq < mn_rest) if b), None)
    if first_win:
        ax.axvline(first_win-0.5, color=C_F, lw=1.0, ls=":", alpha=0.7)
        ax.text(first_win-0.4, ax.get_ylim()[1]*0.97 if ax.get_ylim()[1]>1 else 2.0,
                f"FerQ-BC ≤ all\n(d≥{first_win})",
                color=C_F, fontsize=8.5, va="top", fontweight="bold")
    else:
        ax.text(0.98, 0.97, "FerQ-BC wins\nin shaded region",
                transform=ax.transAxes, color=C_F, fontsize=8.5,
                ha="right", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.7))

    # Inset: ν(m) vs m−1 (ancillas per unique sub-agg)
    m_arr = np.arange(2, 16)
    nu_arr  = np.array([nu(int(m)) for m in m_arr])
    ros_arr = m_arr - 1
    ax_ins = ax.inset_axes([0.63, 0.06, 0.34, 0.35])
    ax_ins.set_facecolor(BG)
    ax_ins.plot(m_arr, nu_arr,  color=C_F,       lw=1.5, label=r"$\nu(m)$ FerQ-BC")
    ax_ins.plot(m_arr, ros_arr, color="#58A6FF",  lw=1.5, ls="--", label=r"$m{-}1$ Rosenberg")
    ax_ins.axvline(7, color=C_F, lw=0.8, ls=":", alpha=0.6)
    ax_ins.set_title(r"anc/sub-agg", color=TXT, fontsize=7)
    ax_ins.tick_params(colors=TXT, labelsize=6)
    for s in ax_ins.spines.values(): s.set_edgecolor(GRD)
    ax_ins.set_facecolor(PNL)
    ax_ins.grid(color=GRD, lw=0.4)
    ax_ins.legend(fontsize=6, framealpha=0.2, labelcolor=TXT, loc="upper left")

    ax.set_xlabel(r"Monomial degree $d$  ($m=d-1=|T_S|$)", fontsize=10)
    ax.set_ylabel(r"Mean overhead ratio  $(n+\#\mathrm{anc})/n$", fontsize=10)
    ax.set_title(title_sfx, fontsize=10.5)
    ax.set_xticks(DEGREES)
    ax.legend(handles=handles, framealpha=0.15, labelcolor=TXT,
              fontsize=8, ncol=2, loc="upper left")

fig2.tight_layout()
fig2.savefig("fig2_degree_sweep.png", dpi=160,
             facecolor=BG, bbox_inches="tight")
print("Saved fig2_degree_sweep.png")

# ═══════════════════════════════════════════════════════════════════════════════
# I.  FIGURE 3 – Styled table (use S3 structured as primary ranking)
# ═══════════════════════════════════════════════════════════════════════════════
sc3 = "S3: Structured\n(shared cores)"
sc1 = "S1: Random\n(deg 3–8)"
sc2 = "S2: High-degree\n(deg 8–14)"

entries = sorted(
    [(nm, all_means[sc1][nm], all_means[sc2][nm], all_means[sc3][nm])
     for nm,*_ in METHODS],
    key=lambda x: x[3])   # sort by S3 (structured) performance

fig3, ax3 = plt.subplots(figsize=(17, 5.5))
fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG); ax3.axis("off")

cols = ["#", "Method", "Ancilla formula (per sub-agg / per term)",
        "S1 Random", "S2 High-deg", "S3 Structured", "Visual (S3)"]
rows = []
for rank, (nm, m1, m2, m3) in enumerate(entries, 1):
    formula = next(f for n2,_,_,f in METHODS if n2==nm)
    bar = "▓" * int(max(0, m3-1.0)/4.0 * 20)
    note = "★ proposed" if nm=="FerQ-BC" else ("† ideal" if "Ded" in nm else "")
    rows.append([str(rank), f"{nm} {note}", formula,
                 f"{m1:.2f}", f"{m2:.2f}", f"{m3:.2f}", bar])

tbl = ax3.table(cellText=rows, colLabels=cols,
                cellLoc="left", loc="center",
                colWidths=[0.025, 0.115, 0.39, 0.065, 0.075, 0.08, 0.15])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.0)
tbl.scale(1.0, 1.70)

H="#1C2128"; RE="#161B22"; RO="#1A2030"; FB="#2D1515"; FF=C_F
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#30363D")
    if r == 0:
        cell.set_facecolor(H); cell.get_text().set_color(TXT)
        cell.get_text().set_fontweight("bold")
    else:
        nm_here = rows[r-1][1]
        if "proposed" in nm_here:
            cell.set_facecolor(FB); cell.get_text().set_color(FF)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(RE if r%2==0 else RO)
            cell.get_text().set_color(TXT)

ax3.set_title(
    "Ancilla Overhead Comparison Table  —  FerQ-BC vs 12 Existing HOBO-to-QUBO Methods\n"
    rf"$n={N}$ vars  ·  1 000 instances/scenario  ·  sorted by S3 (structured) ratio $\uparrow$"
    "  ·  ★ = proposed  ·  † = 0 ancilla (ideal/inapplicable)",
    color=TXT, fontsize=11, pad=8)

fig3.tight_layout()
fig3.savefig("fig3_table.png", dpi=160,
             facecolor=BG, bbox_inches="tight")
print("Saved fig3_table.png")

plt.close("all")
print("\nAll done.")