"""
Pooled ground-truth response histogram for choosing the active threshold tau.
Reads data/CC/raw/CC_reformat_z.csv (column 'output' = global z-scored response).
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

scratch = sys.argv[1] if len(sys.argv) > 1 else "."
CSV = "data/CC/raw/CC_reformat_z.csv"
REC_TAU = 2.0          # recommended threshold (highlighted)
N_MIN = 5              # min actives for a receptor to be "evaluable"

df = pd.read_csv(CSV)
o = df['output'].to_numpy()
N = len(o)
g = df.groupby('RECEPTOR')['output']

# ---- diagnostics table ----
taus = [1.0, 1.5, 2.0, 2.5, 3.0]
rows = []
for t in taus:
    n_act = int((o >= t).sum())
    per_rec = g.apply(lambda s: (s >= t).sum())
    rows.append((t, n_act, 100*n_act/N, int((per_rec >= 5).sum()),
                 int((per_rec >= 10).sum())))

# ---- figure ----
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': '#888',
                     'axes.linewidth': 0.8, 'font.family': 'DejaVu Sans'})
BAR='#3a6ea5'; TAUC='#c0392b'; GRAYT='#9aa0a6'; SHADE='#f6d9d4'; KDEC='#20344a'
bins = np.arange(-2, 6.55, 0.1)

try:
    from scipy.stats import gaussian_kde
    grid = np.linspace(-2, 6.5, 1000)
    dens = gaussian_kde(o, bw_method=0.15)(grid)
except Exception:
    grid = dens = None

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8.4),
                               gridspec_kw={'height_ratios':[1,1], 'hspace':0.10})

def style(ax):
    ax.grid(axis='y', color='#eeeeee', lw=0.8); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)

# --- Panel A: linear ---
ax1.hist(o, bins=bins, color=BAR, edgecolor='white', linewidth=0.3)
if dens is not None:
    ax1.plot(grid, dens*N*0.1, color=KDEC, lw=2)
ax1.set_ylabel('count (odorant–receptor pairs)')
ax1.text(0.012, 0.95, 'linear scale — the inactive block near 0 dominates',
         transform=ax1.transAxes, ha='left', va='top', color='#555', fontsize=11,
         style='italic')
ax1.annotate('inactive mode\n≈ −0.35', xy=(-0.35, 1380), xytext=(-1.9, 1050),
             fontsize=10, color='#333', va='top',
             arrowprops=dict(arrowstyle='->', color='#777', lw=1))
style(ax1)

# diagnostics table in the empty right region of panel A
tbl  = "  τ    % active   evaluable receptors\n"
tbl += "                   (n_act ≥ 5)\n"
tbl += "  " + "─"*34 + "\n"
for t, n_act, pct, r5, r10 in rows:
    mark = "  ◀ rec." if t == REC_TAU else ""
    tbl += f"  {t:>3.1f}   {pct:>5.1f}%       {r5:>2d} / 50{mark}\n"
ax1.text(0.42, 0.90, tbl, transform=ax1.transAxes, ha='left', va='top',
         fontsize=10.5, family='monospace', color='#222',
         bbox=dict(boxstyle='round,pad=0.6', fc='#fbfbfb', ec='#ccc', lw=1))

# --- Panel B: log ---
ax2.hist(o, bins=bins, color=BAR, edgecolor='white', linewidth=0.3)
ax2.set_yscale('log')
ax2.set_ylabel('count (log scale)')
ax2.set_xlabel('output  (global z-scored response)')
ax2.text(0.012, 0.95, 'log scale — the active tail (z ≈ 1–6) is now visible',
         transform=ax2.transAxes, ha='left', va='top', color='#555', fontsize=11,
         style='italic')
ax2.axvspan(REC_TAU, bins[-1], color=SHADE, alpha=0.5, zorder=0)
style(ax2)

# candidate tau lines + rotated labels at the top of the log panel
ytop = ax2.get_ylim()[1]
for t, *_ in rows:
    chosen = (t == REC_TAU)
    for ax in (ax1, ax2):
        ax.axvline(t, color=TAUC if chosen else GRAYT, ls='--',
                   lw=2.2 if chosen else 1.0, alpha=0.95 if chosen else 0.65, zorder=5)
    ax2.text(t, ytop*0.6, f'τ={t:g}', rotation=90, ha='right', va='top',
             color=TAUC if chosen else '#777', fontsize=10,
             weight='bold' if chosen else 'normal')

ax2.annotate('recommended\nτ = 2.0  (5.3% active)', xy=(2.0, 30), xytext=(3.1, 300),
             fontsize=10.5, color=TAUC, weight='bold', va='center',
             arrowprops=dict(arrowstyle='->', color=TAUC, lw=1.4))

ax1.set_xlim(-2, 6.5); ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
fig.suptitle('Ground-truth response distribution  →  choosing the active threshold τ',
             fontsize=14, weight='bold', x=0.5, y=0.965)

out = f"{scratch}/threshold_histogram.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print("saved:", out)
