import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

wd='../FIBER_GEOMETRY/FORGE/' ###CHANGE###
# ── Parametri ────────────────────────────────────────────────────────────────
FILE_IN   = wd + "Vel_model_FORGE_DAS.txt"      ###CHANGE###
N_TARGET  = 5      # numero approssimativo di cluster desiderati
SMOOTH_W  = 10       # finestra di smoothing per il calcolo del trend (campioni)
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path):
    depths, vp, vs = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Depth"):
                continue
            parts = line.split()
            if len(parts) == 3:
                depths.append(float(parts[0]))
                vp.append(float(parts[1]))
                vs.append(float(parts[2]))
    return np.array(depths), np.array(vp), np.array(vs)


def moving_average(arr, w):
    """Media mobile centrata (bordi trattati con finestra ridotta)."""
    return np.convolve(arr, np.ones(w) / w, mode="same")


def compute_trend_changes(vp, vs, n_target, smooth_w):
    """
    Rileva i punti di cambio di tendenza calcolando la derivata prima
    delle curve Vp e Vs smoottate, poi seleziona i ~n_target-1 punti
    dove il cambiamento cumulato è massimo.
    """
    n = len(vp)

    vp_s = moving_average(vp, smooth_w)
    vs_s = moving_average(vs, smooth_w)

    # Derivata prima normalizzata
    dvp = np.gradient(vp_s)
    dvs = np.gradient(vs_s)
    dvp_norm = dvp / (np.max(np.abs(dvp)) + 1e-12)
    dvs_norm = dvs / (np.max(np.abs(dvs)) + 1e-12)

    # Variazione locale combinata (modulo della variazione della derivata)
    delta = np.abs(np.gradient(dvp_norm)) + np.abs(np.gradient(dvs_norm))

    # Escludi i bordi
    delta[:smooth_w] = 0
    delta[-smooth_w:] = 0

    # Seleziona i n_target-1 picchi più pronunciati come breakpoints
    n_breaks = n_target - 1
    breakpoints = []
    delta_copy = delta.copy()

    for _ in range(n_breaks):
        idx = int(np.argmax(delta_copy))
        breakpoints.append(idx)
        # sopprime il picco e i suoi vicini (min_gap = n/n_target/2)
        gap = max(3, n // (n_target * 2))
        lo = max(0, idx - gap)
        hi = min(n, idx + gap)
        delta_copy[lo:hi] = 0

    breakpoints = sorted(breakpoints)
    return breakpoints


def build_clusters(depths, vp, vs, breakpoints):
    """Costruisce i cluster e ne calcola le medie."""
    n = len(depths)
    edges = [0] + breakpoints + [n]

    clusters = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        clusters.append({
            "depth_start": float(depths[lo]),
            "depth_end":   float(depths[hi - 1]),
            "vp_mean":     float(np.mean(vp[lo:hi])),
            "vs_mean":     float(np.mean(vs[lo:hi])),
            "n_points":    hi - lo,
        })
    return clusters


def print_clusters(clusters):
    header = (f"{'#':>3}  {'Depth start':>12}  {'Depth end':>10}  "
              f"{'Vp mean':>10}  {'Vs mean':>10}  {'N pts':>6}")
    print(header)
    print("-" * len(header))
    for i, c in enumerate(clusters, 1):
        print(f"{i:>3}  {c['depth_start']:>12.2f}  {c['depth_end']:>10.2f}  "
              f"{c['vp_mean']:>10.1f}  {c['vs_mean']:>10.1f}  {c['n_points']:>6}")


def plot_clusters(depths, vp, vs, clusters, out_png="clusters_plot.png"):
    """
    Figura con due pannelli affiancati (Vp | Vs), profondità sull'asse Y
    (crescente verso il basso).  Per ogni cluster:
      - banda colorata di sfondo
      - linea verticale tratteggiata con la velocità media
      - etichetta con il numero del cluster
    """
    cmap   = get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(clusters))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, vel, label, unit in [
        (ax1, vp, "Vp", "m/s"),
        (ax2, vs, "Vs", "m/s"),
    ]:
        # curva originale
        ax.plot(vel, depths, color="0.35", lw=0.8, zorder=3, label="Dati originali")

        for i, (c, col) in enumerate(zip(clusters, colors)):
            d0, d1 = c["depth_start"], c["depth_end"]
            vmean  = c["vp_mean"] if ax is ax1 else c["vs_mean"]

            # banda di sfondo
            ax.axhspan(d0, d1, alpha=0.18, color=col, zorder=1)

            # linea media
            ax.plot([vmean, vmean], [d0, d1],
                    color=col, lw=2.2, ls="--", zorder=4)

            # etichetta cluster (centrata verticalmente nella banda)
            d_mid = (d0 + d1) / 2
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else vel.min() * 0.98,
                    d_mid, f" C{i+1}",
                    va="center", ha="left", fontsize=7.5,
                    color=col, fontweight="bold", zorder=5)

        ax.set_xlabel(f"{label}  [{unit}]", fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.set_title(label, fontsize=13, fontweight="bold")

    ax1.set_ylabel("Profondità  [m]", fontsize=11)

    # Legenda cluster
    patches = [mpatches.Patch(color=colors[i], alpha=0.7,
                               label=f"C{i+1}  {c['depth_start']:.0f}–{c['depth_end']:.0f} m")
               for i, c in enumerate(clusters)]
    fig.legend(handles=patches, title="Cluster", loc="lower center",
               ncol=5, fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

    fig.suptitle("Modello di velocità — clustering per trend\n"
                 f"({len(clusters)} cluster, {len(depths)} campioni)",
                 fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.gca().invert_yaxis()
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    print(f"Figura salvata in '{out_png}'")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
depths, vp, vs = load_data(FILE_IN)
print(f"Dati caricati: {len(depths)} campioni  "
      f"(profondità {depths[0]:.1f} – {depths[-1]:.1f} m)\n")

breakpoints = compute_trend_changes(vp, vs, N_TARGET, SMOOTH_W)
clusters    = build_clusters(depths, vp, vs, breakpoints)

print(f"Cluster trovati: {len(clusters)}  (target: {N_TARGET})\n")
print_clusters(clusters)

# ── Export CSV opzionale ───────────────────────────────────────────────────────
out_csv = wd + f"forge_vel_model_{N_TARGET}_clusters.txt"            ###CHANGE###
with open(out_csv, "w") as f:
    f.write("cluster,depth_start_m,depth_end_m,vp_mean_ms,vs_mean_ms,n_points\n")
    for i, c in enumerate(clusters, 1):
        f.write(f"{i},{c['depth_start']:.2f},{c['depth_end']:.2f},"
                f"{c['vp_mean']:.1f},{c['vs_mean']:.1f},{c['n_points']}\n")
print(f"\nRisultati salvati in '{out_csv}'")

# ── Plot ───────────────────────────────────────────────────────────────────────
plot_clusters(depths, vp, vs, clusters, out_png= wd + f"forge_vel_model_{N_TARGET}_clusters.pdf")          ###CHANGE###