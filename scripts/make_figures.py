"""Turn the raw result tables into the figures and summary tables of the README.

Usage
-----
    python scripts/make_figures.py                 # reads results/, writes figures/

Nothing is recomputed here: every number comes from ``results/*.csv``, and every
model choice is re-derived from the stored cross-validation scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from qksvm.analysis import (
    CLASSICAL_FAMILIES,
    FAMILIES,
    QUANTUM_FAMILIES,
    haar_fidelity_moments,
    paired_difference,
    select_by_cv,
    summarise,
)

# Okabe-Ito colour-blind-safe palette.
COLOURS = {
    "Linear": "#999999",
    "Polynomial": "#E69F00",
    "RBF": "#000000",
    "Z": "#56B4E9",
    "ZZ-linear": "#009E73",
    "ZZ-full": "#D55E00",
}
DIMENSION_COLOURS = {4: "#56B4E9", 6: "#0072B2", 8: "#CC79A7"}
MARKERS = {"Linear": "v", "Polynomial": "D", "RBF": "s", "Z": "o", "ZZ-linear": "^", "ZZ-full": "P"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
    }
)


def errorbar(ax: plt.Axes, summary: pd.DataFrame, x: str, **style: object) -> None:
    """Plot mean +/- 95% CI from a :func:`qksvm.analysis.summarise` table."""
    ax.errorbar(summary[x], summary["mean"], yerr=summary["ci95"], capsize=2.5, **style)


def figure_accuracy(sweep: pd.DataFrame, n_train: int, out: Path) -> None:
    """Test accuracy of every tuned kernel family at each PCA dimension."""
    selected = select_by_cv(sweep[sweep.n_train == n_train])
    dimensions = sorted(selected.n_features.unique())
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    width = 0.8 / len(dimensions)
    for position, n_features in enumerate(dimensions):
        offset = (position - (len(dimensions) - 1) / 2) * width
        subset = selected[selected.n_features == n_features]
        for index, family in enumerate(FAMILIES):
            values = subset.loc[subset.family == family, "test_accuracy"].to_numpy()
            x = index + offset
            ax.scatter(
                np.full(len(values), x),
                values,
                s=7,
                color=DIMENSION_COLOURS[n_features],
                alpha=0.3,
                linewidths=0,
            )
        summary = summarise(subset, by=("family",)).set_index("family").loc[list(FAMILIES)]
        ax.errorbar(
            np.arange(len(FAMILIES)) + offset,
            summary["mean"],
            yerr=summary["ci95"],
            fmt="o",
            capsize=2.5,
            color=DIMENSION_COLOURS[n_features],
            label=f"{n_features} PCA features",
        )
    ax.axvline(len(CLASSICAL_FAMILIES) - 0.5, color="0.6", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(FAMILIES)), FAMILIES)
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        f"Tuned kernels, {n_train} training points (mean and 95% CI over training subsamples)",
        pad=16,
    )
    ax.text(0.25, 1.01, "classical", transform=ax.transAxes, ha="center", color="0.4")
    ax.text(0.75, 1.01, "quantum (simulated)", transform=ax.transAxes, ha="center", color="0.4")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "accuracy_by_kernel.png")
    plt.close(fig)


def figure_depth(sweep: pd.DataFrame, n_train: int, out: Path) -> None:
    """Accuracy against feature-map depth, with and without bandwidth tuning."""
    data = sweep[sweep.n_train == n_train]
    dimensions = sorted(data.n_features.unique())
    fig, axes = plt.subplots(
        len(dimensions),
        len(QUANTUM_FAMILIES),
        figsize=(7.4, 1.9 * len(dimensions) + 0.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    tuned = select_by_cv(data, within=("family", "reps"))
    best_classical = select_by_cv(data)
    for row, n_features in enumerate(dimensions):
        rbf = summarise(
            best_classical[
                (best_classical.family == "RBF") & (best_classical.n_features == n_features)
            ],
            by=("family",),
        ).iloc[0]
        for column, family in enumerate(QUANTUM_FAMILIES):
            ax = axes[row, column]
            ax.axhspan(rbf["mean"] - rbf["ci95"], rbf["mean"] + rbf["ci95"], color="0.85")
            ax.axhline(rbf["mean"], color=COLOURS["RBF"], linewidth=1.0, label="RBF (tuned)")
            mask = (data.family == family) & (data.n_features == n_features)
            fixed = summarise(data[mask & (data.bandwidth == 1.0)], by=("reps",))
            errorbar(
                ax,
                fixed,
                "reps",
                color=COLOURS[family],
                linestyle="--",
                marker="x",
                alpha=0.75,
                label="bandwidth c = 1 (fixed)",
            )
            chosen = summarise(
                tuned[(tuned.family == family) & (tuned.n_features == n_features)], by=("reps",)
            )
            errorbar(
                ax,
                chosen,
                "reps",
                color=COLOURS[family],
                marker=MARKERS[family],
                label="bandwidth tuned by CV",
            )
            ax.axhline(0.5, color="0.5", linewidth=0.8, linestyle=":")
            if row == 0:
                ax.set_title(f"{family} feature map")
            if column == 0:
                ax.set_ylabel(f"{n_features} features\nTest accuracy")
            if row == len(dimensions) - 1:
                ax.set_xlabel("Depth (repetitions L)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(name) for name in ("bandwidth tuned by CV", "bandwidth c = 1 (fixed)")]
    order.append(labels.index("RBF (tuned)"))
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "accuracy_vs_depth.png")
    plt.close(fig)


def figure_heatmaps(sweep: pd.DataFrame, n_features: int, n_train: int, out: Path) -> None:
    """Accuracy, alignment and concentration over the (depth, bandwidth) plane."""
    data = sweep[(sweep.n_features == n_features) & (sweep.n_train == n_train)]
    panels = [
        ("test_accuracy", "Test accuracy", "viridis", lambda v: v, "{:.2f}"),
        ("alignment", "Centred kernel-target alignment", "viridis", lambda v: v, "{:.2f}"),
        ("offdiag_var", r"$\log_{10}$ Var[off-diagonal $k$]", "magma", np.log10, "{:.1f}"),
    ]
    fig, axes = plt.subplots(
        len(QUANTUM_FAMILIES), len(panels), figsize=(10.0, 6.3), sharex=True, sharey=True
    )
    for row, family in enumerate(QUANTUM_FAMILIES):
        subset = data[data.family == family]
        for column, (value, title, cmap, transform, fmt) in enumerate(panels):
            ax = axes[row, column]
            grid = transform(
                subset.pivot_table(index="reps", columns="bandwidth", values=value, aggfunc="mean")
            )
            image = ax.imshow(grid.to_numpy(), aspect="auto", cmap=cmap, origin="lower")
            threshold = np.nanmin(grid.to_numpy()) + 0.6 * np.ptp(grid.to_numpy())
            for (i, j), cell in np.ndenumerate(grid.to_numpy()):
                ax.text(
                    j,
                    i,
                    fmt.format(cell),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black" if cell > threshold else "white",
                )
            ax.set_xticks(range(len(grid.columns)), [f"{c:.3g}" for c in grid.columns], rotation=60)
            ax.set_yticks(range(len(grid.index)), [int(r) for r in grid.index])
            ax.grid(visible=False)
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"{family}\nDepth L")
            if row == len(QUANTUM_FAMILIES) - 1:
                ax.set_xlabel("Bandwidth c")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{n_features} qubits, {n_train} training points (mean over training subsamples)", y=0.995
    )
    fig.tight_layout()
    fig.savefig(out / "bandwidth_depth_heatmaps.png")
    plt.close(fig)


def figure_concentration(sweep: pd.DataFrame, n_train: int, out: Path) -> None:
    """Off-diagonal kernel statistics against depth at the untuned bandwidth c = 1."""
    data = sweep[(sweep.n_train == n_train) & (sweep.bandwidth == 1.0)]
    dimensions = sorted(data.n_features.unique())
    fig, axes = plt.subplots(
        2, len(QUANTUM_FAMILIES), figsize=(7.4, 4.3), sharex=True, sharey="row"
    )
    for column, family in enumerate(QUANTUM_FAMILIES):
        for n_features in dimensions:
            subset = data[(data.family == family) & (data.n_features == n_features)]
            haar_mean, haar_variance = haar_fidelity_moments(n_features)
            colour = DIMENSION_COLOURS[n_features]
            for row, (value, haar) in enumerate(
                (("offdiag_mean", haar_mean), ("offdiag_var", haar_variance))
            ):
                curve = subset.groupby("reps")[value].mean()
                axes[row, column].plot(
                    curve.index, curve, marker="o", color=colour, label=f"{n_features} qubits"
                )
                axes[row, column].axhline(haar, color=colour, linestyle=":", linewidth=1.0)
        axes[0, column].set_title(f"{family} feature map")
        axes[1, column].set_xlabel("Depth (repetitions L)")
    for ax in axes.ravel():
        ax.set_yscale("log")
    axes[0, 0].set_ylabel("Mean of off-diagonal $k$")
    axes[1, 0].set_ylabel("Variance of off-diagonal $k$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(dimensions), fontsize=8)
    fig.suptitle("Kernel concentration at bandwidth c = 1 (dotted: Haar-random states)", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "concentration.png")
    plt.close(fig)


def figure_alignment(sweep: pd.DataFrame, n_features: int, n_train: int, out: Path) -> pd.DataFrame:
    """Plot test accuracy against kernel-target alignment and return rank correlations."""
    data = sweep[(sweep.n_features == n_features) & (sweep.n_train == n_train)]
    hyper = ["family", "gamma", "degree", "bandwidth", "reps"]
    points = data.groupby(hyper, dropna=False)[["alignment", "test_accuracy"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    rows = []
    for family in FAMILIES:
        subset = points[points.family == family]
        ax.scatter(
            subset["alignment"],
            subset["test_accuracy"],
            s=16,
            color=COLOURS[family],
            marker=MARKERS[family],
            alpha=0.8,
            linewidths=0,
            label=family,
        )
        if len(subset) > 2:
            rho = stats.spearmanr(subset["alignment"], subset["test_accuracy"]).statistic
            rows.append({"family": family, "candidates": len(subset), "spearman": rho})
    overall = stats.spearmanr(points["alignment"], points["test_accuracy"]).statistic
    rows.append({"family": "all kernels", "candidates": len(points), "spearman": overall})
    ax.set_xlabel("Centred kernel-target alignment (training set)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"One point per candidate kernel ({n_features} features, N = {n_train})")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(out / "alignment_vs_accuracy.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def figure_learning_and_geometry(
    sweep: pd.DataFrame,
    geometry: pd.DataFrame,
    n_features: int,
    n_train: int,
    reg: float,
    out: Path,
) -> None:
    """Learning curves and the geometric difference to the tuned RBF kernel."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.1))

    curve = summarise(select_by_cv(sweep[sweep.n_features == n_features]), by=("family", "n_train"))
    for family in FAMILIES:
        errorbar(
            left,
            curve[curve.family == family].sort_values("n_train"),
            "n_train",
            color=COLOURS[family],
            marker=MARKERS[family],
            label=family,
            linewidth=1.2,
        )
    left.set_xscale("log", base=2)
    left.set_xlabel("Training points N")
    left.set_ylabel("Test accuracy")
    left.set_title(f"Learning curves ({n_features} features)")
    left.legend(fontsize=7.5, ncols=2)

    data = geometry[
        (geometry.n_features == n_features)
        & (geometry.n_train == n_train)
        & np.isclose(geometry.reg, reg)
    ]
    for family in QUANTUM_FAMILIES:
        subset = data[data.family == family]
        median = subset.groupby("reps")["geometric_difference"].median()
        low = subset.groupby("reps")["geometric_difference"].quantile(0.25)
        high = subset.groupby("reps")["geometric_difference"].quantile(0.75)
        right.plot(
            median.index, median, color=COLOURS[family], marker=MARKERS[family], label=family
        )
        right.fill_between(median.index, low, high, color=COLOURS[family], alpha=0.15, linewidth=0)
    right.axhline(np.sqrt(n_train), color="0.3", linestyle="--", linewidth=1.0)
    right.text(1.0, np.sqrt(n_train) * 0.72, r"$\sqrt{N}$", color="0.3")
    right.axhline(1.0, color="0.3", linestyle=":", linewidth=1.0)
    right.set_yscale("log")
    right.set_xlabel("Depth (repetitions L)")
    right.set_ylabel(r"$g(K_{\mathrm{RBF}} \,\|\, K_Q)$")
    right.set_title(f"Geometric difference (N = {n_train}, median and IQR)")
    right.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "learning_curve_and_geometry.png")
    plt.close(fig)


def figure_shots(shots: pd.DataFrame, n_train: int, out: Path) -> None:
    """Accuracy of the selected quantum kernels under finite-shot estimation."""
    data = shots[shots.n_train == n_train]
    dimensions = sorted(data.n_features.unique())
    fig, axes = plt.subplots(1, len(dimensions), figsize=(7.4, 2.8), sharey=True, squeeze=False)
    for ax, n_features in zip(axes[0], dimensions, strict=True):
        subset = data[data.n_features == n_features]
        for family in QUANTUM_FAMILIES:
            rows = subset[subset.family == family]
            finite = summarise(rows[np.isfinite(rows.shots)], by=("shots",)).sort_values("shots")
            errorbar(
                ax, finite, "shots", color=COLOURS[family], marker=MARKERS[family], label=family
            )
            exact = rows.loc[~np.isfinite(rows.shots), "test_accuracy"].mean()
            ax.axhline(exact, color=COLOURS[family], linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Shots per kernel entry")
        ax.set_title(f"{n_features} qubits")
    axes[0, 0].set_ylabel("Test accuracy")
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Finite-shot kernel estimation (dotted: exact kernel)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "shot_noise.png")
    plt.close(fig)


def _markdown(table: pd.DataFrame, **kwargs: object) -> str:
    """Render a table as Markdown with its index levels as ordinary columns."""
    return table.reset_index().to_markdown(index=False, **kwargs)


def _family_order(table: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by the canonical family order, then by the remaining columns."""
    rank = table["family"].map({family: i for i, family in enumerate(FAMILIES)})
    others = [c for c in ("n_features", "setting") if c in table.columns]
    return table.assign(_rank=rank).sort_values(["_rank", *others]).drop(columns="_rank")


def write_summary(
    sweep: pd.DataFrame,
    geometry: pd.DataFrame,
    shots: pd.DataFrame,
    correlations: pd.DataFrame,
    n_train: int,
    path: Path,
) -> None:
    """Write the numeric tables quoted in the README as Markdown."""
    main = sweep[sweep.n_train == n_train]
    selected = select_by_cv(main)
    quantum = main[main.family.isin(QUANTUM_FAMILIES)]
    lines = [
        "# Summary tables",
        "",
        "Generated by `scripts/make_figures.py` from `results/*.csv`. Do not edit by hand.",
        "",
        f"## Test accuracy of tuned kernels (N = {n_train}; mean ± std over seeds, in %)",
        "",
    ]
    summary = summarise(selected, by=("family", "n_features"))
    summary["cell"] = [
        f"{100 * m:.1f} ± {100 * s:.1f}"
        for m, s in zip(summary["mean"], summary["std"], strict=True)
    ]
    table = summary.pivot(index="family", columns="n_features", values="cell").loc[list(FAMILIES)]
    table.columns = [f"{d} features" for d in table.columns]
    lines += [_markdown(table), ""]

    lines += [
        "## Paired difference to the tuned RBF kernel (percentage points)",
        "",
        "Same training subsample and test set for both models; mean ± 95% CI half-width "
        "over seeds; W/T/L = seeds won/tied/lost; two-sided Wilcoxon signed-rank p-value "
        "(uncorrected for multiple comparisons).",
        "",
    ]
    paired = pd.concat(
        [paired_difference(selected, family) for family in FAMILIES if family != "RBF"]
    )
    paired["difference"] = [
        f"{100 * m:+.2f} ± {100 * c:.2f}"
        for m, c in zip(paired["mean_difference"], paired["ci95"], strict=True)
    ]
    paired["W/T/L"] = [
        f"{w}/{t}/{lo}" for w, t, lo in zip(paired.wins, paired.ties, paired.losses, strict=True)
    ]
    paired["p"] = paired["p_wilcoxon"].map("{:.3f}".format)
    lines += [
        paired[["family", "n_features", "difference", "W/T/L", "p"]].to_markdown(index=False),
        "",
    ]

    lines += [f"## Accuracy against depth (N = {n_train}; mean over seeds, in %)", ""]
    by = ("family", "n_features", "reps")
    tuned = summarise(select_by_cv(quantum, within=("family", "reps")), by=by)
    fixed = summarise(quantum[quantum.bandwidth == 1.0], by=by)
    depth = pd.concat([tuned.assign(setting="tuned c"), fixed.assign(setting="c = 1")])
    depth["mean"] = (100 * depth["mean"]).map("{:.1f}".format)
    depth["reps"] = "L = " + depth["reps"].astype(int).astype(str)
    depth = depth.pivot(index=["family", "n_features", "setting"], columns="reps", values="mean")
    lines += [_family_order(depth.reset_index()).to_markdown(index=False), ""]

    lines += ["## Selected hyper-parameters (median over seeds)", ""]
    chosen = selected.groupby(["family", "n_features"])[
        ["C", "gamma", "degree", "bandwidth", "reps"]
    ].median()
    lines += [_family_order(chosen.reset_index()).to_markdown(index=False, floatfmt=".3g"), ""]

    lines += ["## Fraction of seeds whose selected model sits on a grid edge", ""]
    edge_rows = []
    for family in FAMILIES:
        chosen_rows = selected[selected.family == family]
        entry: dict[str, object] = {"family": family}
        for name in ("C", "gamma", "bandwidth"):
            grid = main.loc[main.family == family, name].dropna()
            if len(grid):
                entry[f"{name} = min"] = float(np.isclose(chosen_rows[name], grid.min()).mean())
                entry[f"{name} = max"] = float(np.isclose(chosen_rows[name], grid.max()).mean())
        edge_rows.append(entry)
    lines += [pd.DataFrame(edge_rows).to_markdown(index=False, floatfmt=".2f"), ""]

    lines += [
        "## Spearman correlation between alignment and test accuracy",
        "",
        f"Across candidate kernels ({int(main.n_features.max())} features, N = {n_train}; "
        "both quantities averaged over seeds).",
        "",
        correlations.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]

    lines += [
        f"## Geometric difference to the tuned RBF kernel (N = {n_train}, "
        f"sqrt(N) = {np.sqrt(n_train):.1f})",
        "",
        "g(K_RBF || K_Q) at the cross-validated bandwidth, pooled over seeds and depths, "
        "for each ridge parameter.",
        "",
    ]
    pooled = geometry[(geometry.n_train == n_train) & geometry.family.isin(QUANTUM_FAMILIES)]
    for label, aggregate in (("Median", "median"), ("Maximum", "max")):
        table = pooled.pivot_table(
            index=["family", "n_features"],
            columns="reg",
            values="geometric_difference",
            aggfunc=aggregate,
        )
        table.columns = [f"ridge {reg:g}" for reg in table.columns]
        lines += [
            f"{label}:",
            "",
            _family_order(table.reset_index()).to_markdown(index=False, floatfmt=".2f"),
            "",
        ]

    lines += [f"## Finite-shot accuracy (N = {n_train}; mean over seeds, in %)", ""]
    noisy = shots[shots.n_train == n_train].copy()
    noisy["shots"] = noisy["shots"].map(lambda s: "exact" if np.isinf(s) else f"{int(s)}")
    order = sorted(noisy["shots"].unique(), key=lambda s: np.inf if s == "exact" else int(s))
    table = 100 * noisy.pivot_table(
        index=["family", "n_features"], columns="shots", values="test_accuracy", aggfunc="mean"
    )
    lines += [
        _family_order(table[order].reset_index()).to_markdown(index=False, floatfmt=".1f"),
        "",
    ]

    n_features = int(sweep.n_features.max())
    lines += [
        f"## Learning curve ({n_features} features; mean ± std over seeds, in %)",
        "",
        "Columns are the number of training points N.",
        "",
    ]
    curve = summarise(select_by_cv(sweep[sweep.n_features == n_features]), by=("family", "n_train"))
    curve["cell"] = [
        f"{100 * m:.1f} ± {100 * s:.1f}" for m, s in zip(curve["mean"], curve["std"], strict=True)
    ]
    table = curve.pivot(index="family", columns="n_train", values="cell").loc[list(FAMILIES)]
    lines += [_markdown(table), ""]
    path.write_text("\n".join(lines))


def main() -> None:
    """Read the result tables and write all figures and the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--n-train", type=int, default=512, help="training size of the main study")
    parser.add_argument("--reg", type=float, default=0.01, help="ridge of the geometry figure")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv(args.results / "sweep.csv")
    geometry = pd.read_csv(args.results / "geometry.csv")
    shots = pd.read_csv(args.results / "shots.csv")
    n_features = int(sweep.n_features.max())

    figure_accuracy(sweep, args.n_train, args.out)
    figure_depth(sweep, args.n_train, args.out)
    figure_heatmaps(sweep, n_features, args.n_train, args.out)
    figure_concentration(sweep, args.n_train, args.out)
    correlations = figure_alignment(sweep, n_features, args.n_train, args.out)
    figure_learning_and_geometry(sweep, geometry, n_features, args.n_train, args.reg, args.out)
    figure_shots(shots, args.n_train, args.out)
    write_summary(sweep, geometry, shots, correlations, args.n_train, args.results / "summary.md")
    print(f"Wrote figures to {args.out}/ and tables to {args.results / 'summary.md'}")


if __name__ == "__main__":
    main()
