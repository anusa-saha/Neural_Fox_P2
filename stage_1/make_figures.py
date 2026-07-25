"""
Generate every figure Stage I's data supports, reading only the JSON/NPZ
files written by stage1_localize.py (no model/GPU needed to run this).

Produces:
  1. fig_baseline_defaultness.png       - Fig.1-style bar chart, both channels
                                          (token-mass and LID) side by side.
  2. fig_layer_scores_heatmap_<model>.png
                                        - layer x language heatmap of mean top-K
                                          localization score (where the signal lives).
  3. fig_selectivity_vs_lift_<model>_<lang>_layer<L>.png
                                        - scatter of selectivity vs causal lift for
                                          candidate features, top-K highlighted.
  4. fig_feature_count_curve_<model>_<lang>.png
                                        - score of the top feature per layer (depth profile).
  5. fig_channel_agreement_<model>.png - layer x language heatmap of whether the
                                          combined top-K edit's mass gain and LID gain
                                          agree in direction (construct-validity check).

Run with:  python make_figures.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from config import MODELS, LANGUAGES, OUTPUT_DIR

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_summary(model_key, lang_key):
    path = os.path.join(OUTPUT_DIR, model_key, lang_key, "stage1_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig_baseline_defaultness():
    langs = list(LANGUAGES.keys())
    models = list(MODELS.keys())
    mass_vals = np.full((len(models), len(langs)), np.nan)
    lid_vals = np.full((len(models), len(langs)), np.nan)
    for mi, model_key in enumerate(models):
        for li, lang_key in enumerate(langs):
            s = load_summary(model_key, lang_key)
            if s:
                mass_vals[mi, li] = s["baseline_default_mass"]
                lid_vals[mi, li] = s["baseline_default_lid"]

    x = np.arange(len(langs))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for ax, vals, title in zip(axes, [mass_vals, lid_vals],
                                ["Token-mass channel  M_target - M_english",
                                 "LID channel  P(target) - P(english)"]):
        for mi, model_key in enumerate(models):
            ax.bar(x + mi * width, vals[mi], width, label=model_key)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(langs)
        ax.set_title(title)
    axes[0].set_ylabel("mean defaultness (no edit, weak prompts, T=1..3)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Baseline English-default bias - both channels")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_baseline_defaultness.png"), dpi=150)
    plt.close(fig)


def fig_layer_scores_heatmap(model_key):
    langs = list(LANGUAGES.keys())
    layer_keys = None
    grid_rows = []
    for lang_key in langs:
        s = load_summary(model_key, lang_key)
        if s is None:
            grid_rows.append(None)
            continue
        layers_sorted = sorted(s["layers"].keys(), key=int)
        if layer_keys is None:
            layer_keys = layers_sorted
        row = [np.mean(list(s["layers"][l]["score"].values())) if s["layers"][l]["score"] else 0.0
               for l in layers_sorted]
        grid_rows.append(row)

    if layer_keys is None:
        return
    grid = np.array([r if r is not None else [np.nan] * len(layer_keys) for r in grid_rows])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)
    ax.set_xticks(range(len(layer_keys)))
    ax.set_xticklabels(layer_keys, rotation=90)
    ax.set_xlabel("layer")
    ax.set_title(f"Mean language-neuron score per layer/language - {model_key}")
    fig.colorbar(im, ax=ax, label="mean Score(l,j) over candidates")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig_layer_scores_heatmap_{model_key}.png"), dpi=150)
    plt.close(fig)


def fig_selectivity_vs_lift(model_key, lang_key):
    s = load_summary(model_key, lang_key)
    if s is None:
        return
    for layer_key, data in s["layers"].items():
        sel = data["selectivity"]
        lift = data["lift"]
        top = set(data["top_features"])
        if not sel:
            continue
        ids = list(sel.keys())
        xs = [sel[i] for i in ids]
        ys = [lift[i] for i in ids]
        colors = ["tab:red" if int(i) in top else "tab:blue" for i in ids]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys, c=colors, alpha=0.7, s=18)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("selectivity  Sel_j")
        ax.set_ylabel("causal lift  Lift_j  (mass channel, T=1..3)")
        ax.set_title(f"{model_key} / {lang_key} / layer {layer_key}\n(red = selected N_lt)")
        fig.tight_layout()
        fname = f"fig_selectivity_vs_lift_{model_key}_{lang_key}_layer{layer_key}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
        plt.close(fig)


def fig_feature_count_curve(model_key, lang_key):
    s = load_summary(model_key, lang_key)
    if s is None:
        return
    layers_sorted = sorted(s["layers"].keys(), key=int)
    top_scores = [
        max(s["layers"][l]["score"].values()) if s["layers"][l]["score"] else 0.0
        for l in layers_sorted
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([int(l) for l in layers_sorted], top_scores, marker="o")
    ax.set_xlabel("layer")
    ax.set_ylabel("max Score(l,j) among candidates")
    ax.set_title(f"Depth profile of language-neuron strength - {model_key} / {lang_key}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig_feature_count_curve_{model_key}_{lang_key}.png"), dpi=150)
    plt.close(fig)


def fig_channel_agreement(model_key):
    langs = list(LANGUAGES.keys())
    layer_keys = None
    grid_rows = []
    for lang_key in langs:
        s = load_summary(model_key, lang_key)
        if s is None:
            grid_rows.append(None)
            continue
        layers_sorted = sorted(s["layers"].keys(), key=int)
        if layer_keys is None:
            layer_keys = layers_sorted
        row = [1.0 if s["layers"][l]["channels_agree"] else 0.0 for l in layers_sorted]
        grid_rows.append(row)

    if layer_keys is None:
        return
    grid = np.array([r if r is not None else [np.nan] * len(layer_keys) for r in grid_rows])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)
    ax.set_xticks(range(len(layer_keys)))
    ax.set_xticklabels(layer_keys, rotation=90)
    ax.set_xlabel("layer")
    ax.set_title(f"Mass <-> LID direction agreement (combined top-K edit) - {model_key}")
    fig.colorbar(im, ax=ax, label="1 = channels agree, 0 = disagree")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig_channel_agreement_{model_key}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_baseline_defaultness()
    for model_key in MODELS:
        fig_layer_scores_heatmap(model_key)
        fig_channel_agreement(model_key)
        for lang_key in LANGUAGES:
            fig_selectivity_vs_lift(model_key, lang_key)
            fig_feature_count_curve(model_key, lang_key)
    print("Figures written to:", FIG_DIR)
