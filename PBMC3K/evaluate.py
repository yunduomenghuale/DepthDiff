import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config
from diffusion import make_diffusion
from models import make_baseline, make_depthdiff
from train_baselines import BASELINES
from utils import depth_to_name, ensure_dirs, get_device, load_processed, pearson_flat, rmse, set_seed


@torch.no_grad()
def predict_baseline(name, model, x_low, depth, device):
    model.eval()
    preds = []
    for start in range(0, x_low.shape[0], config.BATCH_SIZE):
        batch = torch.from_numpy(x_low[start : start + config.BATCH_SIZE]).to(device)
        depths = torch.full((batch.size(0),), float(depth), device=device)
        output = model(batch, depths)
        if name == "cvae":
            output = output[0]
        output = torch.clamp(output, min=0.0)
        preds.append(output.cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


@torch.no_grad()
def predict_depthdiff(model, diffusion, x_low, depth, device):
    """Reverse-diffusion enhancement.

    Runs the conditional reverse process and returns the posterior-mean estimate
    over ``config.EVAL_NUM_SAMPLES`` independent samples. Averaging multiple
    samples gives a Monte-Carlo MMSE estimate that is appropriate for point
    metrics (RMSE / Pearson); a single sample better preserves gene variance.
    """
    model.eval()
    n_genes = x_low.shape[1]
    preds = []
    for start in range(0, x_low.shape[0], config.BATCH_SIZE):
        low = torch.from_numpy(x_low[start : start + config.BATCH_SIZE]).to(device)
        depths = torch.full((low.size(0),), float(depth), device=device)
        acc = torch.zeros_like(low)
        for _ in range(config.EVAL_NUM_SAMPLES):
            acc = acc + torch.clamp(diffusion.sample(model, low, depths, n_genes), min=0.0)
        preds.append((acc / config.EVAL_NUM_SAMPLES).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def load_baselines(n_genes, device):
    models = {}
    for name in BASELINES:
        path = config.CHECKPOINT_DIR / f"{config.BENCHMARK_NAME}_baseline_{name}_best.pt"
        if path.exists():
            model = make_baseline(name, n_genes).to(device)
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            models[name] = model
        else:
            warnings.warn(f"Missing checkpoint: {path}")
    return models


def load_depthdiff(n_genes, device):
    path = config.CHECKPOINT_DIR / f"{config.BENCHMARK_NAME}_{config.DEPTHDIFF_METHOD_NAME}_best.pt"
    if not path.exists():
        warnings.warn(f"Missing checkpoint: {path}")
        return None
    model = make_depthdiff(n_genes).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def cluster_metrics(target, pred, n_clusters=8):
    high_labels = KMeans(n_clusters=n_clusters, random_state=config.SEED, n_init=20).fit_predict(target)
    pred_labels = KMeans(n_clusters=n_clusters, random_state=config.SEED, n_init=20).fit_predict(pred)
    return {
        "cluster_ari": adjusted_rand_score(high_labels, pred_labels),
        "cluster_nmi": normalized_mutual_info_score(high_labels, pred_labels),
        "high_labels": high_labels,
    }


def prediction_cluster_metrics(reference_labels, pred, n_clusters=8):
    pred_labels = KMeans(n_clusters=n_clusters, random_state=config.SEED, n_init=20).fit_predict(pred)
    return {
        "cluster_ari": adjusted_rand_score(reference_labels, pred_labels),
        "cluster_nmi": normalized_mutual_info_score(reference_labels, pred_labels),
    }


def marker_metrics(target, pred, labels, top_k=50):
    high_effects = []
    pred_effects = []
    high_top = set()
    pred_top = set()
    for cluster in np.unique(labels):
        mask = labels == cluster
        high_effect = target[mask].mean(axis=0) - target[~mask].mean(axis=0)
        pred_effect = pred[mask].mean(axis=0) - pred[~mask].mean(axis=0)
        high_effects.append(high_effect)
        pred_effects.append(pred_effect)
        high_top.update(np.argsort(-high_effect)[:top_k].tolist())
        pred_top.update(np.argsort(-pred_effect)[:top_k].tolist())
    high_effects = np.concatenate(high_effects)
    pred_effects = np.concatenate(pred_effects)
    corr = stats.spearmanr(high_effects, pred_effects).correlation
    overlap = len(high_top & pred_top) / max(len(high_top), 1)
    return float(corr), float(overlap)


def deg_overlap(target, pred, labels, top_k=100):
    group = labels == np.bincount(labels).argmax()
    high_t = stats.ttest_ind(target[group], target[~group], axis=0, equal_var=False).statistic
    pred_t = stats.ttest_ind(pred[group], pred[~group], axis=0, equal_var=False).statistic
    high_t = np.nan_to_num(np.abs(high_t))
    pred_t = np.nan_to_num(np.abs(pred_t))
    high_top = set(np.argsort(-high_t)[:top_k].tolist())
    pred_top = set(np.argsort(-pred_t)[:top_k].tolist())
    return len(high_top & pred_top) / top_k


def logfc_metrics(target, pred, labels):
    group = labels == np.bincount(labels).argmax()
    high_logfc = target[group].mean(axis=0) - target[~group].mean(axis=0)
    pred_logfc = pred[group].mean(axis=0) - pred[~group].mean(axis=0)
    pearson = pearson_flat(high_logfc, pred_logfc)
    spearman = stats.spearmanr(high_logfc, pred_logfc).correlation

    high_t = stats.ttest_ind(target[group], target[~group], axis=0, equal_var=False).statistic
    pred_t = stats.ttest_ind(pred[group], pred[~group], axis=0, equal_var=False).statistic
    high_t = np.nan_to_num(high_t)
    pred_t = np.nan_to_num(pred_t)
    top_idx = np.argsort(-np.abs(high_t))[:100]
    direction = np.mean(np.sign(high_logfc[top_idx]) == np.sign(pred_logfc[top_idx]))
    return float(pearson), float(spearman), float(direction)


def module_score_pearson(target, pred, labels, genes_per_cluster=30):
    high_scores = []
    pred_scores = []
    for cluster in np.unique(labels):
        mask = labels == cluster
        high_effect = target[mask].mean(axis=0) - target[~mask].mean(axis=0)
        module_idx = np.argsort(-high_effect)[:genes_per_cluster]
        high_scores.append(target[:, module_idx].mean(axis=1))
        pred_scores.append(pred[:, module_idx].mean(axis=1))
    high_scores = np.vstack(high_scores).T
    pred_scores = np.vstack(pred_scores).T
    return pearson_flat(high_scores, pred_scores)


def plot_metric_bars(metrics):
    df = pd.DataFrame(metrics)
    for metric in [
        "rmse",
        "pearson",
        "cluster_ari",
        "marker_spearman",
        "deg_top100_overlap",
        "logfc_pearson",
        "module_score_pearson",
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        pivot = df.pivot(index="depth", columns="method", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Sequencing depth")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / f"{metric}.png", dpi=180)
        plt.close(fig)


def plot_embedding(target, predictions, depth):
    methods = ["high_depth"] + list(predictions.keys())
    matrices = [target] + [predictions[m] for m in predictions]
    max_cells = min(500, target.shape[0])
    rng = np.random.default_rng(config.SEED)
    take = rng.choice(target.shape[0], max_cells, replace=False)
    stacked = np.vstack([m[take] for m in matrices])
    labels = np.repeat(methods, max_cells)
    coords = PCA(n_components=2, random_state=config.SEED).fit_transform(stacked)

    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        mask = labels == method
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.65, label=method)
    ax.set_title(f"PCA embedding at {depth:.0%} depth")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"embedding_{depth_to_name(depth)}.png", dpi=180)
    plt.close(fig)


def evaluate():
    set_seed()
    ensure_dirs()
    data = load_processed()
    device = get_device()
    n_genes = data.high.shape[1]
    depthdiff_model = load_depthdiff(n_genes, device)
    diffusion = make_diffusion(device)
    baseline_models = load_baselines(n_genes, device)

    target = data.high[data.test_idx]
    labels = KMeans(n_clusters=8, random_state=config.SEED, n_init=20).fit_predict(target)
    rows = []
    for depth in config.DEPTHS:
        x_low = data.lows[depth][data.test_idx]
        predictions = {"raw_low_depth": x_low}
        if depthdiff_model is not None:
            predictions[config.DEPTHDIFF_METHOD_NAME] = predict_depthdiff(
                depthdiff_model,
                diffusion,
                x_low,
                depth,
                device,
            )
        for name, baseline_model in baseline_models.items():
            predictions[name] = predict_baseline(name, baseline_model, x_low, depth, device)

        for method, pred in predictions.items():
            cmetrics = prediction_cluster_metrics(labels, pred)
            marker_spearman, marker_overlap = marker_metrics(target, pred, labels)
            logfc_pearson, logfc_spearman, deg_direction = logfc_metrics(target, pred, labels)
            rows.append(
                {
                    "depth": depth,
                    "method": method,
                    "rmse": rmse(pred, target),
                    "pearson": pearson_flat(pred, target),
                    "cluster_ari": cmetrics["cluster_ari"],
                    "cluster_nmi": cmetrics["cluster_nmi"],
                    "marker_spearman": marker_spearman,
                    "marker_top50_overlap": marker_overlap,
                    "deg_top100_overlap": deg_overlap(target, pred, labels),
                    "logfc_pearson": logfc_pearson,
                    "logfc_spearman": logfc_spearman,
                    "deg_direction_consistency": deg_direction,
                    "module_score_pearson": module_score_pearson(target, pred, labels),
                }
            )
        plot_embedding(target, predictions, depth)

    metrics = pd.DataFrame(rows).sort_values(["depth", "method"])
    metrics = metrics.round(3)
    out_path = config.RESULTS_DIR / f"metrics_{config.BENCHMARK_NAME}.csv"
    metrics.to_csv(out_path, index=False)
    metrics.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    plot_metric_bars(rows)
    print(metrics)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    evaluate()
