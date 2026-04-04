import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deploy_mujoco.offline_data_utils import collect_pkl_files, load_chains_from_pkl_file

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for PCA/clustering. Please install it first: pip install scikit-learn"
    ) from e


FAIL_KEYS = ("fallen", "base_collision", "thigh_collision", "stuck")


@dataclass
class TransitionRow:
    source_file: str
    chain_idx: int
    step_idx: int
    reward: float
    done: int
    failure: int


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_1d_array(v: Any) -> np.ndarray:
    arr = np.asarray(v if v is not None else [], dtype=np.float32)
    return arr.reshape(-1)


def _is_failure(info: Dict[str, Any]) -> int:
    if not isinstance(info, dict):
        return 0
    return int(any(bool(info.get(k, False)) for k in FAIL_KEYS))


def _build_feature_vector(tr: Dict[str, Any], feat_cfg: Dict[str, Any]) -> np.ndarray:
    parts: List[np.ndarray] = []

    if bool(feat_cfg.get("use_obs", True)):
        parts.append(_to_1d_array(tr.get("obs", [])))
    if bool(feat_cfg.get("use_action", True)):
        parts.append(_to_1d_array(tr.get("action", [])))
    if bool(feat_cfg.get("use_next_obs", False)):
        parts.append(_to_1d_array(tr.get("next_obs", [])))

    if bool(feat_cfg.get("use_reward", True)):
        parts.append(np.asarray([float(tr.get("reward", 0.0))], dtype=np.float32))
    if bool(feat_cfg.get("use_done", True)):
        parts.append(np.asarray([float(bool(tr.get("done", False)))], dtype=np.float32))

    info = tr.get("info", {}) if isinstance(tr.get("info", {}), dict) else {}
    if bool(feat_cfg.get("use_failure_flags", True)):
        parts.append(np.asarray([float(_is_failure(info))], dtype=np.float32))

    if len(parts) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _load_transitions_from_pkls(
    pkl_paths: Sequence[str],
    feat_cfg: Dict[str, Any],
    max_files: int,
    max_chains_per_file: int,
    max_transitions_per_chain: int,
) -> Tuple[np.ndarray, List[TransitionRow], Dict[str, int]]:
    files = collect_pkl_files(pkl_paths)
    if max_files > 0:
        files = files[:max_files]

    vectors: List[np.ndarray] = []
    rows: List[TransitionRow] = []

    skipped_invalid = 0
    skipped_dim_mismatch = 0
    expected_dim: Optional[int] = None

    for fp in files:
        try:
            chains = load_chains_from_pkl_file(fp, consecutive_fail_keep_k=0)
        except Exception:
            skipped_invalid += 1
            continue

        if max_chains_per_file > 0:
            chains = chains[:max_chains_per_file]

        for cidx, chain in enumerate(chains):
            chain_iter = chain if max_transitions_per_chain <= 0 else chain[:max_transitions_per_chain]
            for sidx, tr in enumerate(chain_iter):
                if not isinstance(tr, dict):
                    skipped_invalid += 1
                    continue

                vec = _build_feature_vector(tr, feat_cfg)
                if vec.size == 0:
                    skipped_invalid += 1
                    continue

                if expected_dim is None:
                    expected_dim = int(vec.shape[0])
                if int(vec.shape[0]) != expected_dim:
                    skipped_dim_mismatch += 1
                    continue

                info = tr.get("info", {}) if isinstance(tr.get("info", {}), dict) else {}
                rows.append(
                    TransitionRow(
                        source_file=fp,
                        chain_idx=int(cidx),
                        step_idx=int(sidx),
                        reward=float(tr.get("reward", 0.0)),
                        done=int(bool(tr.get("done", False))),
                        failure=_is_failure(info),
                    )
                )
                vectors.append(vec)

    if len(vectors) == 0:
        return np.zeros((0, 0), dtype=np.float32), rows, {
            "files_total": len(files),
            "samples_kept": 0,
            "skipped_invalid": skipped_invalid,
            "skipped_dim_mismatch": skipped_dim_mismatch,
        }

    x = np.stack(vectors, axis=0).astype(np.float32)
    return x, rows, {
        "files_total": len(files),
        "samples_kept": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "skipped_invalid": skipped_invalid,
        "skipped_dim_mismatch": skipped_dim_mismatch,
    }


def _generate_demo_transitions(n_per_cluster: int = 120) -> Tuple[np.ndarray, List[TransitionRow], Dict[str, int]]:
    rng = np.random.RandomState(0)
    centers = np.asarray([
        [2.0, 2.0, 0.5, 0.1],
        [-2.0, 1.5, -0.3, 0.2],
        [0.0, -2.0, 0.8, -0.4],
    ], dtype=np.float32)

    x_list: List[np.ndarray] = []
    rows: List[TransitionRow] = []
    idx = 0
    for cidx, c in enumerate(centers):
        for sidx in range(n_per_cluster):
            point = c + 0.35 * rng.randn(c.shape[0]).astype(np.float32)
            x_list.append(point)
            rows.append(
                TransitionRow(
                    source_file="demo",
                    chain_idx=int(cidx),
                    step_idx=int(sidx),
                    reward=float(point[0] * 0.1),
                    done=int(sidx == n_per_cluster - 1),
                    failure=int(cidx == 2 and (sidx % 17 == 0)),
                )
            )
            idx += 1

    x = np.stack(x_list, axis=0)
    return x, rows, {
        "files_total": 1,
        "samples_kept": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "skipped_invalid": 0,
        "skipped_dim_mismatch": 0,
    }


def _fit_pca(x: np.ndarray, pca_cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    if x.shape[0] == 0:
        return x, {}, x

    standardize = bool(pca_cfg.get("standardize", True))
    n_components = pca_cfg.get("n_components", 2)
    whiten = bool(pca_cfg.get("whiten", False))

    x_proc = x
    if standardize:
        scaler = StandardScaler()
        x_proc = scaler.fit_transform(x_proc)

    pca = PCA(n_components=n_components, whiten=whiten, random_state=int(pca_cfg.get("random_state", 0)))
    x_pca = pca.fit_transform(x_proc)

    info = {
        "n_components": int(pca.n_components_) if isinstance(pca.n_components_, (int, np.integer)) else pca.n_components_,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "standardize": standardize,
        "whiten": whiten,
    }
    return x_pca.astype(np.float32), info, x_proc.astype(np.float32)


def _safe_silhouette(x: np.ndarray, labels: np.ndarray) -> Optional[float]:
    unique = np.unique(labels)
    # silhouette requires at least 2 clusters, and excludes all-noise dbscan case
    if x.shape[0] < 2 or unique.shape[0] < 2:
        return None
    try:
        return float(silhouette_score(x, labels))
    except Exception:
        return None


def _run_kmeans(x: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    model = KMeans(
        n_clusters=int(cfg.get("n_clusters", 4)),
        random_state=int(cfg.get("random_state", 0)),
        n_init=int(cfg.get("n_init", 10)),
        max_iter=int(cfg.get("max_iter", 300)),
    )
    labels = model.fit_predict(x)
    counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    return labels.astype(np.int32), {
        "n_clusters": int(len(counts)),
        "counts": counts,
        "inertia": float(model.inertia_),
        "silhouette": _safe_silhouette(x, labels),
    }


def _run_dbscan(x: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    model = DBSCAN(
        eps=float(cfg.get("eps", 0.8)),
        min_samples=int(cfg.get("min_samples", 10)),
        metric=str(cfg.get("metric", "euclidean")),
    )
    labels = model.fit_predict(x)
    counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    n_noise = int(counts.get(-1, 0))
    n_clusters = int(len([k for k in counts.keys() if k != -1]))
    return labels.astype(np.int32), {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "counts": counts,
        "silhouette": _safe_silhouette(x, labels),
    }


def _write_results(
    out_dir: str,
    rows: List[TransitionRow],
    x_pca: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    summary: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save raw arrays for reproducibility.
    npz_payload: Dict[str, Any] = {"pca": x_pca.astype(np.float32)}
    for name, labels in labels_dict.items():
        npz_payload[f"labels_{name}"] = labels.astype(np.int32)
    np.savez_compressed(os.path.join(out_dir, "cluster_results.npz"), **npz_payload)

    # Save human-readable csv.
    csv_path = os.path.join(out_dir, "transition_clusters.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = [
            "source_file",
            "chain_idx",
            "step_idx",
            "reward",
            "done",
            "failure",
        ]
        headers += [f"pca_{i}" for i in range(x_pca.shape[1])]
        for name in labels_dict.keys():
            headers.append(f"{name}_label")
        f.write(",".join(headers) + "\n")

        for i, row in enumerate(rows):
            vals = [
                row.source_file.replace(",", " "),
                str(row.chain_idx),
                str(row.step_idx),
                f"{row.reward:.8f}",
                str(row.done),
                str(row.failure),
            ]
            vals += [f"{float(v):.8f}" for v in x_pca[i]]
            for name in labels_dict.keys():
                vals.append(str(int(labels_dict[name][i])))
            f.write(",".join(vals) + "\n")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def _plot_results(out_dir: str, x_pca: np.ndarray, labels_dict: Dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[transition_pca_cluster] matplotlib not installed, skipping plot.")
        return
    
    for method, labels in labels_dict.items():
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Cluster Label')
        plt.title(f'PCA Clustering: {method.upper()}')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plot_path = os.path.join(out_dir, f"plot_{method}.png")
        plt.savefig(plot_path)
        plt.close()

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read transitions and run PCA + clustering (KMeans/DBSCAN).")
    parser.add_argument("--config", type=str, default="", help="Path to yaml config.")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data instead of reading PKL files.")
    parser.add_argument("--method", type=str, default="", choices=["kmeans", "dbscan", "both"], help="Override clustering method.")
    parser.add_argument("--out", type=str, default="", help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    default_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transition_analysis_config.yaml")
    cfg_path = args.config if args.config else default_cfg_path
    cfg = _load_yaml(cfg_path)

    if args.method:
        cfg["method"] = args.method
    if args.out:
        cfg["output_dir"] = args.out

    method = str(cfg.get("method", "both")).lower()
    if method not in ("kmeans", "dbscan", "both"):
        raise ValueError("method must be one of: kmeans, dbscan, both")

    if args.demo:
        x, rows, data_stats = _generate_demo_transitions(n_per_cluster=int(cfg.get("demo_n_per_cluster", 120)))
    else:
        pkl_paths = []
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        for f in cfg.get("pkl_paths", []):
            pkl_paths.append(os.path.join(curr_dir, f))
        x, rows, data_stats = _load_transitions_from_pkls(
            pkl_paths=pkl_paths,
            feat_cfg=cfg.get("features", {}),
            max_files=int(cfg.get("max_files", 0)),
            max_chains_per_file=int(cfg.get("max_chains_per_file", 0)),
            max_transitions_per_chain=int(cfg.get("max_transitions_per_chain", 0)),
        )

    if x.shape[0] == 0:
        raise RuntimeError("No valid transitions loaded. Check pkl_paths and feature settings.")

    x_pca, pca_info, x_proc = _fit_pca(x, cfg.get("pca", {}))

    cluster_input = x_pca if bool(cfg.get("cluster_on_pca", True)) else x_proc

    labels_dict: Dict[str, np.ndarray] = {}
    cluster_summary: Dict[str, Any] = {}

    if method in ("kmeans", "both"):
        labels_k, info_k = _run_kmeans(cluster_input, cfg.get("kmeans", {}))
        labels_dict["kmeans"] = labels_k
        cluster_summary["kmeans"] = info_k

    if method in ("dbscan", "both"):
        labels_d, info_d = _run_dbscan(cluster_input, cfg.get("dbscan", {}))
        labels_dict["dbscan"] = labels_d
        cluster_summary["dbscan"] = info_d

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(curr_dir, "logs", str(cfg.get("output_dir", "transition_analysis_out"))))

    summary = {
        "config_path": cfg_path,
        "method": method,
        "cluster_on_pca": bool(cfg.get("cluster_on_pca", True)),
        "data": data_stats,
        "pca": pca_info,
        "clusters": cluster_summary,
    }

    _write_results(out_dir, rows, x_pca, labels_dict, summary)
    _plot_results(out_dir, x_pca, labels_dict)

    print("[transition_pca_cluster] done")
    print(f"  samples: {x.shape[0]}, feature_dim: {x.shape[1]}, pca_dim: {x_pca.shape[1]}")
    print(f"  output: {out_dir}")
    for k, v in cluster_summary.items():
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

