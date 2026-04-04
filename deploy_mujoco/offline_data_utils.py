import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import glob
import pickle
from typing import Callable, Dict, Iterable, List, Optional, Sequence

DEFAULT_FAIL_KEYS = (
    "fallen",
    "collided",
    "base_collision",
    "thigh_collision",
    "stuck"
)


def collect_pkl_files(paths: Sequence[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if not p:
            continue
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.pkl")))
        elif os.path.isfile(p) and p.lower().endswith(".pkl"):
            files.append(p)
    return sorted(list(set(files)))


def load_chains_from_pkl_file(
    pkl_path: str,
    consecutive_fail_keep_k: int = 0,
    fail_keys: Iterable[str] = DEFAULT_FAIL_KEYS,
) -> List[List[Dict]]:
    obj = _load_pickle_file(pkl_path)
    chains = _extract_chains_from_obj(obj)

    out: List[List[Dict]] = []
    for chain in chains:
        filtered = _cap_consecutive_failures(chain, max_keep_fail=int(consecutive_fail_keep_k), fail_keys=fail_keys)
        if len(filtered) > 0:
            out.append(filtered)
    return out


def _load_pickle_file(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _extract_chains_from_obj(obj) -> List[List[Dict]]:
    chains: List[List[Dict]] = []

    if isinstance(obj, dict):
        if "chain" in obj and isinstance(obj["chain"], list):
            chains.append(obj["chain"])
        elif "chains" in obj and isinstance(obj["chains"], list):
            for c in obj["chains"]:
                if isinstance(c, list):
                    chains.append(c)

    elif isinstance(obj, list):
        if len(obj) == 0:
            return chains
        if isinstance(obj[0], dict) and "chain" in obj[0]:
            for ep in obj:
                c = ep.get("chain", [])
                if isinstance(c, list):
                    chains.append(c)
        elif isinstance(obj[0], dict) and "obs" in obj[0] and "action" in obj[0]:
            chains.append(obj)

    return chains


def _is_failure_transition(tr: Dict, fail_keys: Iterable[str] = DEFAULT_FAIL_KEYS) -> bool:
    if not isinstance(tr, dict):
        return False

    for k in fail_keys:
        if k in tr:
            return bool(tr.get(k, False))

    info = tr.get("info", {})
    if isinstance(info, dict):
        return any(bool(info.get(k, False)) for k in fail_keys)

    return False


def _is_stuck_transition(tr: Dict) -> bool:
    if not isinstance(tr, dict):
        return False
    if "stuck" in tr:
        return bool(tr.get("stuck", False))
    info = tr.get("info", {})
    if isinstance(info, dict):
        return bool(info.get("stuck", False))
    return False


def filter_chain_for_replay(
    chain: List[Dict],
    consecutive_fail_keep_k: int = 0,
    fail_keys: Iterable[str] = DEFAULT_FAIL_KEYS,
    extra_keep_fn: Optional[Callable[[Dict, int], bool]] = None,
) -> List[Dict]:
    """Filter one transition chain for replay/add and fix done after filtering.

    Rules:
    - Only trim runs where transition is both failure and stuck.
    - Keep at most K frames in each consecutive fail+stuck run.
    - Optionally apply extra_keep_fn(tr, idx) for custom keep logic.
    - If filtering creates a discontinuity, force done=True on the previous kept frame.
    """
    if not isinstance(chain, list) or len(chain) == 0:
        return []

    k = int(max(0, consecutive_fail_keep_k))
    keep_indices: List[int] = []
    stuck_fail_run = 0

    for idx, tr in enumerate(chain):
        is_fail = _is_failure_transition(tr, fail_keys=fail_keys)
        is_stuck = _is_stuck_transition(tr)
        is_stuck_fail = is_fail and is_stuck

        if is_stuck_fail:
            stuck_fail_run += 1
        else:
            stuck_fail_run = 0

        keep = True
        if k > 0 and is_stuck_fail and stuck_fail_run > k:
            keep = False
        if keep and extra_keep_fn is not None:
            keep = bool(extra_keep_fn(tr, idx))

        if keep:
            keep_indices.append(idx)

    selected: List[Dict] = []
    for j, idx in enumerate(keep_indices):
        tr = dict(chain[idx])
        next_is_contiguous = (j + 1 < len(keep_indices)) and (keep_indices[j + 1] == idx + 1)
        tr["done"] = bool(tr.get("done", False) or (not next_is_contiguous))
        selected.append(tr)
    return selected


def _cap_consecutive_failures(chain: List[Dict], max_keep_fail: int, fail_keys: Iterable[str] = DEFAULT_FAIL_KEYS) -> List[Dict]:
    """Keep only the first K frames in each consecutive *stuck-failure* run.

    - Trimming is triggered only when a frame is both failure and stuck.
    - Non-stuck failures are always kept.
    - If max_keep_fail <= 0, no trimming is applied.
    """
    return filter_chain_for_replay(
        chain,
        consecutive_fail_keep_k=int(max_keep_fail),
        fail_keys=fail_keys,
    )


# def load_chains_from_pkl_paths(
#     pkl_paths: Sequence[str],
#     consecutive_fail_keep_k: int = 0,
#     fail_keys: Iterable[str] = DEFAULT_FAIL_KEYS,
# ) -> List[List[Dict]]:
#     chains_out: List[List[Dict]] = []
#     for fp in collect_pkl_files(pkl_paths):
#         try:
#             chains = load_chains_from_pkl_file(
#                 fp,
#                 consecutive_fail_keep_k=consecutive_fail_keep_k,
#                 fail_keys=fail_keys,
#             )
#             chains_out.extend(chains)
#         except Exception:
#             continue
#     return chains_out


