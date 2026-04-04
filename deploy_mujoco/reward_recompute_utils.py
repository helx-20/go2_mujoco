import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from typing import Dict
import yaml


def load_reward_cfg_from_yaml(path: str) -> Dict:
    """Load event/reward config from a yaml file.

    Returns an empty dict if file is missing or malformed.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return {}
        reward_cfg = cfg.get("event_and_reward", cfg)
        return reward_cfg if isinstance(reward_cfg, dict) else {}
    except Exception:
        return {}


def _reward_item_enabled(reward_cfg: Dict, key: str) -> bool:
    """A fail condition is disabled only when its scale exists and is exactly zero."""
    if key not in reward_cfg:
        return True
    try:
        return float(reward_cfg.get(key, 0.0)) != 0.0
    except Exception:
        return True


def _is_fail_flag_enabled(reward_cfg: Dict, flag: str, reward_key: str) -> bool:
    fail_cfg = reward_cfg.get("failure_flags", {}) if isinstance(reward_cfg, dict) else {}
    if isinstance(fail_cfg, dict) and flag in fail_cfg:
        return bool(fail_cfg.get(flag, True))
    return _reward_item_enabled(reward_cfg, reward_key)


def recompute_fail_flags_from_info(info: Dict, reward_cfg: Dict) -> Dict:
    info = info or {}
    reward_cfg = reward_cfg or {}

    fallen = bool(info.get("fallen", False)) and _is_fail_flag_enabled(reward_cfg, "fallen", "fall_reward")
    collided = bool(info.get("collided", False)) and _is_fail_flag_enabled(reward_cfg, "collided", "collision_reward")
    base_collision = bool(info.get("base_collision", False)) and _is_fail_flag_enabled(reward_cfg, "base_collision", "base_collision_reward")
    thigh_collision = bool(info.get("thigh_collision", False)) and _is_fail_flag_enabled(reward_cfg, "thigh_collision", "thigh_collision_reward")
    stuck = bool(info.get("stuck", False)) and _is_fail_flag_enabled(reward_cfg, "stuck", "stuck_reward")

    any_fail = bool(fallen or collided or base_collision or thigh_collision or stuck)
    return {
        "fallen": fallen,
        "collided": collided,
        "base_collision": base_collision,
        "thigh_collision": thigh_collision,
        "stuck": stuck,
        "any_fail": any_fail,
    }


def recompute_reward_from_info(info: Dict, reward_cfg: Dict) -> float:
    info = info or {}
    reward_cfg = reward_cfg or {}
    r = 0.0

    flags = recompute_fail_flags_from_info(info, reward_cfg)
    if flags["fallen"]:
        r += float(reward_cfg.get("fall_reward", 0.0))
    if flags["base_collision"]:
        r += float(reward_cfg.get("base_collision_reward", 0.0))
    if flags["thigh_collision"]:
        r += float(reward_cfg.get("thigh_collision_reward", 0.0))
    if flags["collided"]:
        r += float(reward_cfg.get("collision_reward", 0.0))
    if flags["stuck"]:
        r += float(reward_cfg.get("stuck_reward", 0.0))

    if "tilt" in info:
        r += float(reward_cfg.get("tilt_reward_scale", 0.0)) * float(info.get("tilt", 0.0))

    if "speed" in info:
        target_speed = float(reward_cfg.get("target_speed", 0.0))
        speed = float(info.get("speed", 0.0))
        speed_loss = max(0.0, target_speed - speed)
        r += float(reward_cfg.get("speed_reward_scale", 0.0)) * speed_loss

    return float(r)

