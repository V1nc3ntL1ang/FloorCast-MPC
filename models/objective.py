import config as cfg


def compute_objective(total_time, total_energy):
    """Compute weighted total cost combining time and energy."""
    return cfg.WEIGHT_TIME * total_time + cfg.WEIGHT_ENERGY * total_energy
