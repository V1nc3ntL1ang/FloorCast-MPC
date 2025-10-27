import config as cfg


def hold_time(boarding_weight, alighting_weight):
    """Compute holding (door open) time as a function of boarding and alighting weight."""
    total_weight = boarding_weight + alighting_weight
    if total_weight <= cfg.HOLD_CONGESTION_THRESHOLD:
        return cfg.HOLD_BASE_TIME + cfg.HOLD_EFF_NORMAL * total_weight
    else:
        normal_part = cfg.HOLD_EFF_NORMAL * cfg.HOLD_CONGESTION_THRESHOLD
        congested_part = cfg.HOLD_EFF_CONGESTED * (
            total_weight - cfg.HOLD_CONGESTION_THRESHOLD
        )
        return cfg.HOLD_BASE_TIME + normal_part + congested_part
