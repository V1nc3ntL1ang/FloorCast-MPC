from dataclasses import dataclass

import config as cfg


WAIT_PENALTY_SCALE = getattr(cfg, "WAIT_PENALTY_SCALE", 60.0)
WAIT_PENALTY_EXPONENT = getattr(cfg, "WAIT_PENALTY_EXPONENT", 1.5)
WAIT_PENALTY_THRESHOLD = getattr(cfg, "WAIT_PENALTY_THRESHOLD", 0.0)
EMPTYLOAD_PENALTY_MULTIPLIER = getattr(cfg, "EMPTYLOAD_PENALTY_MULTIPLIER", 2.0)


@dataclass
class PassengerMetrics:
    total_passenger_time: float
    total_wait_time: float
    total_in_cab_time: float
    wait_penalty_total: float
    served_count: int


@dataclass
class ObjectiveBreakdown:
    total_cost: float
    wait_cost: float
    ride_cost: float
    running_energy_cost: float
    emptyload_energy_cost: float


def wait_penalty(wait_time: float) -> float:
    """
    Piecewise (truncated) super-linear penalty for passenger waiting time.
    仅当等待时间超过阈值时施加额外非线性惩罚。

    参数:
        wait_time : 等待时间 (s)
    超参:
        WAIT_PENALTY_SCALE     - 时间归一化尺度 (s)
        WAIT_PENALTY_EXPONENT  - 非线性指数 (>1)
        WAIT_PENALTY_THRESHOLD - 开始施加非线性惩罚的阈值 (s)
    """
    if wait_time <= 0.0:
        return 0.0

    scale = max(WAIT_PENALTY_SCALE, 1e-6)
    exponent = max(WAIT_PENALTY_EXPONENT, 1.0)
    threshold = max(WAIT_PENALTY_THRESHOLD, 0.0)

    if wait_time <= threshold:
        # 阈值以下 → 线性惩罚（可直接等同于时间本身）
        return wait_time
    else:
        # 超过阈值部分施加非线性放大
        excess = wait_time - threshold
        normalized = excess / scale
        nonlinear_penalty = excess * (1.0 + normalized**exponent)
        return threshold + nonlinear_penalty


def summarize_passenger_metrics(served_requests) -> PassengerMetrics:
    """
    Aggregate passenger-centric statistics from the served request list.
    Returns totals for passenger time, waits, in-cab durations, the amplified
    wait penalty, and the served passenger count.
    """
    total_passenger_time = 0.0
    total_wait_time = 0.0
    total_in_cab_time = 0.0
    total_wait_penalty = 0.0
    served_count = 0

    for req in served_requests:
        arr = getattr(req, "arrival_time", None)
        origin_arrival = getattr(req, "origin_arrival_time", None)
        dest_arrival = getattr(req, "destination_arrival_time", None)

        if arr is None or dest_arrival is None:
            continue

        served_count += 1
        trip_total = max(dest_arrival - arr, 0.0)
        total_passenger_time += trip_total

        if origin_arrival is not None:
            wait = max(origin_arrival - arr, 0.0)
            total_wait_time += wait
            total_wait_penalty += wait_penalty(wait)
            total_in_cab_time += max(dest_arrival - origin_arrival, 0.0)
        else:
            total_in_cab_time += trip_total

    return PassengerMetrics(
        total_passenger_time=total_passenger_time,
        total_wait_time=total_wait_time,
        total_in_cab_time=total_in_cab_time,
        wait_penalty_total=total_wait_penalty,
        served_count=served_count,
    )


def compute_objective(
    wait_time: float,
    in_cab_time: float,
    emptyload_energy: float,
    running_energy: float,
    *,
    wait_penalty_value: float | None = None,
) -> ObjectiveBreakdown:
    """
    Compute weighted losses for waiting, riding, and energy usage, then aggregate.

    Parameters
    ----------
    wait_time:
        Cumulative passenger waiting time (seconds).
    in_cab_time:
        Cumulative passenger in-cab time (seconds).
    emptyload_energy:
        Energy spent while traveling empty to pick up requests (J).
    running_energy:
        Total traction + standby energy over the horizon (J).
    """
    wait_penalty_value = (
        wait_penalty(wait_time) if wait_penalty_value is None else wait_penalty_value
    )
    wait_cost = cfg.WEIGHT_TIME * wait_penalty_value
    ride_cost = cfg.WEIGHT_TIME * in_cab_time
    running_energy_cost = cfg.WEIGHT_ENERGY * running_energy
    extra_multiplier = max(EMPTYLOAD_PENALTY_MULTIPLIER - 1.0, 0.0)
    emptyload_energy_cost = cfg.WEIGHT_ENERGY * emptyload_energy * extra_multiplier

    total_cost = wait_cost + ride_cost + running_energy_cost + emptyload_energy_cost

    return ObjectiveBreakdown(
        total_cost=total_cost,
        wait_cost=wait_cost,
        ride_cost=ride_cost,
        running_energy_cost=running_energy_cost,
        emptyload_energy_cost=emptyload_energy_cost,
    )
