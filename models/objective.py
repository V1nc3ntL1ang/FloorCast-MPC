from dataclasses import dataclass
import math

import config as cfg
from models.energy import segment_energy, standby_energy
from models.kinematics import travel_time


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
    Aggregate passenger-centric statistics / 汇总乘客相关指标。
    返回乘客总时间、等待、轿厢内时间、惩罚值以及服务数量。
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
    Compute weighted losses for waiting, riding, and energy usage /
    计算等待、乘坐与能耗的加权损失。

    Parameters / 参数
    -----------------
    wait_time: cumulative waiting time in seconds / 乘客等待总时间（秒）。
    in_cab_time: cumulative in-cab time in seconds / 乘客乘坐总时间（秒）。
    emptyload_energy: energy spent running empty (J) / 空载行驶能耗（焦耳）。
    running_energy: total traction + standby energy (J) / 总牵引加待机能耗（焦耳）。
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


def compute_theoretical_limit(
    requests,
) -> tuple[ObjectiveBreakdown, float, float, float, float]:
    """
    Estimate an M/D/c-inspired lower bound / 估计基于 M/D/c 模型的理论下界。
    假设所有请求同时出现，仅计算乘坐时间与能耗，并对等待使用闭式近似。
    """

    total_in_cab_time = 0.0
    total_running_energy = 0.0
    durations = []
    valid_requests = []

    for req in requests:
        origin = getattr(req, "origin", None)
        destination = getattr(req, "destination", None)
        load = getattr(req, "load", 0.0)

        if origin is None or destination is None or origin == destination:
            continue

        duration = travel_time(load, origin, destination)
        distance = abs(destination - origin) * cfg.BUILDING_FLOOR_HEIGHT
        direction = "up" if destination > origin else "down"

        total_in_cab_time += duration
        total_running_energy += segment_energy(load, distance, direction)
        total_running_energy += standby_energy(duration)
        durations.append(duration)
        valid_requests.append(req)

    nr = len(valid_requests)

    if nr == 0:
        wait_avg = 0.0
    else:
        tau_bar = sum(durations) / nr if durations else 0.0
        horizon = max(getattr(cfg, "SIM_TIME_HORIZON", 0), 1e-6)
        arrival_rate = nr / horizon
        tau_bar = max(tau_bar, 1e-6)
        ne = max(cfg.ELEVATOR_COUNT, 1)
        service_capacity = ne / tau_bar
        rho = (
            min(arrival_rate * tau_bar / ne, 0.999999) if service_capacity > 0 else 1.0
        )

        if arrival_rate >= service_capacity:
            wait_avg = tau_bar
        else:
            wait_avg = arrival_rate * (tau_bar**2) / (2.0 * ne * max(1.0 - rho, 1e-6))

    total_wait_time = nr * wait_avg

    if nr == 0 or wait_avg <= 0.0:
        total_wait_penalty = 0.0
    else:
        scale = max(WAIT_PENALTY_SCALE, 1e-6)
        exponent = max(WAIT_PENALTY_EXPONENT, 1.0)
        threshold = max(WAIT_PENALTY_THRESHOLD, 0.0)
        rate = 1.0 / max(wait_avg, 1e-9)
        tail_factor = math.exp(-rate * threshold)
        extra_term = tail_factor * math.gamma(exponent + 2.0) / (
            (scale**exponent) * (rate ** (exponent + 1.0))
        )
        if not math.isfinite(extra_term) or extra_term < 0.0:
            extra_term = 0.0
        wait_penalty_avg = wait_avg + extra_term
        total_wait_penalty = nr * wait_penalty_avg

    wait_cost = cfg.WEIGHT_TIME * total_wait_penalty
    ride_cost = cfg.WEIGHT_TIME * total_in_cab_time
    running_energy_cost = cfg.WEIGHT_ENERGY * total_running_energy
    emptyload_energy_cost = 0.0
    total_cost = ride_cost + running_energy_cost

    breakdown = ObjectiveBreakdown(
        total_cost=total_cost,
        wait_cost=wait_cost,
        ride_cost=ride_cost,
        running_energy_cost=running_energy_cost,
        emptyload_energy_cost=emptyload_energy_cost,
    )

    return (
        breakdown,
        total_in_cab_time,
        total_running_energy,
        total_wait_time,
        total_wait_penalty,
    )
