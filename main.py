from __future__ import annotations

import os
from copy import deepcopy
from typing import Callable, Dict, List, Sequence, Tuple

import config as cfg
from models.baseline_scheduler import assign_requests_greedy, simulate_dispatch
from models.objective import (
    compute_objective,
    compute_theoretical_limit,
    summarize_passenger_metrics,
)
from models.request import generate_requests_day
from models.utils import (
    DEFAULT_PLOT_DIR,
    log_results,
    plot_elevator_movements,
    plot_elevator_movements_time,
    plot_wait_distribution,
)
from models.variables import ElevatorState
from mpc_scheduler import assign_requests_mpc


def _extract_wait_times(served_requests) -> List[float]:
    """Collect wait durations per request / 提取每个请求的等待时间。"""
    waits: List[float] = []
    for req in served_requests:
        arrival = getattr(req, "arrival_time", None)
        origin_arrival = getattr(req, "origin_arrival_time", None)
        pickup = getattr(req, "pickup_time", None)
        if arrival is None:
            continue
        boarding_time = origin_arrival if origin_arrival is not None else pickup
        if boarding_time is None:
            continue
        waits.append(max(boarding_time - arrival, 0.0))
    return waits


def _run_strategy(
    name: str,
    assign_fn: Callable[[List[object], List[ElevatorState]], None],
    base_requests: List[object],
) -> Dict[str, object]:
    """
    Execute a scheduling strategy and gather metrics /
    执行给定调度策略并收集指标。
    """
    requests_copy = deepcopy(base_requests)
    elevators = [ElevatorState(id=k + 1, floor=1) for k in range(cfg.ELEVATOR_COUNT)]

    assign_fn(requests_copy, elevators)

    (
        system_time,
        total_energy,
        served_requests,
        emptyload_energy,
    ) = simulate_dispatch(elevators)

    passenger_metrics = summarize_passenger_metrics(served_requests)
    running_energy = total_energy

    objective_breakdown = compute_objective(
        passenger_metrics.total_wait_time,
        passenger_metrics.total_in_cab_time,
        emptyload_energy,
        running_energy,
        wait_penalty_value=passenger_metrics.wait_penalty_total,
    )
    (
        theoretical_breakdown,
        theoretical_in_cab_time,
        theoretical_running_energy,
        theoretical_wait_time,
        theoretical_wait_penalty,
    ) = compute_theoretical_limit(served_requests)

    wait_times = _extract_wait_times(served_requests)

    if cfg.SIM_ENABLE_LOG:
        log_results(
            elevators,
            system_time,
            running_energy,
            objective_breakdown,
            passenger_metrics.total_passenger_time,
            passenger_metrics.total_wait_time,
            passenger_metrics.total_in_cab_time,
            passenger_metrics.wait_penalty_total,
            emptyload_energy,
            theoretical_breakdown,
            theoretical_in_cab_time,
            theoretical_running_energy,
            theoretical_wait_time,
            theoretical_wait_penalty,
            strategy_label=name,
        )

    return {
        "name": name,
        "elevators": elevators,
        "system_time": system_time,
        "running_energy": running_energy,
        "emptyload_energy": emptyload_energy,
        "served_count": passenger_metrics.served_count,
        "passenger_total_time": passenger_metrics.total_passenger_time,
        "passenger_wait_time": passenger_metrics.total_wait_time,
        "passenger_in_cab_time": passenger_metrics.total_in_cab_time,
        "wait_penalty": passenger_metrics.wait_penalty_total,
        "objective": objective_breakdown,
        "theoretical": {
            "breakdown": theoretical_breakdown,
            "in_cab_time": theoretical_in_cab_time,
            "running_energy": theoretical_running_energy,
            "wait_time": theoretical_wait_time,
            "wait_penalty": theoretical_wait_penalty,
        },
        "wait_times": wait_times,
    }


def main() -> None:
    # 1. 生成请求 / generate requests
    requests = generate_requests_day(cfg.SIM_TOTAL_REQUESTS)

    strategies: Sequence[
        Tuple[str, Callable[[List[object], List[ElevatorState]], None]]
    ] = (
        ("baseline", assign_requests_greedy),
        ("mpc", assign_requests_mpc),
    )

    results: List[Dict[str, object]] = [
        _run_strategy(name, assign_fn, requests) for name, assign_fn in strategies
    ]

    if cfg.SIM_ENABLE_PLOTS:
        wait_series: List[Tuple[str, Sequence[float]]] = []
        for result in results:
            strat_name = result["name"]
            elevator_list = result["elevators"]
            wait_series.append((strat_name, result["wait_times"]))
            plot_elevator_movements(
                elevator_list,
                filename=os.path.join(
                    DEFAULT_PLOT_DIR, f"elevator_schedule_{strat_name}.png"
                ),
            )
            plot_elevator_movements_time(
                elevator_list,
                filename=os.path.join(
                    DEFAULT_PLOT_DIR, f"elevator_schedule_time_{strat_name}.png"
                ),
            )
        plot_wait_distribution(wait_series)

    for result in results:
        obj = result["objective"]
        theo = result["theoretical"]
        print(
            "\n=== Strategy: {name} ===\n"
            "Passenger Time: {total:.2f}s "
            "(waiting {wait:.2f}s | in-cab {incab:.2f}s)\n"
            "Wait Penalty (super-linear): {penalty:.2f}\n"
            "Energy (total): {energy:.2f}J\n"
            "Energy (empty-load): {empty:.2f}J\n"
            "System Active Time: {sys:.2f}s\n"
            "Served Requests: {served}\n"
            "Objective Cost: {cost:.2f}\n"
            "  - Wait Cost: {wait_cost:.2f}\n"
            "  - Ride Cost: {ride_cost:.2f}\n"
            "  - Running Energy Cost: {run_cost:.2f}\n"
            "  - Empty-load Energy Surcharge: {empty_cost:.2f}\n"
            "Theoretical Best-case: wait {theo_wait:.2f}s "
            "(penalty {theo_penalty:.2f}) | "
            "time {theo_time:.2f}s | "
            "energy {theo_energy:.2f}J | "
            "cost {theo_cost:.2f}".format(
                name=result["name"],
                total=result["passenger_total_time"],
                wait=result["passenger_wait_time"],
                incab=result["passenger_in_cab_time"],
                penalty=result["wait_penalty"],
                energy=result["running_energy"],
                empty=result["emptyload_energy"],
                sys=result["system_time"],
                served=result["served_count"],
                cost=obj.total_cost,
                wait_cost=obj.wait_cost,
                ride_cost=obj.ride_cost,
                run_cost=obj.running_energy_cost,
                empty_cost=obj.emptyload_energy_cost,
                theo_wait=theo["wait_time"],
                theo_penalty=theo["wait_penalty"],
                theo_time=theo["in_cab_time"],
                theo_energy=theo["running_energy"],
                theo_cost=theo["breakdown"].total_cost,
            )
        )


if __name__ == "__main__":
    main()
