"""
Rolling-horizon (MPC-lite) scheduler without external solvers /
滚动时域（轻量 MPC）调度器，在不依赖外部求解器的情况下，为电梯分配请求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import config as cfg
from models.energy import segment_energy, standby_energy
from models.kinematics import travel_time
from models.temporal import hold_time


MPC_LOOKAHEAD_WINDOW = getattr(cfg, "MPC_LOOKAHEAD_WINDOW", 240.0)
MPC_MAX_BATCH = getattr(cfg, "MPC_MAX_BATCH", 12)


@dataclass
class _PlanState:
    floor: int
    time: float


def assign_requests_mpc(
    requests: List[object],
    elevators: List[object],
    *,
    lookahead_window: float | None = None,
    max_batch: int | None = None,
) -> None:
    """
    Assign requests using a rolling-horizon heuristic /
    采用滚动时域启发式将请求分配给电梯。

    Parameters / 参数
    -----------------
    lookahead_window:
        Maximum seconds beyond earliest unassigned arrival / 视窗长度（秒）。
    max_batch:
        Maximum candidate requests per iteration / 每轮评估的候选请求数。
    """
    if not elevators:
        return

    horizon = MPC_LOOKAHEAD_WINDOW if lookahead_window is None else lookahead_window
    batch_limit = MPC_MAX_BATCH if max_batch is None else max_batch
    if batch_limit <= 0:
        batch_limit = max(len(elevators) * 3, 1)

    for elev in elevators:
        elev.queue = []
        elev.served_requests = []

    if not requests:
        return

    unassigned = list(sorted(requests, key=lambda r: r.arrival_time))
    plans = {elev.id: _PlanState(floor=elev.floor, time=0.0) for elev in elevators}
    elevator_lookup = {elev.id: elev for elev in elevators}
    eps = 1e-9

    while unassigned:
        earliest_arrival = unassigned[0].arrival_time
        window_limit = earliest_arrival + horizon

        candidate_indices: List[int] = []
        for idx, req in enumerate(unassigned):
            if req.arrival_time <= window_limit:
                candidate_indices.append(idx)
            elif len(candidate_indices) < batch_limit:
                candidate_indices.append(idx)
            if len(candidate_indices) >= batch_limit:
                break

        if not candidate_indices:
            candidate_indices = list(range(min(batch_limit, len(unassigned))))

        best_choice: Tuple[int, int, float, float] | None = None
        best_cost: float | None = None

        for idx in candidate_indices:
            req = unassigned[idx]
            for elev in elevators:
                estimate = _estimate_incremental_cost(plans[elev.id], req)
                if estimate is None:
                    continue
                cost, finish_time, passenger_time = estimate

                if (
                    best_cost is None
                    or cost < best_cost - eps
                    or (
                        abs(cost - best_cost) <= eps
                        and finish_time < best_choice[3] - eps
                    )
                    or (
                        abs(cost - best_cost) <= eps
                        and abs(finish_time - best_choice[3]) <= eps
                        and passenger_time < best_choice[2]
                    )
                ):
                    best_cost = cost
                    best_choice = (idx, elev.id, passenger_time, finish_time)

        if best_choice is None:
            # Fallback to least-busy elevator / 回退到最空闲电梯以避免停滞。
            idx = candidate_indices[0]
            req = unassigned.pop(idx)
            target_id = min(plans, key=lambda eid: plans[eid].time)
            estimate = _estimate_incremental_cost(plans[target_id], req)
            finish_time = plans[target_id].time
            if estimate is not None:
                finish_time = estimate[1]
            _apply_assignment(elevator_lookup[target_id], req)
            plans[target_id].time = finish_time
            plans[target_id].floor = req.destination
            continue

        idx, elevator_id, _, finish_time = best_choice
        req = unassigned.pop(idx)
        _apply_assignment(elevator_lookup[elevator_id], req)
        plans[elevator_id].time = finish_time
        plans[elevator_id].floor = req.destination


def _estimate_incremental_cost(
    plan: _PlanState, request: object
) -> Tuple[float, float, float] | None:
    """Return (total_cost, finish_time, passenger_time) / 返回追加请求后的成本和完成时间。"""
    current_floor = plan.floor
    available_time = plan.time
    origin = request.origin
    destination = request.destination

    travel_to_origin = travel_time(0.0, current_floor, origin)
    arrival_at_origin = available_time + travel_to_origin
    start_service = max(arrival_at_origin, request.arrival_time)
    dwell = hold_time(request.load, 0.0)
    depart_time = start_service + dwell
    travel_to_dest = travel_time(request.load, origin, destination)
    finish_time = depart_time + travel_to_dest

    passenger_time = finish_time - request.arrival_time

    energy = 0.0
    if current_floor != origin:
        distance = abs(current_floor - origin) * cfg.BUILDING_FLOOR_HEIGHT
        energy += segment_energy(0.0, distance, _direction(current_floor, origin))
        energy += standby_energy(travel_to_origin)

    energy += standby_energy(dwell)

    if origin != destination:
        distance = abs(destination - origin) * cfg.BUILDING_FLOOR_HEIGHT
        energy += segment_energy(
            request.load, distance, _direction(origin, destination)
        )
        energy += standby_energy(travel_to_dest)

    total_cost = cfg.WEIGHT_TIME * passenger_time + cfg.WEIGHT_ENERGY * energy
    # Small bias for smoother schedules when tied / 在成本相同时加入轻微偏置以保持平滑。
    total_cost += 1e-6 * finish_time

    return total_cost, finish_time, passenger_time


def _apply_assignment(elevator, request: object) -> None:
    elevator.queue.append(request)
    elevator.served_requests.append(request)


def _direction(start: int, end: int) -> str:
    if end > start:
        return "up"
    if end < start:
        return "down"
    return "up"
