import argparse
from models.baseline_scheduler import assign_requests_greedy, simulate_dispatch
from models.objective import compute_objective, summarize_passenger_metrics
from models.request import generate_requests_day
from models.utils import (
    log_results,
    plot_elevator_movements,
    plot_elevator_movements_time,
    print_elevator_queues,
)
from models.variables import ElevatorState
import config as cfg
from mpc_scheduler import assign_requests_mpc

try:
    from milp_scheduler import assign_requests_milp
except Exception:
    assign_requests_milp = None


def main(strategy: str | None = None):
    valid_strategies = {"baseline", "milp", "mpc"}

    if strategy is None:
        parser = argparse.ArgumentParser(description="Load-Aware Elevator Simulation")
        parser.add_argument(
            "--strategy",
            choices=sorted(valid_strategies),
            required=True,
            help="Scheduling strategy to run",
        )
        args = parser.parse_args()
        strategy = args.strategy
    else:
        if strategy not in valid_strategies:
            raise ValueError(f"Unsupported strategy '{strategy}'.")

    # 1. 生成请求
    requests = generate_requests_day(cfg.SIM_TOTAL_REQUESTS)

    # 2. 初始化电梯
    elevators = [ElevatorState(id=k + 1, floor=1) for k in range(cfg.ELEVATOR_COUNT)]

    # 3. 调度（可选 baseline 或 MILP）
    used_strategy = strategy
    if strategy == "mpc":
        assign_requests_mpc(requests, elevators)
    elif strategy == "milp":

        if assign_requests_milp is None:
            raise RuntimeError(
                "MILP strategy requested but milp_scheduler is unavailable."
            )

        success = assign_requests_milp(requests, elevators)
        if not success:
            print("[MILP] Scheduling did not succeed.")
            return
    else:
        assign_requests_greedy(requests, elevators)

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

    # 4. 输出与可视化
    # print_elevator_queues(elevators)

    passenger_total_time = passenger_metrics.total_passenger_time
    passenger_wait_time = passenger_metrics.total_wait_time
    passenger_in_cab_time = passenger_metrics.total_in_cab_time
    reported_wait_penalty = passenger_metrics.wait_penalty_total
    served_count = passenger_metrics.served_count

    if cfg.SIM_ENABLE_LOG:
        log_results(
            elevators,
            system_time,
            running_energy,
            objective_breakdown,
            passenger_total_time,
            passenger_wait_time,
            passenger_in_cab_time,
            reported_wait_penalty,
            emptyload_energy,
        )

    if cfg.SIM_ENABLE_PLOTS:
        plot_elevator_movements(elevators)
        plot_elevator_movements_time(elevators)

    print(
        f"Strategy: {used_strategy}\n"
        f"Passenger Time: {passenger_total_time:.2f}s "
        f"(waiting {passenger_wait_time:.2f}s | in-cab {passenger_in_cab_time:.2f}s)\n"
        f"Wait Penalty (super-linear): {reported_wait_penalty:.2f}\n"
        f"Energy (total): {running_energy:.2f}J\n"
        f"Energy (empty-load): {emptyload_energy:.2f}J\n"
        f"System Active Time: {system_time:.2f}s\n"
        f"Served Requests: {served_count}\n"
        f"Objective Cost: {objective_breakdown.total_cost:.2f}\n"
        f"  - Wait Cost: {objective_breakdown.wait_cost:.2f}\n"
        f"  - Ride Cost: {objective_breakdown.ride_cost:.2f}\n"
        f"  - Running Energy Cost: {objective_breakdown.running_energy_cost:.2f}\n"
        f"  - Empty-load Energy Surcharge: {objective_breakdown.emptyload_energy_cost:.2f}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        option_map = {
            "1": "baseline",
            "baseline": "baseline",
            "2": "milp",
            "milp": "milp",
            "3": "mpc",
            "mpc": "mpc",
        }

        selected = None
        while selected is None:
            user_input = (
                input("请选择调度策略 (1=baseline, 2=milp, 3=mpc): ").strip().lower()
            )
            selected = option_map.get(user_input)
            if selected is None:
                print("输入无效，请重新输入。")

        main(strategy=selected)
