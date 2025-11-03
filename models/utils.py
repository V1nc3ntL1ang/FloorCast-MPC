import math
import os
import random
from datetime import datetime
from typing import Sequence, Tuple


DEFAULT_PLOT_DIR = (
    "/home/v1nc3nt/WinDesktop/SCUT/作业/优化方法/LoadAwareElevator/results/plots"
)
DEFAULT_SUMMARY_DIR = (
    "/home/v1nc3nt/WinDesktop/SCUT/作业/优化方法/LoadAwareElevator/results/summary"
)


def h2s(hour, minute: int = 0) -> float:
    """
    Convert time-of-day to seconds / 将时刻转换为秒数.
    支持以下输入形式：
      - 两个整数：h2s(7, 30)
      - 单个整数：h2s(7)
      - 字符串：h2s("7:30") 或 h2s("18:45")
    """
    if isinstance(hour, str):
        parts = hour.strip().split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    return hour * 3600.0 + minute * 60.0


def duration_seconds(start, end) -> float:
    """Compute interval length in seconds (cross-day safe) / 计算时间段长度（秒），支持跨日。"""
    if isinstance(start, tuple):
        start_s = h2s(*start)
    else:
        start_s = h2s(start)
    if isinstance(end, tuple):
        end_s = h2s(*end)
    else:
        end_s = h2s(end)
    if end_s < start_s:
        end_s += 24 * 3600  # 跨日修正
    return end_s - start_s


def validate_ratios(
    r_origin1: float, r_dest1: float, r_other: float
) -> Tuple[float, float]:
    """Normalize ratios and return cumulative cut-offs / 归一化方向比例并返回累计边界。"""
    total = r_origin1 + r_dest1 + r_other
    if total <= 0:
        raise ValueError("All ratios are zero or negative.")

    r_origin1 /= total
    r_dest1 /= total
    r_other /= total

    c1 = r_origin1
    c2 = r_origin1 + r_dest1
    return c1, c2


def rand_upper_floor(max_floor: int) -> int:
    """Randomly select a floor in [2, max_floor] / 随机取 2..max_floor 的楼层。"""
    if max_floor < 2:
        raise ValueError("max_floor must be at least 2.")
    return random.randint(2, max_floor)


def rand_other_pair(max_floor: int) -> Tuple[int, int]:
    """Return two distinct non-1 floors / 返回两个不等且不为 1 的楼层（范围 2..max_floor）。"""
    a = rand_upper_floor(max_floor)
    b = rand_upper_floor(max_floor)
    while b == a:
        b = rand_upper_floor(max_floor)
    return a, b


# ============================================================
# Logging & Result Summaries
# ============================================================


def ensure_directory(path: str):
    """Create directory when missing / 若目录不存在则创建。"""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _ensure_matplotlib():
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("KMP_AFFINITY", "disabled")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"[Plot Skipped] {exc}")
        return None


def _collect_global_entries(elevators):
    entries = []
    for elev in elevators:
        for req in getattr(elev, "served_requests", []):
            origin = getattr(req, "origin", None)
            destination = getattr(req, "destination", None)
            if origin is None or destination is None:
                continue
            start_time = getattr(req, "pickup_time", None)
            if start_time is None:
                start_time = getattr(req, "origin_arrival_time", None)
            if start_time is None:
                start_time = getattr(req, "arrival_time", 0.0)
            entries.append((float(start_time), elev.id, req))
    entries.sort(
        key=lambda item: (
            item[0],
            getattr(item[2], "arrival_time", 0.0),
            getattr(item[2], "id", 0),
        )
    )
    return entries


def plot_elevator_movements(
    elevators,
    filename: str | None = None,
    *,
    strategy_label: str | None = None,
) -> None:
    """
    Global task-order view (index vs floor) / 全局任务顺序视图（索引-楼层）。
    """
    plt = _ensure_matplotlib()
    if plt is None:
        return

    entries = _collect_global_entries(elevators)
    if not entries:
        print("[Plot Skipped] No served requests to plot (global order).")
        return

    if filename is None:
        filename = os.path.join(DEFAULT_PLOT_DIR, "elevator_schedule_global.png")

    label = strategy_label or "Strategy"
    ensure_directory(os.path.dirname(filename))
    plt.figure(figsize=(25.6, 14.4))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    color_map = {elev.id: colors[i % len(colors)] for i, elev in enumerate(elevators)}

    series: dict[int, tuple[list[float], list[float]]] = {}
    for idx, (_, elev_id, req) in enumerate(entries):
        origin = getattr(req, "origin", None)
        destination = getattr(req, "destination", None)
        if origin is None or destination is None:
            continue
        xs, ys = series.setdefault(elev_id, ([], []))
        xs.extend([idx, idx + 1])
        ys.extend([origin, destination])

    for elev_id, (xs, ys) in series.items():
        if not xs:
            continue
        plt.plot(
            xs,
            ys,
            color=color_map[elev_id],
            linewidth=1.2,
            alpha=0.85,
            label=f"Elevator {elev_id}",
        )

    plt.xlabel("Global Task Index")
    plt.ylabel("Floor Level")
    plt.title(f"Global Service Sequence ({label})")
    if series:
        plt.legend()
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Plot Saved] Global sequence plot saved to: {filename}")


def plot_elevator_movements_time(
    elevators,
    filename: str | None = None,
    *,
    strategy_label: str | None = None,
) -> None:
    """
    Global service timeline (time vs floor) / 全局服务时间线（时间-楼层）。
    """
    plt = _ensure_matplotlib()
    if plt is None:
        return
    from matplotlib.ticker import FuncFormatter

    entries = _collect_global_entries(elevators)
    if not entries:
        print("[Plot Skipped] No served requests to plot (global timeline).")
        return

    if filename is None:
        filename = os.path.join(DEFAULT_PLOT_DIR, "elevator_schedule_time_global.png")

    label = strategy_label or "Strategy"
    ensure_directory(os.path.dirname(filename))
    plt.figure(figsize=(25.6, 14.4))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    color_map = {elev.id: colors[i % len(colors)] for i, elev in enumerate(elevators)}
    legend_added = set()

    for start_time, elev_id, req in entries:
        origin = getattr(req, "origin", None)
        destination = getattr(req, "destination", None)
        if origin is None or destination is None:
            continue
        pickup = getattr(req, "pickup_time", None)
        if pickup is None:
            pickup = getattr(req, "origin_arrival_time", None)
        if pickup is None:
            pickup = start_time
        dropoff = getattr(req, "destination_arrival_time", None)
        if dropoff is None:
            dropoff = getattr(req, "dropoff_time", None)
        if dropoff is None:
            dropoff = pickup
        color = color_map.get(elev_id, "C7")
        label_entry = (
            f"Elevator {elev_id}" if elev_id not in legend_added else "_nolegend_"
        )
        plt.plot(
            [pickup, dropoff],
            [origin, destination],
            color=color,
            marker="o",
            alpha=0.9,
            linewidth=1.4,
            label=label_entry,
        )
        legend_added.add(elev_id)

    plt.xlabel("Time of Day (s)")
    plt.ylabel("Floor Level")
    plt.title(f"Global Service Timeline ({label})")
    ax = plt.gca()

    def _format_hhmm(x, pos):
        total = int(max(x, 0)) % (24 * 3600)
        hour = total // 3600
        minute = (total % 3600) // 60
        return f"{hour:02d}:{minute:02d}"

    ax.xaxis.set_major_formatter(FuncFormatter(_format_hhmm))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Plot Saved] Global timeline plot saved to: {filename}")


def plot_wait_distribution(
    wait_data: Sequence[Tuple[str, Sequence[float]]],
    filename=os.path.join(DEFAULT_PLOT_DIR, "wait_distribution.png"),
) -> None:
    """
    Plot wait-time histograms for multiple strategies / 绘制多策略等待时间分布直方图。
    wait_data expects (label, waits) pairs / 输入为 (策略名称, 等待时间序列)。
    """
    plt = _ensure_matplotlib()
    if plt is None:
        return

    series = [
        (label, [w for w in waits if w is not None]) for label, waits in wait_data
    ]
    series = [(label, waits) for label, waits in series if waits]

    if not series:
        print("[Plot Skipped] No wait-time data available.")
        return

    ensure_directory(os.path.dirname(filename))
    plt.figure(figsize=(12, 6))

    total_samples = sum(len(waits) for _, waits in series)
    bins = max(10, min(60, int(math.sqrt(total_samples))))

    for label, waits in series:
        plt.hist(
            waits,
            bins=bins,
            alpha=0.6,
            label=label,
            edgecolor="black",
            linewidth=0.4,
        )

    plt.xlabel("Wait Time (s)")
    plt.ylabel("Frequency")
    plt.title("Passenger Wait Time Distribution by Strategy")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Plot Saved] Wait-time distribution saved to: {filename}")


def print_elevator_queues(elevators):
    """
    Print served requests for each elevator / 输出各电梯的服务队列。
    """
    print("\n=== Elevator Queues ===")
    for elev in elevators:
        print(f"Elevator {elev.id}:")
        if not elev.served_requests:
            print("  (No requests assigned)")
            continue
        for req in elev.served_requests:
            print(
                f"  Req#{req.id}: Floor {req.origin} → {req.destination} | Load={req.load:.1f}kg"
            )
    print("========================\n")


def _format_time(value):
    """Pretty-print time value / 时间值格式化。"""
    return f"{value:.1f}s" if value is not None else "N/A"


def _format_table(value, width):
    """Pad table cell / 表格列格式化。"""
    return f"{value:>{width}.1f}" if value is not None else "N/A".rjust(width)


def log_results(
    elevators,
    system_time,
    running_energy,
    objective_breakdown,
    passenger_total_time,
    passenger_wait_time,
    passenger_in_cab_time,
    passenger_wait_penalty,
    emptyload_energy,
    theoretical_breakdown,
    theoretical_in_cab_time,
    theoretical_running_energy,
    theoretical_wait_time,
    theoretical_wait_penalty,
    *,
    strategy_label: str | None = None,
    outdir: str | None = None,
):
    """Enhanced logging with per-elevator/per-request views / 输出增强日志（按电梯与按请求）。"""
    target_dir = outdir or DEFAULT_SUMMARY_DIR
    if strategy_label:
        target_dir = os.path.join(target_dir, f"strategy_{strategy_label}")

    ensure_directory(target_dir)

    by_elevator_path = os.path.join(target_dir, "summary_by_elevator.txt")
    with open(by_elevator_path, "w", encoding="utf-8") as f:
        f.write("=== Elevator Service Summary (by Elevator) ===\n")
        f.write(f"Generated at: {datetime.now()}\n")
        f.write(f"System Active Time: {system_time:.2f} s\n")
        f.write(f"Total Energy (running): {running_energy:.2f} J\n")
        f.write(
            "Passenger Time: {:.2f} s (wait {:.2f} s | in-cab {:.2f} s)\n".format(
                passenger_total_time, passenger_wait_time, passenger_in_cab_time
            )
        )
        f.write("Theoretical lower bound (idealized) reference:\n")
        f.write(
            "  Wait: {wait:.2f} s (penalty {penalty:.2f}) | "
            "Ride: {time:.2f} s | Running Energy LB: {energy:.2f} J | "
            "Cost: {cost:.2f}\n\n".format(
                wait=theoretical_wait_time,
                penalty=theoretical_wait_penalty,
                time=theoretical_in_cab_time,
                energy=theoretical_running_energy,
                cost=theoretical_breakdown.total_cost,
            )
        )
        f.write(f"Aggregated Wait Penalty: {passenger_wait_penalty:.2f}\n")
        f.write(f"Empty-load Energy: {emptyload_energy:.2f} J\n")
        f.write("Objective Cost Breakdown:\n")
        f.write(
            "  Total: {total:.2f} | Wait: {wait:.2f} | Ride: {ride:.2f} | "
            "Running Energy: {run:.2f} | Empty-load Surcharge: {empty:.2f}\n\n".format(
                total=objective_breakdown.total_cost,
                wait=objective_breakdown.wait_cost,
                ride=objective_breakdown.ride_cost,
                run=objective_breakdown.running_energy_cost,
                empty=objective_breakdown.emptyload_energy_cost,
            )
        )

        for elev in elevators:
            f.write(
                f"[Elevator {elev.id}] Served {len(elev.served_requests)} requests\n"
            )
            if not elev.served_requests:
                f.write("  (No requests)\n\n")
                continue

            for req in elev.served_requests:
                origin_arrival = getattr(req, "origin_arrival_time", None)
                pickup = getattr(req, "pickup_time", None)
                dest_arrival = getattr(req, "destination_arrival_time", None)
                completion = getattr(req, "dropoff_time", None)

                wait_time = (
                    origin_arrival - req.arrival_time
                    if origin_arrival is not None
                    else None
                )
                onboard_time = (
                    dest_arrival - origin_arrival
                    if dest_arrival is not None and origin_arrival is not None
                    else None
                )
                total_duration = (
                    dest_arrival - req.arrival_time
                    if dest_arrival is not None
                    else None
                )

                f.write(
                    "  Req#{rid:03d}: {origin:02d} → {dest:02d} | "
                    "Load={load:.1f}kg | Arr={arr:.1f}s | "
                    "Orig={orig} | Pick={pick} | Dest={dest_arr} | Comp={comp} | "
                    "Wait={wait} | Onboard={onboard} | Total={total}\n".format(
                        rid=req.id,
                        origin=req.origin,
                        dest=req.destination,
                        load=req.load,
                        arr=req.arrival_time,
                        orig=_format_time(origin_arrival),
                        pick=_format_time(pickup),
                        dest_arr=_format_time(dest_arrival),
                        comp=_format_time(completion),
                        wait=_format_time(wait_time),
                        onboard=_format_time(onboard_time),
                        total=_format_time(total_duration),
                    )
                )
            f.write("\n")

    print(f"[Log Saved] Elevator summary → {by_elevator_path}")

    all_requests = []
    for elev in elevators:
        for req in elev.served_requests:
            all_requests.append((elev.id, req))

    all_requests.sort(key=lambda x: x[1].arrival_time)

    by_request_path = os.path.join(target_dir, "summary_by_request.txt")
    with open(by_request_path, "w", encoding="utf-8") as f:
        f.write("=== Request Timeline Summary (by Request) ===\n")
        f.write(f"Generated at: {datetime.now()}\n")
        f.write(f"Total Elevators: {len(elevators)}\n")
        f.write(f"Total Requests: {len(all_requests)}\n\n")

        header = (
            "ReqID | Elevator | Origin→Dest | Load(kg) | Start(s) | ElevArr(s) | "
            "Board(s) | DestArr(s) | Complete(s) | WaitQ(s) | InCab(s) | Total(s)\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")

        for eid, req in all_requests:
            origin_arrival = getattr(req, "origin_arrival_time", None)
            pickup = getattr(req, "pickup_time", None)
            dest_arrival = getattr(req, "destination_arrival_time", None)
            completion = getattr(req, "dropoff_time", None)

            wait_time = (
                origin_arrival - req.arrival_time
                if origin_arrival is not None
                else None
            )
            onboard_time = (
                dest_arrival - origin_arrival
                if dest_arrival is not None and origin_arrival is not None
                else None
            )
            total_duration = (
                dest_arrival - req.arrival_time if dest_arrival is not None else None
            )

            f.write(
                f"{req.id:4d} | {eid:8d} | "
                f"{req.origin:2d}→{req.destination:2d} | "
                f"{req.load:8.1f} | "
                f"{req.arrival_time:9.1f} | {_format_table(origin_arrival, 10)} | {_format_table(pickup, 9)} | "
                f"{_format_table(dest_arrival, 9)} | {_format_table(completion, 11)} | "
                f"{_format_table(wait_time, 7)} | {_format_table(onboard_time, 8)} | {_format_table(total_duration, 8)}\n"
            )

    print(f"[Log Saved] Request summary → {by_request_path}")
