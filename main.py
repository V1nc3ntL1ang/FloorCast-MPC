import config as cfg
from models import kinematics, energy, temporal, objective
from models.variables import Request


def main():
    # Example: a single elevator trip
    req = Request(id=1, origin=2, destination=10, load=100.0, arrival_time=0.0)

    # Compute motion and energy metrics
    t_move = kinematics.travel_time(req.load, req.origin, req.destination)
    t_hold = temporal.hold_time(req.load, 0)
    direction = "up" if req.destination > req.origin else "down"
    distance = abs(req.destination - req.origin) * cfg.BUILDING_FLOOR_HEIGHT
    e_motion = energy.segment_energy(req.load, distance, direction)

    # Combine objective
    z_value = objective.compute_objective(t_move + t_hold, e_motion)

    print(f"Origin Floor: {req.origin}")
    print(f"Destination Floor: {req.destination}")
    print(f"Travel Time: {t_move:.2f} s")
    print(f"Hold Time: {t_hold:.2f} s")
    print(f"Motion Energy: {e_motion:.2f} J")
    print(f"Objective Value: {z_value:.2f}")


if __name__ == "__main__":
    main()
