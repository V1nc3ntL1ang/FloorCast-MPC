from dataclasses import dataclass


@dataclass
class Request:
    id: int
    origin: int
    destination: int
    load: float
    arrival_time: float


@dataclass
class ElevatorState:
    id: int
    floor: int
    load: float = 0.0
    direction: str = "idle"  # "up", "down", or "idle"
