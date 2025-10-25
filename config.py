# ============================================================
# Load-Aware Elevator Scheduling — Configuration File
# ============================================================

# ------------------------
# Building Parameters
# ------------------------
BUILDING_FLOORS = 15  # total number of floors
BUILDING_FLOOR_HEIGHT = 3.5  # meters between adjacent floors
ELEVATOR_COUNT = 1  # number of elevators
ELEVATOR_CAPACITY = 1200.0  # kg, maximum allowed load

# ------------------------
# Kinematic Parameters
# ------------------------
g = 9.81  # N/m
KIN_MAX_SPEED_UP_EMPTY = 3.0  # m/s, upward, no load
KIN_MAX_SPEED_UP_FULL = 2.5  # m/s, upward, full load
KIN_MAX_SPEED_DOWN_EMPTY = 3.0  # m/s, downward, no load
KIN_MAX_SPEED_DOWN_FULL = 2.6  # m/s, downward, full load
KIN_SPEED_DECAY_RATE = 1.2  # exponential decay factor for speed vs. load

KIN_ACC_UP_EMPTY = 1.2  # m/s², acceleration upward, no load
KIN_ACC_UP_FULL = 0.9  # m/s², acceleration upward, full load
KIN_DEC_DOWN_EMPTY = 1.2  # m/s², deceleration downward, no load
KIN_DEC_DOWN_FULL = 1.0  # m/s², deceleration downward, full load
KIN_ACC_DECAY_RATE = 1.3  # exponential decay factor for acceleration vs. load

# ------------------------
# Temporal Parameters
# ------------------------
HOLD_BASE_TIME = 1.5  # s, minimum holding (door open + reaction)
HOLD_EFF_NORMAL = 0.002  # s/kg, normal boarding rate
HOLD_EFF_CONGESTED = 0.005  # s/kg, congested boarding rate
HOLD_CONGESTION_THRESHOLD = 400  # kg, threshold for congestion effects

# ------------------------
# Energy Parameters
# ------------------------
ENERGY_CAR_MASS = 600.0  # kg, elevator cabin
ENERGY_COUNTERWEIGHT_MASS = 500.0  # kg, counterweight
ENERGY_FRICTION_PER_METER = 50.0  # J/m, mechanical loss
ENERGY_MOTOR_EFFICIENCY = 0.85  # efficiency ratio (0–1)
ENERGY_STANDBY_POWER = 500.0  # W, base power per elevator

# ------------------------
# Simulation Parameters
# ------------------------
SIM_TIME_HORIZON = 300  # s, total simulated duration
SIM_TIME_STEP = 1.0  # s, simulation granularity
SIM_RANDOM_SEED = 42

# ------------------------
# Objective Weights
# ------------------------
WEIGHT_TIME = 1.0  # weight for total service time
WEIGHT_ENERGY = 0.001  # weight for total energy consumption
