import pinocchio as pin
import numpy as np
import hppfcl
from simulation_utils import (
    addFloor,
    setPhysicsProperties,
    removeBVHModelsIfAny,
    Simulation,
    addSystemCollisionPairs
)
from simulation_args import SimulationArgs

args = SimulationArgs().parse_args()
allowed_solvers = ["ADMM", "PGS"]
if args.contact_solver not in allowed_solvers:
    print(
        f"Error: unsupported simulator. Avalaible simulators: {allowed_solvers}. Exiting"
    )
    exit(1)
np.random.seed(args.seed)
pin.seed(args.seed)

# ============================================================================
# SCENE CREATION
# ============================================================================
# Create model
model, geom_model, _ = pin.buildModelsFromMJCF("../urdf/go2.xml")

q0 = np.array([0, 0, 1.335, 0, 0, 0, 1,
    0.068, 0.785, -1.440,
    -0.068, 0.785, -1.440,
    0.068, 0.785, -1.440,
    -0.068, 0.785, -1.440,
])
v0 = np.zeros(model.nv)

# Add plane in geom_model
visual_model = geom_model.copy()
addFloor(geom_model, visual_model)
setPhysicsProperties(geom_model, args.material, args.compliance)
removeBVHModelsIfAny(geom_model)
addSystemCollisionPairs(model, geom_model, q0)

# Create the simulator object
simulator = Simulation(model, geom_model, visual_model, q0, v0, args)

for t in range(50):
    torque = np.zeros(18)
    simulator.execute(torque)
