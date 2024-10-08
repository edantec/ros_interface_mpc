import numpy as np
import pinocchio as pin
import os 
import example_robot_data

CURRENT_DIRECTORY = os.getcwd()
URDF_DIRECTORY = CURRENT_DIRECTORY + '/src/cpp_pubsub/urdf'

URDF_FILENAME = "talos_reduced.urdf"
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return np.array([x, y, z, w])

def loadTalos():
    robotComplete = example_robot_data.load("talos")
    qComplete = robotComplete.model.referenceConfigurations["half_sitting"]

    locked_joints = [20,21,22,23,28,29,30,31]
    locked_joints += [32, 33]
    robot = robotComplete.buildReducedRobot(locked_joints, qComplete)
    rmodel: pin.Model = robot.model
    q0 = rmodel.referenceConfigurations["half_sitting"]

    URDF_SUBPATH = "/talos_data/robots/talos_reduced.urdf"
    package_dir = example_robot_data.getModelPath(URDF_SUBPATH)
    file_path = package_dir + URDF_SUBPATH

    with open(file_path, 'r') as file:
        file_content = file.read()

    geom_model=pin.GeometryModel()
    pin.buildGeomFromUrdfString(rmodel, file_content, pin.GeometryType.VISUAL, geom_model, package_dir)

    return robotComplete.model, rmodel, qComplete, q0, geom_model

def loadGo2():
    """ robot = example_robot_data.load("go2")
    rmodel = robot.model
    q0 = rmodel.referenceConfigurations["standing"]

    URDF_SUBPATH = "/go2_description/urdf/go2.urdf"
    package_dir = example_robot_data.getModelPath(URDF_SUBPATH)
    file_path = package_dir + URDF_SUBPATH

    with open(file_path, 'r') as file:
        file_content = file.read()

    geom_model=pin.GeometryModel()
    pin.buildGeomFromUrdfString(rmodel, file_content, pin.GeometryType.VISUAL, geom_model, package_dir) """
    
    package_dir = '/home/edantec/Documents/git/unitree_ros/robots'
    urdf_path = package_dir + '/go2_description/urdf/go2_description.urdf'
    rmodel = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())

    with open(urdf_path, 'r') as file:
        file_content = file.read()
    
    geom_model=pin.GeometryModel()
    pin.buildGeomFromUrdfString(rmodel, file_content, pin.GeometryType.VISUAL, geom_model, package_dir)

    #model, geom_model, _ = pin.buildModelsFromMJCF(PARENT_DIRECTORY + "/urdf/go2.xml")
    
    return rmodel, geom_model


def computeCoP(LF_pose, RF_pose, LF_force, LF_torque, RF_force, RF_torque):
    cop_total = np.zeros(3)
    total_z_force = 0
    if LF_force[2] > 1.:
        local_cop_left = np.array([-LF_torque[1] / LF_force[2],
                                  LF_torque[0] / LF_force[2], 
                                  0.0]
        )
        cop_total += (LF_pose.rotation @ local_cop_left +
                      LF_pose.translation) * LF_force[2]
        total_z_force += LF_force[2]
    if RF_force[2] > 1.:
        local_cop_right = np.array([-RF_torque[1] / RF_force[2],
                                   RF_torque[0] / RF_force[2], 
                                   0.0]
        )
        cop_total += (RF_pose.rotation @ local_cop_right + 
                      RF_pose.translation) * RF_force[2]
        total_z_force += RF_force[2]
    if (total_z_force < 1.): print("Zero force detected")
    cop_total /= total_z_force
    
    return cop_total

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)