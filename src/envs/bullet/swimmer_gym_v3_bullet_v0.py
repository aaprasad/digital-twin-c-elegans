""" work in progress """

""" This is modeled after Swimmer-v3 from OpenAI Gym
- with Bullet physics instead of MuJoCo
- tries to be the same as Swimmer-v3
"""

import gym
import pybullet
import pybullet_data
import pybullet_envs
# from pybullet_envs.robot_locomotors import WalkerBase
import os
import numpy as np

"""
class SwimmerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # connect pybullet, define action_space & observation_space
        # self.step_counter = 0
        pybullet.connect(pybullet.DIRECT)  # pybullet.GUI
        # pybullet.resetDebugVisualizerCamera(
        #     cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2]
        # )
        self.swimmer = None
        self.action_space = gym.spaces.Box(
            low=np.array([-1.] * 2, dtype=np.float32), high=np.array([1.] * 2, dtype=np.float32)
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 8, dtype=np.float64), high=np.array([np.inf] * 8, dtype=np.float64)
        )

    def step(self, action):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = pybullet.getQuaternionFromEuler([0., -np.pi, np.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = pybullet.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = pybullet.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

        pybullet.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], pybullet.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        pybullet.stepSimulation()

        state_object, _ = pybullet.getBasePositionAndOrientation(self.objectUid)
        state_robot = pybullet.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (pybullet.getJointState(self.pandaUid,9)[0], pybullet.getJointState(self.pandaUid, 10)[0])



        if state_object[2]>0.45:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        info = {'object_position': state_object}
        self.observation = state_robot + state_fingers
        state = np.array(self.observation).astype(np.float32)
        return state, reward, done, info

    def reset(self):
        # self.step_counter = 0
        pybullet.resetSimulation()
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        # urdfRootPath=pybullet_data.getDataPath()
        # pybullet.setGravity(0, 0, -10)

        # planeUid = pybullet.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        # rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        # self.pandaUid = pybullet.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        self.swimmer = pybullet.loadMJCF('assets/swimmer.xml')  # UID
        # for i in range(7):
        #     pybullet.resetJointState(self.pandaUid,i, rest_poses[i])
        # pybullet.resetJointState(self.pandaUid, 9, 0.08)
        # pybullet.resetJointState(self.pandaUid,10, 0.08)
        pybullet.resetJointState(self.swimmer, , )
        # tableUid = pybullet.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        # trayUid = pybullet.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        # state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        # self.objectUid = pybullet.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        # state_robot = pybullet.getLinkState(self.pandaUid, 11)[0]
        # state_fingers = (pybullet.getJointState(self.pandaUid,9)[0], pybullet.getJointState(self.pandaUid, 10)[0])
        # self.observation = state_robot + state_fingers
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        # return np.array(self.observation).astype(np.float32)
        return np.array(self.observation, dtype=np.float64)

    def render(self, mode='human'):
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = pybullet.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = pybullet.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    # def _get_state(self):
    #     return self.observation

    def close(self):
        # disconnect pybullet
        pybullet.disconnect()
"""


pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(os.path.join(os.getcwd(), 'src/envs/bullet'))
pybullet.loadMJCF('assets/swimmer.xml')

while True:
    pybullet.stepSimulation()
