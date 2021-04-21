""" nematode with specific `joint_range`, `body_len` and `arrangement`  """

from src.envs.mujoco.nematode_gym_v0 import nematode
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os


def make_nematode(joint_range, body_len, arrangement=None, camera_pos=None):
    """ create nematode env """
    # generate xml str
    xml_folder = 'src/envs/mujoco/assets/'
    xml_str = nematode(joint_range=joint_range, body_len=body_len, xml_file=os.path.join(xml_folder, 'swimmer.xml'), arrangement=arrangement, camera_pos=camera_pos)
    # write temp xml file, make env and delete temp file
    xml_file = os.path.join(xml_folder, 'nematode_temp.xml')
    with open(xml_file, 'wb') as f:
        f.write(xml_str)
    env = gym.make('Swimmer-v3', xml_file=os.path.join(os.getcwd(), xml_file))
    if os.path.exists(xml_file):
        os.remove(xml_file)
    # action: Box(0.0, 1.0, (95,), float32)
    print(env.action_space, env.action_space.low, env.action_space.high)
    # observation: Box(-inf, inf, (54,), float64)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    return env


def test_random():
    """ take random actions """
    # body_len >= 0.2
    env = make_nematode(joint_range='-100 100', body_len=0.26, arrangement=None, camera_pos='0 -5 5')
    # record video
    os.makedirs('video', exist_ok=True)
    rec = VideoRecorder(env, base_path='video/nematode_gym_v0', enabled=True)
    # run
    observation = env.reset()
    rec.capture_frame()
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rec.capture_frame()
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    rec.close()
    env.close()


if __name__ == '__main__':
    test_random()
