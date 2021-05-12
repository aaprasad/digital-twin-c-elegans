""" muscle swimmer implemented based on OpenAI Gym Swimmer-v3 """

from src.envs.mujoco.muscle_swimmer_gym_v0 import swimmer
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import mujoco_py
import numpy as np
import os


def test_random():
    """ take random actions """
    # generate xml str
    xml_str = swimmer(xml_file='src/envs/mujoco/assets/swimmer.xml')
    # build model
    model = mujoco_py.load_model_from_xml(xml_str.decode("utf-8"))
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    sim_state = sim.get_state()
    while True:
        sim.set_state(sim_state)
        for i in range(1000):
            for j in range(len(sim.data.ctrl)):
                sim.data.ctrl[j] = np.random.random()
            sim.step()
            viewer.render()
        if os.getenv('TESTING') is not None:
            break


def make_swimmer():
    """ create swimmer env """
    # generate xml str
    xml_folder = 'src/envs/mujoco/assets/'
    xml_str = swimmer(xml_file=os.path.join(xml_folder, 'swimmer.xml'))
    # write temp xml file, make env and delete temp file
    xml_file = os.path.join(xml_folder, 'swimmer_temp.xml')
    with open(xml_file, 'wb') as f:
        f.write(xml_str)
    env = gym.make('Swimmer-v3', xml_file=os.path.join(os.getcwd(), xml_file))
    if os.path.exists(xml_file):
        os.remove(xml_file)

    # action: Box(0.0, 1.0, (4,), float32), muscles' actions
    print(env.action_space, env.action_space.low, env.action_space.high)
    # observation: Box(-inf, inf, (8,), float64), the same as Swimmer-v3
    # the states of tendon/muscle need to be added
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    return env


def test_random_gym():
    """ take random actions with gym env """
    # make env
    env = make_swimmer()
    # record video
    os.makedirs('video', exist_ok=True)
    rec = VideoRecorder(env, base_path='video/muscle_swimmer_gym_v0', enabled=True)  # Create the video recorder
    # run and record video
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
    # test_random()
    test_random_gym()
