""" muscle swimmer implemented based on OpenAI Gym Swimmer-v3 """

from src.envs.mujoco.muscle_swimmer_gym_v0 import swimmer
import mujoco_py
import os
import random


def test_random():
    """ take random actions """
    # generate xml str
    xml_str = swimmer(xml_file='src/envs/mujoco/assets/swimmer.xml')
    with open('swimmer_temp.xml', 'wb') as f:
        f.write(xml_str)
    # build model
    model = mujoco_py.load_model_from_xml(xml_str.decode("utf-8"))
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    sim_state = sim.get_state()
    while True:
        sim.set_state(sim_state)
        for i in range(1000):
            for j in range(len(sim.data.ctrl)):
                sim.data.ctrl[j] = random.random()
            sim.step()
            viewer.render()
        if os.getenv('TESTING') is not None:
            break


if __name__ == '__main__':
    test_random()
