""" swimmer: chemotaxis """

from swimmer_gym_v3_v2 import make_swimmer


def test_random():
    """ take random actions
    multibody model:
        - radius=0.04mm, body_len=0.1mm, n_bodies=12, q_max=0.69rad (~39.53409 degrees)
        - joint_size=0.1 (radius) -> body_len=0.25
        - citation: A computational model of internal representations of chemical gradients in environments for chemotaxis of Caenorhabditis elegans
    """
    env = make_swimmer(n_bodies=12, joint_range='-40 40', body_len=0.25, camera_pos='0 -6 6')
    observation = env.reset()
    for i in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()


if __name__ == '__main__':
    test_random()
