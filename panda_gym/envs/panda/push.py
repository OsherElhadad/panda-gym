from gym import utils
from panda_gym.envs import panda_env

MODEL_JSON_PATH = 'push.json'


class PandaPushEnv(panda_env.PandaEnv, utils.EzPickle):
    def __init__(self, render=False, reward_type='sparse'):
        initial_qpos = {
            'object': [1.7, 1.1, 0.425, 1., 0., 0., 0.],
        }
        panda_env.PandaEnv.__init__(
            self, MODEL_JSON_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, render=render)
        utils.EzPickle.__init__(self)
