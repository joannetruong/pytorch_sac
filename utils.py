import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
import math
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv, NavigateRandomEnvSim2Real
from gibson2.data.utils import get_train_models
import gibson2
from collections import OrderedDict

def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def make_gibson_env(cfg):
    """Helper function to create gibson environment"""
#    env = NavigateRandomEnv(config_file=cfg.gibson_cfg, mode='headless')
    sim2real_track = 'static'
    env = NavigateRandomEnvSim2Real(config_file=cfg.gibson_cfg,
                                    mode='headless',
                                    action_timestep=1.0 / 60.0,
                                    physics_timestep=1.0 / 40.0,
                                    track=sim2real_track)

    return env

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        h, w, c = env.observation_space.spaces["rgb"].shape
        rgb_shp = (c, h, w)
        sensor = np.array([1, 2])
        s_shp = sensor.shape
        self.sensor_space = gym.spaces.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(s_shp),
                                           dtype=env.observation_space.spaces["sensor"].dtype)
        self.rgb_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((rgb_shp[0] * k,) + rgb_shp[1:]),
            dtype=env.observation_space.spaces["rgb"].dtype
        )
        self.observation_space=OrderedDict()
        self.observation_space['sensor'] = self.sensor_space
        self.observation_space['rgb'] = self.rgb_space
#        self._max_episode_steps = env._max_episode_steps
        self.max_step = env.max_step
        self.robot = env.robot

    def reset(self):
        obs = self.env.reset()
        obs["rgb"] = (obs["rgb"] * 255).round().astype(np.uint8)
        obs["rgb"] = np.moveaxis(obs["rgb"], 2, 0)
        for _ in range(self._k):
            self._frames.append(obs["rgb"])
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["rgb"] = (obs["rgb"] * 255).round().astype(np.uint8)
        obs["rgb"] = np.moveaxis(obs["rgb"], 2, 0)
        self._frames.append(obs["rgb"])
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        assert len(self._frames) == self._k
        obs["rgb"] = np.concatenate(list(self._frames), axis=0)
        return obs

    def get_rgb(self):
        frame = self.env.get_rgb()
        frame = (frame * 255).round()
        return frame

    def get_initial_pos(self):
        return self.env.initial_pos

    def get_target_pos(self):
        return self.env.target_pos

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        h, w, c = env.observation_space.spaces["rgb"].shape
        rgb_shp = (c, h, w)
        sensor = np.array([1, 2])
        s_shp = sensor.shape
        self.sensor_space = gym.spaces.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(s_shp),
                                           dtype=env.observation_space.spaces["sensor"].dtype)
        self.rgb_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((rgb_shp[0] * k,) + rgb_shp[1:]),
            dtype=env.observation_space.spaces["rgb"].dtype
        )
        self.observation_space=OrderedDict()
        self.observation_space['sensor'] = self.sensor_space
        self.observation_space['rgb'] = self.rgb_space
#        self._max_episode_steps = env._max_episode_steps
        self.max_step = env.max_step

    def reset(self):
        obs = self.env.reset()
        obs["rgb"] = (obs["rgb"] * 255).round().astype(np.uint8)
        obs["rgb"] = np.moveaxis(obs["rgb"], 2, 0)
        for _ in range(self._k):
            self._frames.append(obs["rgb"])
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["rgb"] = (obs["rgb"] * 255).round().astype(np.uint8)
        obs["rgb"] = np.moveaxis(obs["rgb"], 2, 0)
        self._frames.append(obs["rgb"])
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        assert len(self._frames) == self._k
        obs["rgb"] = np.concatenate(list(self._frames), axis=0)
        return obs

    def get_rgb(self):
        frame = self.env.get_rgb()
        frame = (frame * 255).round()
        return frame

    def get_initial_pos(self):
        return self.env.initial_pos

    def get_target_pos(self):
        return self.env.target_pos

class FrameStackDepth(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self.height, self.width, self.channels = env.observation_space.spaces["depth"].shape
        
        depth_shp = (self.channels, self.height, self.width)
        sensor = np.array([1, 2])
        s_shp = sensor.shape
        self.sensor_space = gym.spaces.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(s_shp),
                                           dtype=env.observation_space.spaces["sensor"].dtype)
        self.depth_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((depth_shp[0] * k,) + depth_shp[1:]),
            dtype=env.observation_space.spaces["depth"].dtype
        )
        self.observation_space=OrderedDict()
        self.observation_space['sensor'] = self.sensor_space
        self.observation_space['depth'] = self.depth_space
#        self._max_episode_steps = env._max_episode_steps
        self.max_step = env.max_step
        self.target_dist_min = env.target_dist_min
        self.target_dist_max = env.target_dist_max
        self.current_step = env.current_step
        self.robot = env.robots[0]
        self.high_level_action_space = env.robots[0].high_level_action_space

    def reset(self, eval=False):
        obs = self.env.reset(eval)
        obs["depth"] = (obs["depth"] * 255).round().astype(np.uint8)
        obs["depth"] = np.moveaxis(obs["depth"], 2, 0)
        for _ in range(self._k):
            self._frames.append(obs["depth"])
        return self._get_obs(obs)

    def step(self, action, low_level=False):
        obs, reward, done, info = self.env.step(action, low_level)
        obs["depth"] = (obs["depth"] * 255).round().astype(np.uint8)
        obs["depth"] = np.moveaxis(obs["depth"], 2, 0)
        self._frames.append(obs["depth"])
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        assert len(self._frames) == self._k
        obs["depth"] = np.concatenate(list(self._frames), axis=0)
        return obs

    def get_rgb(self):
        frame = self.env.get_rgb()
        frame = (frame * 255).round()
        return frame

    def get_depth(self):
        frame = self.env.get_depth()
        frame = (frame * 255).round()
        return frame

    def get_map(self):
        frame = self.env.get_top_down_map()
        return frame

    def get_resolution(self):
        return self.height, self.width

    def get_initial_pos(self):
        return self.env.initial_pos

    def get_target_pos(self):
        return self.env.target_pos

    def set_min_max_dist(self, min, max):
        self.env.target_dist_min = min
        self.env.target_dist_max = max
        self.target_dist_min = min
        self.target_dist_max = max

    def increase_step(self):
        self.env.current_step +=1

    def get_current_step(self):
        return self.env.current_step
