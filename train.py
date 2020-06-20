#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from gibson2.data.utils import get_train_models
import gibson2
from daisy_toolkit.daisy_raibert_controller import DaisyRaibertController, BehaviorParameters
import daisy_hardware.motion_library as motion_library
import pybullet as p

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import dmc2gym
import hydra


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



class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        #self.env = utils.make_env(cfg)
        self.env = utils.make_gibson_env(cfg)

        self.model_dir = utils.make_dir(os.path.join(self.work_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(self.work_dir, 'buffer'))
        print(self.env.observation_space)
        cfg.agent.params.obs_dim = self.env.observation_space.spaces["sensor"].shape[0]-2
        print(self.env.observation_space.spaces["sensor"].shape)
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        tmp = np.array([1, 2])
        self.replay_buffer = ReplayBuffer(tmp.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        self.hz = 240
        p.setTimeStep(1./self.hz)
        self.daisy = self.env.robots[0]
        self.daisy.set_position([0.0, 0.0, 0.3])
        self.daisy.set_orientation([0, 0, 0, 1.])
        self.daisy_state = motion_library.exp_standing(self.daisy, shoulder=1.2, elbow=0.3)
        self.init_state = self.daisy.calc_state()
        self.behavior = BehaviorParameters()

    def evaluate(self):
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            obs = obs["sensor"][:2]
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            self.logger.log('eval/episode_initial_x', self.env.initial_pos[0], self.step)
            self.logger.log('eval/episode_initial_y', self.env.initial_pos[1], self.step)
            self.logger.log('eval/episode_target_x', self.env.target_pos[0], self.step)
            self.logger.log('eval/episode_target_y', self.env.target_pos[1], self.step)
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                self.behavior.target_speed = np.array([action[0], action[1]])
                raibert_controller = DaisyRaibertController(init_state=self.init_state, behavior_parameters=self.behavior)
                time_per_step = 2*self.behavior.stance_duration
                self.daisy_state = motion_library.exp_standing(self.daisy, shoulder=1.2, elbow=0.3)
                for i in range(time_per_step):
                    raibert_action = raibert_controller.get_action(self.init_state, i+1)
                    obs, reward, done, info = self.env.step(raibert_action, low_level=True)
                    self.init_state = self.daisy.calc_state()
                    self.video_recorder.record(self.env, self.cfg.record_params)
                self.env.current_step +=1
                episode_reward += reward

            self.video_recorder.save(f'{self.step}.mp4')
            self.logger.log('eval/episode_reward', episode_reward, self.step)
            self.logger.log('eval/dist_to_goal', info['dist_to_goal'], self.step)
            self.logger.log('eval/success', info['success'], self.step)
            self.logger.log('eval/spl', info['spl'], self.step)
            self.logger.log('eval/num_steps', info['episode_length'], self.step)
            self.logger.log('eval/num_collisions', info['collision_step'], self.step)
            self.logger.log('eval/path_length', info['path_length'], self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    if self.cfg.save_model:
                        self.agent.save(self.model_dir, self.step)
                    if self.cfg.save_buffer:
                        self.replay_buffer.save(self.buffer_dir)

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                obs = obs["sensor"][:2]
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            self.behavior.target_speed = np.array([action[0], action[1]])
            raibert_controller = DaisyRaibertController(init_state=self.init_state, behavior_parameters=self.behavior)
            time_per_step = 2*self.behavior.stance_duration
            self.daisy_state = motion_library.exp_standing(self.daisy, shoulder=1.2, elbow=0.3)
            for i in range(time_per_step):
                raibert_action = raibert_controller.get_action(self.init_state, i+1)
                next_obs, reward, done, info = self.env.step(raibert_action, low_level=True)
                self.init_state = self.daisy.calc_state()
            self.env.current_step +=1
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env.max_step else done
            episode_reward += reward

            next_obs = next_obs["sensor"][:2]
            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1



@hydra.main(config_path="config/train_daisy.yaml", strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
