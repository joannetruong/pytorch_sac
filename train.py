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
from gibson2.utils.utils import l2_distance
import gibson2
from daisy_toolkit.daisy_raibert_controller import DaisyRaibertController, BehaviorParameters
import daisy_hardware.motion_library as motion_library
import pybullet as p

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import hydra

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
        self.env = utils.make_gibson_env(cfg)
        self.eval_env = utils.make_gibson_env(cfg)

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
        if self.cfg.curriculum:
            self.set_min_max_dist(0.1, 0.5)
            self.set_min_max_dist(0.1, 0.5, eval=True)
        self.hz = 240
        p.setTimeStep(1./self.hz)
        self.daisy = self.env.robots[0]
        self.daisy.set_position([3.0, 3.0, 0.3])
        self.daisy.set_orientation([0, 0, 0, 1.])
        self.daisy_state = motion_library.exp_standing(self.daisy, shoulder=1.2, elbow=0.3)
        self.init_state = self.daisy.calc_state()
        self.behavior = BehaviorParameters()

    def set_min_max_dist(self, min, max, eval=False):
        if eval:
            self.eval_env.target_dist_min = min
            self.eval_env.target_dist_max = max
        else:
            self.env.target_dist_min = min
            self.env.target_dist_max = max

    def evaluate(self):
        episode_rewards, dist_to_goals, episode_dists, successes, spls, episode_lengths, collision_steps, path_lengths = [], [], [], [], [], [], [], []
        for episode in range(self.cfg.num_eval_episodes):
            self.set_min_max_dist(1, 10, eval=True)
            obs = self.eval_env.reset()
            obs = obs["sensor"][:2]
            self.daisy_state = motion_library.exp_standing(self.daisy, shoulder=1.2, elbow=0.3)
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            initial_pos = self.eval_env.initial_pos
            target_pos = self.eval_env.target_pos
            episode_dist = l2_distance(initial_pos, target_pos)
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                self.behavior.target_speed = np.array([action[0], action[1]])
                raibert_controller = DaisyRaibertController(init_state=self.init_state, behavior_parameters=self.behavior)
                time_per_step = 2*self.behavior.stance_duration
                for i in range(time_per_step):
                    raibert_action = raibert_controller.get_action(self.init_state, i+1)
                    obs, reward, done, info = self.eval_env.step(raibert_action, low_level=True)
                    obs = obs["sensor"][:2]
                    self.init_state = self.daisy.calc_state()
                self.video_recorder.record(self.eval_env, self.cfg.record_params)
                self.eval_env.current_step +=1
                episode_reward += reward
            print('Evaluation. INITIAL POS', initial_pos, ' TARGET POS: ', target_pos, ' EPISODE DIST: ', episode_dist, ' SPL: ', info["spl"], 'EPISODE: ', episode, 'STEP: ', self.step, ' MIN: ', self.eval_env.target_dist_min, ' MAX: ', self.eval_env.target_dist_max)
            self.video_recorder.save(f'{self.step}_{episode}_{episode_dist}_{info["spl"]}.mp4')
            episode_rewards.append(episode_reward)
            dist_to_goals.append(info['dist_to_goal'])
            episode_dists.append(episode_dist)
            successes.append(info['success'])
            spls.append(info['spl'])
            episode_lengths.append(info['episode_length'])
            collision_steps.append(info['collision_step'])
            path_lengths.append(info['path_length'])
        self.logger.log('eval/episode_reward', np.mean(np.asarray(episode_rewards)), self.step)
        self.logger.log('eval/min_dist', self.eval_env.target_dist_min, self.step)
        self.logger.log('eval/max_dist', self.eval_env.target_dist_max, self.step)
        self.logger.log('eval/dist_to_goal', np.mean(np.asarray(dist_to_goals)), self.step)
        self.logger.log('eval/episode_dist', np.mean(np.asarray(episode_dists)), self.step)
        self.logger.log('eval/success', np.mean(np.asarray(successes)), self.step)
        self.logger.log('eval/spl', np.mean(np.asarray(spls)), self.step)
        self.logger.log('eval/num_steps', np.mean(np.asarray(episode_lengths)), self.step)
        self.logger.log('eval/num_collisions', np.mean(np.asarray(collision_steps)), self.step)
        self.logger.log('eval/path_length', np.mean(np.asarray(path_lengths)), self.step)
        self.logger.dump(self.step, ty='eval')
        print('Evaluation. Avg Eval Success: ', np.mean(np.asarray(successes)), 'Step: ', self.step, ' Min: ', self.eval_env.target_dist_min, ' Max: ', self.eval_env.target_dist_max)

        if self.cfg.curriculum:
            self.set_min_max_dist(self.env.target_dist_min, self.env.target_dist_max, eval=True)
            curriculum_successes = []
            for episode in range(self.cfg.num_curriculum_eval_episodes):
                obs = self.eval_env.reset()
                obs = obs["sensor"][:2]
                self.daisy_state = motion_library.exp_standing(self.daisy, shoulder=1.2, elbow=0.3)
                done = False
                episode_reward = 0
                initial_pos = self.eval_env.initial_pos
                target_pos = self.eval_env.target_pos
                episode_dist = l2_distance(initial_pos, target_pos)
                while not done:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=False)
                    self.behavior.target_speed = np.array([action[0], action[1]])
                    raibert_controller = DaisyRaibertController(init_state=self.init_state, behavior_parameters=self.behavior)
                    time_per_step = 2*self.behavior.stance_duration
                    for i in range(time_per_step):
                        raibert_action = raibert_controller.get_action(self.init_state, i+1)
                        obs, reward, done, info = self.eval_env.step(raibert_action, low_level=True)
                        obs = obs["sensor"][:2]
                        self.init_state = self.daisy.calc_state()
                    self.eval_env.current_step +=1
                    episode_reward += reward
                print('Curriculum eval. INITIAL POS', initial_pos, ' TARGET POS: ', target_pos, ' EPISODE DIST: ', episode_dist, ' SPL: ', info["spl"], ' MIN: ', self.eval_env.target_dist_min, ' MAX: ', self.eval_env.target_dist_max)
                curriculum_successes.append(info['success'])
            if np.mean(np.asarray(curriculum_successes)) > 0.5:
                if self.env.target_dist_max < 10:
                    self.env.target_dist_min +=0.5
                    self.env.target_dist_max +=0.5
                else:
                    self.env.target_dist_min = 1
                    self.env.target_dist_max = 10
                self.set_min_max_dist(self.env.target_dist_min, self.env.target_dist_max)
            else:
                self.set_min_max_dist(self.env.target_dist_min, self.env.target_dist_max)
            print('Curriculum curr min, max: ', self.eval_env.target_dist_min, self.eval_env.target_dist_max, 'avg curriculum success: ', np.mean(np.asarray(curriculum_successes)), 'step: ', self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            # evaluate agent periodically
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()
                if self.cfg.save_model:
                    self.agent.save(self.model_dir, self.step)
                if self.cfg.save_buffer:
                    self.replay_buffer.save(self.buffer_dir)

            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    self.logger.log('train/min_dist', self.env.target_dist_min, self.step)
                    self.logger.log('train/max_dist', self.env.target_dist_max, self.step)
                    self.logger.log('train/episode_reward', episode_reward,
                                    self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps), ty='train')
                    start_time = time.time()

                obs = self.env.reset()
                obs = obs["sensor"][:2]
                self.agent.reset()
                self.daisy_state = motion_library.exp_standing(self.daisy, shoulder=1.2, elbow=0.3)

                initial_pos = self.env.initial_pos
                target_pos = self.env.target_pos
                episode_dist = l2_distance(initial_pos, target_pos)
                print('Training. INITIAL POS', initial_pos, ' TARGET POS: ', target_pos, ' EPISODE DIST: ', episode_dist, 'STEP: ', self.step, ' MIN: ', self.env.target_dist_min, ' MAX: ', self.env.target_dist_max)

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

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
