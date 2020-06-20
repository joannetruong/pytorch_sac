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
        print("obs space: ", self.env.observation_space)
        print("obs space shape: ", self.env.observation_space.shape)
        if cfg.encoder_type == 'pixel':
            self.env = utils.FrameStackDepth(self.env, k=cfg.frame_stack)
        print("obs space: ", self.env.observation_space)

        cfg.agent.params.obs_dim = self.env.observation_space["sensor"].shape[0]
        cfg.agent.params.obs_shape = self.env.observation_space["depth"].shape
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        if self.cfg.pretrained:
            self.agent.load("/private/home/jtruong/repos/pytorch_sac_2/pytorch_sac/exp/2020.06.15/multi_goal_stadium_depth/model", "98900")
            print('loading pretrained weights')
        self.replay_buffer = ReplayBuffer(self.env.observation_space["sensor"].shape,
                                          self.env.observation_space["depth"].shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        if self.cfg.curriculum:
            self.env.set_min_max_dist(0.1, 0.5)

    def evaluate(self):
        episode_rewards, dist_to_goals, episode_dists, successes, spls, episode_lengths, collision_steps, path_lengths = [], [], [], [], [], [], [], []
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset(eval=True)
#            obs = obs["sensor"][:2]
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            initial_pos = self.env.get_initial_pos()
            target_pos = self.env.get_target_pos()
            episode_dist = l2_distance(initial_pos, target_pos)
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env, self.cfg.record_params)
                episode_reward += reward
            print('Evaluation. INITIAL POS', initial_pos, ' TARGET POS: ', target_pos, ' EPISODE DIST: ', episode_dist, ' SPL: ', info["spl"], 'EPISODE: ', episode, 'STEP: ', self.step)
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
        self.logger.log('eval/min_dist', self.env.target_dist_min, self.step)
        self.logger.log('eval/max_dist', self.env.target_dist_max, self.step)
        self.logger.log('eval/dist_to_goal', np.mean(np.asarray(dist_to_goals)), self.step)
        self.logger.log('eval/episode_dist', np.mean(np.asarray(episode_dists)), self.step)
        self.logger.log('eval/success', np.mean(np.asarray(successes)), self.step)
        self.logger.log('eval/spl', np.mean(np.asarray(spls)), self.step)
        self.logger.log('eval/num_steps', np.mean(np.asarray(episode_lengths)), self.step)
        self.logger.log('eval/num_collisions', np.mean(np.asarray(collision_steps)), self.step)
        self.logger.log('eval/path_length', np.mean(np.asarray(path_lengths)), self.step)
        self.logger.dump(self.step)

        if self.cfg.curriculum:
            curriculum_successes = []
            for episode in range(self.cfg.num_curriculum_eval_episodes):
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                initial_pos = self.env.get_initial_pos()
                target_pos = self.env.get_target_pos()
                episode_dist = l2_distance(initial_pos, target_pos)
                while not done:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=False)
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward
                print('Curriculum eval. INITIAL POS', initial_pos, ' TARGET POS: ', target_pos, ' EPISODE DIST: ', episode_dist, ' SPL: ', info["spl"])
                curriculum_successes.append(info['success'])
            if np.mean(np.asarray(curriculum_successes)) > 0.5:
                print('prev min, max: ', self.env.target_dist_min, self.env.target_dist_max)
                if self.env.target_dist_max < 10:
                    self.env.target_dist_min +=0.5
                    self.env.target_dist_max +=0.5
                else:
                    self.env.target_dist_min = 1
                    self.env.target_dist_max = 10
                self.env.set_min_max_dist(self.env.target_dist_min, self.env.target_dist_max)
            else:
                self.env.set_min_max_dist(self.env.target_dist_min, self.env.target_dist_max)
            print('curr min, max: ', self.env.target_dist_min, self.env.target_dist_max, 'avg curriculum success: ', np.mean(np.asarray(curriculum_successes)), 'step: ', self.step)
        print('curr min, max: ', self.env.target_dist_min, self.env.target_dist_max, 'avg success: ', np.mean(np.asarray(successes)), 'step: ', self.step)

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

                obs = self.env.reset()
 #               obs = obs["sensor"][:2]
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                self.logger.log('train/episode_reward', episode_reward,
                                    self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            next_obs, reward, done, info = self.env.step(action)
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env.max_step else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)
            # evaluate agent periodically
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()
                if self.cfg.save_model:
                    self.agent.save(self.model_dir, self.step)
                if self.cfg.save_buffer:
                    self.replay_buffer.save(self.buffer_dir)

            obs = next_obs
            episode_step += 1
            self.step += 1



@hydra.main(config_path="config/train_locobot.yaml", strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
