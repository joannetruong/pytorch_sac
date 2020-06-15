import numpy as np
import torch
import os
from collections import OrderedDict

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_sensor_shape, obs_img_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
#        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = OrderedDict()
        self.obses["sensor"] = np.empty((capacity, *obs_sensor_shape), dtype=np.float32)
#        self.obses["rgb"] = np.empty((capacity, *obs_img_shape), dtype=np.uint8)
        self.obses["depth"] = np.empty((capacity, *obs_img_shape), dtype=np.uint8)
        self.next_obses = OrderedDict()
        self.next_obses["sensor"] = np.empty((capacity, *obs_sensor_shape), dtype=np.float32)
#        self.next_obses["rgb"] = np.empty((capacity, *obs_img_shape), dtype=np.uint8)
        self.next_obses["depth"] = np.empty((capacity, *obs_img_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        if torch.is_tensor(obs["sensor"][:2]):
            obs_sensor = obs["sensor"][:2].detach().cpu().numpy()
        else:
            obs_sensor = obs["sensor"][:2]
#        if torch.is_tensor(obs["rgb"]):
#            obs_rgb = obs["rgb"].detach().cpu().numpy().astype(np.uint8)
#        else:
#            obs_rgb = obs["rgb"]
        if torch.is_tensor(obs["depth"]):
            obs_depth = obs["depth"].detach().cpu().numpy().astype(np.uint8)
        else:
            obs_depth = obs["depth"]
        if torch.is_tensor(next_obs["sensor"][:2]):
            next_obs_sensor = next_obs["sensor"][:2].detach().cpu().numpy()
        else:
            next_obs_sensor = next_obs["sensor"][:2]
#        if torch.is_tensor(next_obs["rgb"]):
#            next_obs_rgb = next_obs["rgb"].detach().cpu().numpy().astype(np.uint8)
#        else:
#            next_obs_rgb = next_obs["rgb"]
        if torch.is_tensor(next_obs["depth"]):
            next_obs_depth = next_obs["depth"].detach().cpu().numpy().astype(np.uint8)
        else:
            next_obs_depth = next_obs["depth"]
        np.copyto(self.obses["sensor"][self.idx], obs_sensor)
#        np.copyto(self.obses["rgb"][self.idx], obs_rgb)
        np.copyto(self.obses["depth"][self.idx], obs_depth)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses["sensor"][self.idx], next_obs_sensor)
#        np.copyto(self.next_obses["rgb"][self.idx], next_obs_rgb)
        np.copyto(self.next_obses["depth"][self.idx], next_obs_depth)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        obses = OrderedDict()
        obses["sensor"] = torch.as_tensor(self.obses["sensor"][idxs], device=self.device).float()
#        obses["rgb"] = torch.as_tensor(self.obses["rgb"][idxs], device=self.device).float()
        obses["depth"] = torch.as_tensor(self.obses["depth"][idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = OrderedDict()
        next_obses["sensor"] = torch.as_tensor(self.next_obses["sensor"][idxs],
                                     device=self.device).float()
#        next_obses["rgb"] = torch.as_tensor(self.next_obses["rgb"][idxs],
#                                     device=self.device).float()
        next_obses["depth"] = torch.as_tensor(self.next_obses["depth"][idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses["sensor"][self.last_save:self.idx],
#            self.obses["rgb"][self.last_save:self.idx],
            self.obses["depth"][self.last_save:self.idx],
            self.next_obses["sensor"][self.last_save:self.idx],
#            self.next_obses["rgb"][self.last_save:self.idx],
            self.next_obses["depth"][self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end
