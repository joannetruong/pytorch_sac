import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils
from encoder import make_encoder
from decoder import make_decoder



class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, obs_shape, action_dim, hidden_dim, hidden_depth, 
        encoder_type, encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.tgt_embeding = nn.Linear(obs_dim+1, 32)
        self.ln = nn.LayerNorm(32)
        dim = self.encoder.feature_dim + 32
        self.Q1 = utils.mlp(dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        obs_rgb = self.encoder(obs["rgb"], detach=detach_encoder)
        sensor = torch.stack([obs["sensor"][:, 0], torch.cos(obs["sensor"][:, 1]), torch.sin(obs["sensor"][:, 1])], -1)
        embed_sensor = self.tgt_embeding(sensor)
        embed_sensor_norm = self.ln(embed_sensor)
        obs_sensor = torch.tanh(embed_sensor_norm)
        obs = torch.cat((obs_rgb, obs_sensor), 1)
#        obs = torch.cat((obs_rgb, embed_sensor), 1)
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
