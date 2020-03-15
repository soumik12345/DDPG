import gym
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils import *
from src.memory import *
from src.agents import *
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, config_file):
        self.config = Trainer.parse_config(config_file)
        self.env = gym.make(self.config['env_name'])
        self.apply_seed()
        self.state_dimension = self.env.observation_space.shape[0]
        self.action_dimension = self.env.action_space.shape[0]
        self.max_action = self.float(self.env.action_space.high[0])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent = DDPGAgent(
            state_dim=self.state_dimension, action_dim=self.action_dimension,
            max_action=self.max_action, device=self.device,
            discount=self.config['discount'], tau=self.config['tau']
        )
        self.save_file_name = f"DDPG_{self.config['env_name']}_{self.config['seed']}"
        self.memory = ReplayBuffer(self.state_dimension, self.action_dimension)
        self.writer = SummaryWriter('./logs/' + self.config['env_name'] + '/')

    @staticmethod
    def parse_config(json_file):
        with open(json_file, 'r') as f:
            configs = json.load(f)
        return configs

    def apply_seed(self):
        self.env.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def train(self):
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        evaluations = []
        for ts in tqdm(range(1, self.config['time_steps'] + 1)):
            self.episode_timesteps += 1
            if ts < self.config['start_time_step']:
                action = self.env.action_space.sample()
            else:
                action = (
                        self.agent.select_action(np.array(state))+ np.random.normal(
                            0, self.max_action * self.config['expl_noise'],
                            size=self.action_dimension
                        )
                ).clip(
                    -self.max_action,
                    self.max_action
                )
            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(
                state, action, next_state, reward,
                float(done) if episode_timesteps < self.env._max_episode_steps else 0)
            state = next_state
            episode_reward += reward
            if ts >= self.config['start_time_step']:
                self.agent.train(self.memory, self.config['batch_size'])
            if done:
                self.writer.add_scalar('Episode Reward', episode_reward, ts)
                state = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
        if (ts + 1) % self.config['evaluate_frequency'] == 0:
            evaluations.append(evaluate_policy(self.agent, self.config['env_name'], self.config['seed']))
            self.agent.save(f"./models/{self.save_file_name}")
