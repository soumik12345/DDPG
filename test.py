import gym
import json
from src.utils import *
from src.agents import *


class Tester:

    def __init__(self, config_file, model_file='./models/ddpg_bipedalwalker_v3_0'):
        self.config = Tester.parse_config(config_file)
        self.env = gym.make(self.config['env_name'])
        self.state_dimension = self.env.observation_space.shape[0]
        self.action_dimension = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.device = torch.device('cpu')
        self.agent = DDPGAgent(
            state_dim=self.state_dimension, action_dim=self.action_dimension,
            max_action=self.max_action, device=self.device,
            discount=self.config['discount'], tau=self.config['tau']
        )
        self.agent.load_checkpoint(model_file)

    @staticmethod
    def parse_config(json_file):
        with open(json_file, 'r') as f:
            configs = json.load(f)
        return configs

    def test(self, render=True):
        self.mean_rewards = evaluate_policy(
            self.agent, self.config['env_name'],
            self.config['seed'], eval_episodes=10, render=render
        )
        print(self.mean_rewards)


tester = Tester('./configs/BipedalWalker-v3.json')
tester.test()
