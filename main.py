from test import Tester

# tester = Tester(
#     './configs/BipedalWalker-v3.json',
#     './pretrained_models/bipedal_walker_v3/ddpg_bipedalwalker_v3_0'
# )

# tester = Tester(
#     './configs/LunarLanderContinuous-v2.json',
#     './pretrained_models/lunar_lander_continuous_v2/ddpg_lunarlander_v2_0'
# )

tester = Tester(
    './configs/BipedalWalkerHardcore-v3.json',
    './pretrained_models/bipedal_walker_hardcore_v3/ddpg_bipedalwalker_hardcore_v3_0'
)

print('Mean Reward:', tester.test(eval_episodes=20, render=True))
