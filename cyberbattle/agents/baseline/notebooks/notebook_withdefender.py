# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Attacker agent benchmark comparison in presence of a basic defender

This notebooks can be run directly from VSCode, to generate a
traditional Jupyter Notebook to open in your browser
 you can run the VSCode command `Export Currenty Python File As Jupyter Notebook`.
"""


import sys
import logging
import gym
import importlib
import sys
sys.path.append('/home/windy/Desktop/experiment/CyberBattleSim')
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_randomcredlookup as rca
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
from cyberbattle._env.defender import ScanAndReimageCompromisedMachines
from cyberbattle._env.cyberbattle_env import AttackerGoal, DefenderConstraint
from cyberbattle._env.defender1 import LearningDefender1


importlib.reload(learner)
importlib.reload(p)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")


###1.环境初始化###
#无防御者环境
cyberbattlectf= gym.make('CyberBattleToyCtf-v0',
                                     attacker_goal=AttackerGoal(
                                         own_atleast=0,
                                         own_atleast_percent=1.0
                                     )
                                     )
#PPO防御者环境
cyberbattlectf_defender = gym.make('CyberBattleToyCtf-v0',
                                     attacker_goal=AttackerGoal(
                                         own_atleast=0,
                                         own_atleast_percent=1.0
                                     ),defender_constraint=DefenderConstraint(maintain_sla=0.80),defender_agent=LearningDefender1()
                                     )
#静态防御者环境（CyberBattleSim自带）
cyberbattlectf_static_defender = gym.make('CyberBattleToyCtf-v0',
                                     attacker_goal=AttackerGoal(
                                         own_atleast=0,
                                         own_atleast_percent=1.0
                                     ),defender_constraint=DefenderConstraint(
                                                              maintain_sla=0.80
                                                          ),
                                                          defender_agent=ScanAndReimageCompromisedMachines(
                                                              probability=0.6,
                                                              scan_capacity=2,
                                                              scan_frequency=5)
                                                              )
                                   
ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
        maximum_node_count=22,
    identifiers=cyberbattlectf.identifiers
)

###2.攻击者训练阶段###

#用无defender训练攻击者，得到DQL-Attcker
'''
iteration_count = 9000
training_episode_count = 50
dqn_with_no_defender = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlectf, # type: ignore
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.15,
        replay_memory_size=10000,
        target_update=5,
        batch_size=256,
        learning_rate=0.01),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.9,#0.9
    render=False,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="DQL",
    att_file="attack_dql_model_9000*50"
)

###3.防御者训练阶段###

#用DQL-Attcker训练ppo defender，得到PPO-Defender
learner.train_defender(att_filename='attack_dql_model_9000*50_model.pth',
                       cyberbattle_gym_env=cyberbattlectf_defender, # type: ignore
                       environment_properties=ep,
                       learner=dqla.DeepQLearnerPolicy(ep=ep,
                        gamma=0.15,
                        replay_memory_size=10000,
                        target_update=5,
                        batch_size=256,
                        learning_rate=0.01),
                        episode_count=800,iteration_count=2048,def_file='ppo_defender_800.zip',def_graph='ppo_defender_300.png')

###4.防御者性能评估阶段###
'''
#DQL-Attcker vs PPO-Defender
learner.eval(att_filename='attack_dql_model_9000*50_model.pth',def_filename='ppo_defender_800.zip',learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.15,
        replay_memory_size=10000,
        target_update=5,
        batch_size=256,
        learning_rate=0.01),
        cyberbattle_gym_env=cyberbattlectf_defender,# type: ignore
        environment_properties=ep,title="evaluate" 
        )
#DQL-Attcker vs Static-Defender（Cyberbattlesim 自带）
learner.eval(att_filename='attack_dql_model_9000*50_model.pth',def_filename='ppo_defender_300.zip',learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.15,
        replay_memory_size=10000,
        target_update=5,
        batch_size=256,
        learning_rate=0.01),
        cyberbattle_gym_env=cyberbattlectf_static_defender,# type: ignore
        environment_properties=ep,title="evaluate" 
)
#DQL-Attcker vs NO-Defender
learner.eval(att_filename='attack_dql_model_9000*50_model.pth',def_filename='ppo_defender_300.zip',learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.15,
        replay_memory_size=10000,
        target_update=5,
        batch_size=256,
        learning_rate=0.01),
        cyberbattle_gym_env=cyberbattlectf,# type: ignore
        environment_properties=ep,title="evaluate" 
)