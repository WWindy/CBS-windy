# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Agent wrapper for CyberBattle envrionments exposing additional
features extracted from the environment observations"""

from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import enum
import numpy as np
import gym
from gym import spaces, Wrapper
from gym.spaces.space import Space
from numpy import ndarray
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import logging
import boolean
from cyberbattle.simulation import commandcontrol, model
from plotly.missing_ipywidgets import FigureWidget
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cyberbattle.agents.baseline.environment_event_source import IEnvironmentObserver, EnvironmentEventSource
from cyberbattle._env.defender1 import LearningDefender1

from cyberbattle.agents.baseline.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from cyberbattle.agents.baseline.agent_wrapper_defender import DefenderEnvWrapper

from stable_baselines3 import PPO


class StateAugmentation:
    """Default agent state augmentation, consisting of the gym environment
    observation itself and nothing more."""

    def __init__(self, observation: cyberbattle_env.Observation):
        self.observation = observation

    def on_step(self, action: cyberbattle_env.Action, reward: float, done: bool, observation: cyberbattle_env.Observation):
        self.observation = observation

    def on_reset(self, observation: cyberbattle_env.Observation):
        self.observation = observation


class Feature(spaces.MultiDiscrete):
    """
    Feature consisting of multiple discrete dimensions.
    Parameters:
        nvec: is a vector defining the number of possible values
        for each discrete space.
    """

    def __init__(self, env_properties: EnvironmentBounds, nvec):
        self.env_properties = env_properties
        super().__init__(nvec)

    def flat_size(self):
        return np.prod(self.nvec)

    def name(self):
        """Return the name of the feature"""
        p = len(type(Feature(self.env_properties, [])).__name__) + 1
        return type(self).__name__[p:]

    def get(self, a: StateAugmentation, node: Optional[int]) -> np.ndarray:
        """Compute the current value of a feature value at
        the current observation and specific node"""
        raise NotImplementedError

    def pretty_print(self, v):
        return v


class Feature_active_node_properties(Feature):
    """Bitmask of all properties set for the active node"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.property_count)

    def get(self, a: StateAugmentation, node) -> ndarray:
        assert node is not None, 'feature only valid in the context of a node'

        node_prop = a.observation['discovered_nodes_properties']

        # list of all properties set/unset on the node
        # Remap to get rid of unknown value 0: 1 -> 1, and -1 -> 0 (and 0-> 0)
        assert node < len(node_prop), f'invalid node index {node} (not discovered yet)'
        remapped = np.array((1 + node_prop[node]) / 2, dtype=np.int)
        return remapped


class Feature_active_node_age(Feature):
    """How recently was this node discovered?
    (measured by reverse position in the list of discovered nodes)"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count])

    def get(self, a: StateAugmentation, node) -> ndarray:
        assert node is not None, 'feature only valid in the context of a node'

        discovered_node_count = len(a.observation['discovered_nodes_properties'])

        assert node < discovered_node_count, f'invalid node index {node} (not discovered yet)'

        return np.array([discovered_node_count - node - 1], dtype=np.int)


class Feature_active_node_id(Feature):
    """Return the node id itself"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count] * 1)

    def get(self, a: StateAugmentation, node) -> ndarray:
        return np.array([node], dtype=np.int)


class Feature_discovered_nodeproperties_sliding(Feature):
    """Bitmask indicating node properties seen in last few cache entries"""
    window_size = 3

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.property_count)

    def get(self, a: StateAugmentation, node) -> ndarray:
        node_prop = np.array(a.observation['discovered_nodes_properties'])

        # keep last window of entries
        node_prop_window = node_prop[-self.window_size:, :]

        # Remap to get rid of unknown value 0: 1 -> 1, and -1 -> 0 (and 0-> 0)
        node_prop_window_remapped = np.int32((1 + node_prop_window) / 2)

        countby = np.sum(node_prop_window_remapped, axis=0)

        bitmask = (countby > 0) * 1
        return bitmask


class Feature_discovered_ports(Feature):
    """Bitmask vector indicating each port seen so far in discovered credentials"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.port_count)

    def get(self, a: StateAugmentation, node):
        ccm = a.observation['credential_cache_matrix']
        known_credports = np.zeros(self.env_properties.port_count, dtype=np.int32)
        known_credports[np.int32(ccm[:, 1])] = 1
        return known_credports


class Feature_discovered_ports_sliding(Feature):
    """Bitmask indicating port seen in last few cache entries"""
    window_size = 3

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.port_count)

    def get(self, a: StateAugmentation, node):
        ccm = a.observation['credential_cache_matrix']
        known_credports = np.zeros(self.env_properties.port_count, dtype=np.int32)
        known_credports[np.int32(ccm[-self.window_size:, 1])] = 1
        return known_credports


class Feature_discovered_ports_counts(Feature):
    """Count of each port seen so far in discovered credentials"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1] * p.port_count)

    def get(self, a: StateAugmentation, node):
        ccm = a.observation['credential_cache_matrix']
        return np.bincount(np.int32(ccm[:, 1]), minlength=self.env_properties.port_count)


class Feature_discovered_credential_count(Feature):
    """number of credentials discovered so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1])

    def get(self, a: StateAugmentation, node):
        return [len(a.observation['credential_cache_matrix'])]


class Feature_discovered_node_count(Feature):
    """number of nodes discovered so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        return [len(a.observation['discovered_nodes_properties'])]


class Feature_discovered_notowned_node_count(Feature):
    """number of nodes discovered that are not owned yet (optionally clipped)"""

    def __init__(self, p: EnvironmentBounds, clip: Optional[int]):
        self.clip = p.maximum_node_count if clip is None else clip
        super().__init__(p, [self.clip + 1])

    def get(self, a: StateAugmentation, node):
        node_props = a.observation['discovered_nodes_properties']
        discovered = len(node_props)
        # here we assume that a node is owned just if all its properties are known
        owned = np.count_nonzero(np.all(node_props != 0, axis=1))
        diff = discovered - owned
        return [min(diff, self.clip)]


class Feature_owned_node_count(Feature):
    """number of owned nodes so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        levels = a.observation['nodes_privilegelevel']
        owned_nodes_indices = np.where(levels > 0)[0]
        return [len(owned_nodes_indices)]


class ConcatFeatures(Feature):
    """ Concatenate a list of features into a single feature
    Parameters:
        feature_selection - a selection of features to combine
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature]):
        self.feature_selection = feature_selection
        self.dim_sizes = np.concatenate([f.nvec for f in feature_selection])
        super().__init__(p, [self.dim_sizes])

    def pretty_print(self, v):
        return v

    def get(self, a: StateAugmentation, node=None) -> np.ndarray:
        """Return the feature vector"""
        feature_vector = [f.get(a, node) for f in self.feature_selection]
        return np.concatenate(feature_vector)


class FeatureEncoder(Feature):
    """ Encode a list of featues as a unique index
    """

    feature_selection: List[Feature]

    def vector_to_index(self, feature_vector: np.ndarray) -> int:
        raise NotImplementedError

    def feature_vector_of_observation_at(self, a: StateAugmentation, node: Optional[int]) -> np.ndarray:
        """Return the current feature vector"""
        feature_vector = [f.get(a, node) for f in self.feature_selection]
        # print(f'feature_vector={feature_vector}  self.feature_selection={self.feature_selection}')
        return np.concatenate(feature_vector)

    def feature_vector_of_observation(self, a: StateAugmentation):
        return self.feature_vector_of_observation_at(a, None)

    def encode(self, a: StateAugmentation, node=None) -> int:
        """Return the index encoding of the feature"""
        feature_vector_concat = self.feature_vector_of_observation_at(a, node)
        return self.vector_to_index(feature_vector_concat)

    def encode_at(self, a: StateAugmentation, node) -> int:
        """Return the current feature vector encoding with a node context"""
        feature_vector_concat = self.feature_vector_of_observation_at(a, node)
        return self.vector_to_index(feature_vector_concat)

    def get(self, a: StateAugmentation, node=None) -> np.ndarray:
        """Return the feature vector"""
        return np.array([self.encode(a, node)])

    def name(self):
        """Return a name for the feature encoding"""
        n = ', '.join([f.name() for f in self.feature_selection])
        return f'[{n}]'


class HashEncoding(FeatureEncoder):
    """ Feature defined as a hash of another feature
    Parameters:
       feature_selection: a selection of features to combine
       hash_dim: dimension after hashing with hash(str(feature_vector)) or -1 for no hashing
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature], hash_size: int):
        self.feature_selection = feature_selection
        self.hash_size = hash_size
        super().__init__(p, [hash_size])

    def flat_size(self):
        return self.hash_size

    def vector_to_index(self, feature_vector) -> int:
        """Hash the state vector"""
        return hash(str(feature_vector)) % self.hash_size

    def pretty_print(self, index):
        return f'#{index}'


class RavelEncoding(FeatureEncoder):
    """ Combine a set of features into a single feature with a unique index
     (calculated by raveling the original indices)
    Parameters:
        feature_selection - a selection of features to combine
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature]):
        self.feature_selection = feature_selection
        self.dim_sizes = np.concatenate([f.nvec for f in feature_selection])
        self.ravelled_size: int = np.prod(self.dim_sizes)
        assert np.shape(self.ravelled_size) == (), f'! {np.shape(self.ravelled_size)}'
        super().__init__(p, [self.ravelled_size])

    def vector_to_index(self, feature_vector):
        assert len(self.dim_sizes) == len(feature_vector), \
            f'feature vector of size {len(feature_vector)}, ' \
            f'expecting {len(self.dim_sizes)}: {feature_vector} -- {self.dim_sizes}'
        index: np.int32 = np.ravel_multi_index(feature_vector, self.dim_sizes)
        assert index < self.ravelled_size, \
            f'feature vector out of bound ({feature_vector}, dim={self.dim_sizes}) ' \
            f'-> index={index}, max_index={self.ravelled_size-1})'
        return index

    def unravel_index(self, index) -> np.ndarray:
        return np.unravel_index(index, self.dim_sizes)

    def pretty_print(self, index):
        return self.unravel_index(index)


def owned_nodes(observation):
    """Return the list of owned nodes"""
    return np.nonzero(observation['nodes_privilegelevel'])[0]


def discovered_nodes_notowned(observation):
    """Return the list of discovered nodes that are not owned yet"""
    return np.nonzero(observation['nodes_privilegelevel'] == 0)[0]


class AbstractAction(Feature):
    """An abstraction of the gym state space that reduces
    the space dimension for learning use to just
        - local_attack(vulnid)    (source_node provided)
        - remote_attack(vulnid)   (source_node provided, target_node forgotten)
        - connect(port)           (source_node provided, target_node forgotten, credentials infered from cache)
    """

    def __init__(self, p: EnvironmentBounds):
        self.n_local_actions = p.local_attacks_count
        self.n_remote_actions = p.remote_attacks_count
        self.n_connect_actions = p.port_count
        self.n_actions = self.n_local_actions + self.n_remote_actions + self.n_connect_actions
        super().__init__(p, [self.n_actions])

    def specialize_to_gymaction(self, source_node: np.int32, observation, abstract_action_index: np.int32
                                ) -> Optional[cyberbattle_env.Action]:
        """Specialize an abstract "q"-action into a gym action.
        Return an adjustement weight (1.0 if the choice was deterministic, 1/n if a choice was made out of n)
        and the gym action"""

        abstract_action_index_int = int(abstract_action_index)

        node_prop = np.array(observation['discovered_nodes_properties'])

        if abstract_action_index_int < self.n_local_actions:
            vuln = abstract_action_index_int
            return {'local_vulnerability': np.array([source_node, vuln])}

        abstract_action_index_int -= self.n_local_actions
        if abstract_action_index_int < self.n_remote_actions:
            vuln = abstract_action_index_int

            discovered_nodes_count = len(node_prop)

            if discovered_nodes_count <= 1:
                return None

            # NOTE: We can do better here than random pick: ultimately this
            # should be learnt from target node properties

            # pick any node from the discovered ones
            # excluding the source node itself
            target = (source_node + 1 + np.random.choice(discovered_nodes_count - 1)) % discovered_nodes_count

            return {'remote_vulnerability': np.array([source_node, target, vuln])}

        abstract_action_index_int -= self.n_remote_actions
        port = np.int32(abstract_action_index_int)

        discovered_credentials = np.array(observation['credential_cache_matrix'])
        n_discovered_creds = len(discovered_credentials)
        if n_discovered_creds <= 0:
            # no credential available in the cache: cannot poduce a valid connect action
            return None

        nodes_not_owned = discovered_nodes_notowned(observation)

        # Pick a matching cred from the discovered_cred matrix
        # (at random if more than one exist for this target port)
        match_port = discovered_credentials[:, 1] == port
        match_port_indices = np.where(match_port)[0]

        credential_indices_choices = [c for c in match_port_indices
                                      if discovered_credentials[c, 0] in nodes_not_owned]

        if credential_indices_choices:
            logging.debug('found matching cred in the credential cache')
        else:
            logging.debug('no cred matching requested port, trying instead creds used to access other ports')
            credential_indices_choices = [i for (i, n) in enumerate(discovered_credentials[:, 0])
                                          if n in nodes_not_owned]

            if credential_indices_choices:
                logging.debug('found cred in the credential cache without matching port name')
            else:
                logging.debug('no cred to use from the credential cache')
                return None

        cred = np.int32(np.random.choice(credential_indices_choices))
        target = np.int32(discovered_credentials[cred, 0])
        return {'connect': np.array([source_node, target, port, cred], dtype=np.int32)}

    def abstract_from_gymaction(self, gym_action: cyberbattle_env.Action) -> np.int32:
        """Abstract a gym action into an action to be index in the Q-matrix"""
        if 'local_vulnerability' in gym_action:
            return gym_action['local_vulnerability'][1]
        elif 'remote_vulnerability' in gym_action:
            r = gym_action['remote_vulnerability']
            return self.n_local_actions + r[2]

        assert 'connect' in gym_action
        c = gym_action['connect']

        a = self.n_local_actions + self.n_remote_actions + c[2]
        assert a < self.n_actions
        return np.int32(a)


class ActionTrackingStateAugmentation(StateAugmentation):
    """An agent state augmentation consisting of
    the environment observation augmented with the following dynamic information:
       - success_action_count: count of action taken and succeeded at the current node
       - failed_action_count: count of action taken and failed at the current node
     """

    def __init__(self, p: EnvironmentBounds, observation: cyberbattle_env.Observation):
        self.aa = AbstractAction(p)
        self.success_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.failed_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.env_properties = p
        super().__init__(observation)

    def on_step(self, action: cyberbattle_env.Action, reward: float, done: bool, observation: cyberbattle_env.Observation):
        node = cyberbattle_env.sourcenode_of_action(action)
        abstract_action = self.aa.abstract_from_gymaction(action)
        if reward > 0:
            self.success_action_count[node, abstract_action] += 1
        else:
            self.failed_action_count[node, abstract_action] += 1
        super().on_step(action, reward, done, observation)

    def on_reset(self, observation: cyberbattle_env.Observation):
        p = self.env_properties
        self.success_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.failed_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        super().on_reset(observation)


class Feature_actions_tried_at_node(Feature):
    """A bit mask indicating which actions were already tried
    a the current node: 0 no tried, 1 tried"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return ((a.failed_action_count[node, :] + a.success_action_count[node, :]) != 0) * 1


class Feature_success_actions_at_node(Feature):
    """number of time each action succeeded at a given node"""

    max_action_count = 100

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return np.minimum(a.success_action_count[node, :], self.max_action_count - 1)


class Feature_failed_actions_at_node(Feature):
    """number of time each action failed at a given node"""

    max_action_count = 100

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return np.minimum(a.failed_action_count[node, :], self.max_action_count - 1)


class Verbosity(enum.Enum):
    """Verbosity of the learning function"""
    Quiet = 0
    Normal = 1
    Verbose = 2


class AgentWrapper(Wrapper):
    """Gym wrapper to update the agent state on every step"""

    def __init__(self, env: cyberbattle_env.CyberBattleEnv, state: StateAugmentation):
        super().__init__(env)
        self.state = state
        self.defender_rewards=[]
        self.defender_reward=0.0
        self.total_defender_rewards=0.0
        #每次训练开始之前都需要setup_learn 让模型初始化
        if env.defender_agent:
            env.defender_agent.setup_learn(total_timesteps=300000,eval_env=None,eval_freq=-1,n_eval_episodes=5,eval_log_path=None,reset_num_timesteps=True,tb_log_name="OnPolicyAlgorithm")


    def step(self, action: cyberbattle_env.Action):
        #d_continue,d_new_obs,d_dones是为self.defender准备的
        observation, reward, done, info ,d_continue,d_new_obs,d_dones,d_rewards= self.env.step(action)
        self.state.on_step(action, reward, done, observation)
        self.d_continue=d_continue
        self.d_new_obs=d_new_obs
        self.d_dones=d_dones
        self.defender_reward=d_rewards
        self.defender_rewards.append(d_rewards)
        self.total_defender_rewards+=d_rewards
        #if done :
            #print("yes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #if d_dones:
            #self.render()
            #print("NO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.state.on_reset(observation)
        self.defender_rewards=[]
        self.total_defender_rewards=0.0
        self.defender_reward=0.0
        if self.env.defender_agent:#每个episode开头都要有
            self.env.defender_agent.on_rollout_start()
        return observation
    def defender_on_rollout_end(self):
        if self.env.defender_agent:
            self.env.defender_agent.on_rollout_end(self.d_new_obs,self.d_dones)
    def defender_train(self):#每个episode结束都要有
        #log_training还没考虑到
        if self.env.defender_agent:
            self.defender_on_rollout_end()
            self.env.defender_agent.train()
            self.env.defender_agent.on_rollout_start()

    def defender_on_training_end(self):
        if self.env.defender_agent:
            self.env.defender_agent.on_training_end()

    def defender_save(self,defender_filepath):    
        if self.env.defender_agent and defender_filepath:
            #self.env.logger.info('Defender agent saving...')
            self.env.defender_agent.save(defender_filepath)
    
    def load_defender(self,filename):
        self.env.defender_builder=LoadFileBaselineAgentBuilder(alg_type=PPO,file_path=filename)
        self.defender_wrapper=DefenderEnvWrapper(self,event_source=self.env.event_source)
        self.defender_agent=self.defender_builder.build(self.defender_wrapper,self.logger)
        
class AgentWrapper_chushi(Wrapper):
    """Gym wrapper to update the agent state on every step"""

    def __init__(self, env: cyberbattle_env.CyberBattleEnv, state: StateAugmentation):
        super().__init__(env)
        self.state = state
        
    def step(self, action: cyberbattle_env.Action):
        observation, reward, done, info = self.env.step(action)
        self.state.on_step(action, reward, done, observation)
        
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.state.on_reset(observation)
        return observation


Defender_Observation = TypedDict('Defender_Observation', {'infected_nodes': np.ndarray,
                                                          'incoming_firewall_status':np.ndarray,
                                                          'outgoing_firewall_status':np.ndarray,
                                                          'services_status':np.ndarray})

class DefenderEnvWrapper_chushi(gym.Env, IEnvironmentObserver):
    """Wraps a CyberBattleEnv for stablebaselines-3 models to learn how to defend."""

    nested_spaces = ['credential_cache_matrix', 'leaked_credentials']
    other_removed_spaces = ['local_vulnerability', 'remote_vulnerability', 'connect']
    int32_spaces = ['customer_data_found', 'escalation', 'lateral_move', 'newly_discovered_nodes_count', 'probe_result']
    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]

    def __init__(self,
        cyber_env: cyberbattle_env.CyberBattleEnv,
        #attacker_reward_store: IRewardStore,
        event_source: Optional[EnvironmentEventSource] = None,
        defender: bool = False,
        max_timesteps=100,
        invalid_action_reward=0,
        reset_on_constraint_broken = True):

        super().__init__()
        self.defender = None
        self.cyber_env: cyberbattle_env.CyberBattleEnv = cyber_env
        self.bounds: EnvironmentBounds = self.cyber_env._bounds
        self.num_services = 0
        self.observation_space: Space = self.__create_observation_space(cyber_env)
        self.action_space: Space = self.__create_defender_action_space(cyber_env)
        self.network_availability: float = 1.0
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.max_timesteps = max_timesteps
        self.timesteps = 0
        self.rewards = []
        self.attacker_reward_store = attacker_reward_store # 为什么要对attacker的奖励进行存储？
        self.first = True
        self.reset_request = False
        self.invalid_action_penalty = invalid_action_reward
        # Add this object as an observer of the cyber env.
        if event_source is None:
            event_source = EnvironmentEventSource()

        self.event_source = event_source
        event_source.add_observer(self)
        assert defender is not None, "Attempting to use the defender environment without a defender present."
        self.defender: LearningDefender1 = LearningDefender1()
        self.__last_attacker_reward = None
        self.reset_on_constraint_broken = reset_on_constraint_broken

    def __create_observation_space(self, cyber_env: cyberbattle_env.CyberBattleEnv) -> gym.Space:
        """Creates a compatible version of the attackers observation space."""
        # Calculate how many services there are, this is used to define the maximum number of services active at once.
        for _, node in model.iterate_network_nodes(cyber_env.environment.network):
            for _ in node.services:
                self.num_services +=1
        # All spaces are MultiBinary.
        return spaces.Dict({'infected_nodes': spaces.MultiBinary(len(list(cyber_env.environment.network.nodes))),
                            'incoming_firewall_status': spaces.MultiBinary(len(self.firewall_rule_list)*len(list(cyber_env.environment.network.nodes))),
                            'outgoing_firewall_status': spaces.MultiBinary(len(self.firewall_rule_list)*len(list(cyber_env.environment.network.nodes))),
                            'services_status': spaces.MultiBinary(self.num_services)})

    def __create_defender_action_space(self, cyber_env: cyberbattle_env.CyberBattleEnv) -> gym.Space:
        # 0th index of the action defines which action to use (reimage, block_traffic, allow_traffic, stop_service, start_service)
        # Index 1 is the possible nodes to reimage (all nodes) (Only used on action 0)
        # Index 2, 3, 4 are for action 1 (block traffic) 2nd = node to block on, 3rd =Port to block, 4th = incoming or outgoing
        # Index 5, 6, 7 relate to action 2 (allow traffic), 5th = node to allow on, 6th = Port to allow, 7th = incoming or outgoing
        # Index 8 and 9 are for action 3 (stop service), 8th = node to stop service on, 9th = port to stop service
        # Index 10 and 11 are for action 4 (start service), 10th = node to start service on, 11th = port to start service on.
        total_actions = 5
        reimage_node_number = len(cyber_env.environment.network.nodes)
        block_traffic_node = len(cyber_env.environment.network.nodes)
        block_traffic_port = 6
        block_traffic_incoming = 2
        allow_traffic_node = len(cyber_env.environment.network.nodes)
        allow_traffic_port = 6
        allow_traffic_incoming = 2
        stop_service_node = len(cyber_env.environment.network.nodes)
        stop_service_port = 3
        start_service_node = len(cyber_env.environment.network.nodes)
        start_service_port = 3
        action_space = [total_actions, reimage_node_number, block_traffic_node, block_traffic_port, block_traffic_incoming, allow_traffic_node, allow_traffic_port, allow_traffic_incoming, stop_service_node, stop_service_port, start_service_node, start_service_port]
        logging.info(f"Action space defender = {action_space}")
        return spaces.MultiDiscrete(action_space)

    def step(self, action) -> Tuple[cyberbattle_env.Observation, float, bool, Dict[str, Any]]:
        done = False
        reward = 0
        # Check for action validity
        if not self.is_defender_action_valid(action):
            logging.info(f"Action chosen is outside action space. Defender will skip this turn. Action = {action}")
            self.invalid_action_count += 1
            reward += self.invalid_action_penalty
            # If the action is invalid, pass an empty list to the defender
            action = []
        else:
            self.valid_action_count += 1
        # 操作就是defender的动作空间里面的，如果是合法的就执行，然后根据attacker的得分取反，如果非法，非法的分数和动作+1.
        self.defender.executeAction(action)
        # Take the reward gained this step from the attacker's step and invert it so the defender
        # loses more reward if the attacker succeeds.将攻击者这一步获得的奖励取反，以便在攻击者成功的情况下，防守者失去更多奖励。
        if self.attacker_reward_store.episode_rewards:
            reward += -1*self.attacker_reward_store.episode_rewards[-1]

        if self.defender_constraints_broken():
            reward = self.cyber_env._CyberBattleEnv__LOSING_REWARD
            logging.warning("Defender Lost")
            if self.reset_on_constraint_broken:#这里若为true，那么done为true说明回合结束，说明若防御者边界被打破是会重置环境
                done = True
        if self.cyber_env._CyberBattleEnv__defender_goal_reached():
            reward = self.cyber_env._CyberBattleEnv__WINNING_REWARD
            done = True
        # Generate the new defender observation based on the defender's action
        defender_observation = self.observe()
        self.timesteps += 1

        if self.reset_request:
            done = True
            reward = -1*self.__last_attacker_reward
        elif self.timesteps > self.max_timesteps:
            done = True

        self.rewards.append(reward)
        #...zrf加的
        #if reward>0:
            #print(f'attack####### rewarded action: {action}')
            #print(f'reward={reward}')
            #self.cyber_env.render()
        #。。。
        return defender_observation, reward, done, {}

    def is_defender_action_valid(self, action) -> boolean:
        """Determines if a given action is valid within the environment."""
        
        def get_node_and_info(node_from_action: int):
            """Returns the node id and info for a given node"""
            node_id = get_node_from_action(node_from_action)
            node_info = get_node_info(node_id)
            return node_id, node_info

        def get_node_from_action(node_from_action: int):
            """Gets the node id from an action"""
            return list(self.cyber_env.environment.network.nodes)[node_from_action]

        def get_node_info(node_id: model.NodeID):
            """Given a node ID, find the corresponding node info"""
            return self.cyber_env.environment.get_node(node_id)

        def node_exists(node_id: model.NodeID):
            """Determines if a node exists in the network"""
            return node_id in list(self.cyber_env.environment.network.nodes)

        def node_running(node_info: model.NodeInfo):
            """Determines if a node is currently running"""
            return node_info.status == model.MachineStatus.Running

        def node_exists_and_running(node_from_action: int):
            """Determines if a node exists in the network, and if so if it is running."""
            node_id, node_info = get_node_and_info(node_from_action)
            return (node_exists(node_id) and node_running(node_info))

        def is_reimagable(node_info: model.NodeInfo):
            """Checks if a given node is reimagable"""
            return node_info.reimagable

        def firewall_rule_exists(node_info: model.NodeInfo, port_from_action: int, incoming :bool):
            """Checks a node to see if a given firewall rule exists on it."""
            firewall_list = []
            if incoming:
                for rule in node_info.firewall.incoming:
                    firewall_list.append(rule.port)
            else:
                for rule in node_info.firewall.outgoing:
                    firewall_list.append(rule.port)

            return self.firewall_rule_list[port_from_action] in firewall_list

        def service_exists(node_info: model.NodeInfo, service_from_action: int):
            """Checks if a service exists on a node (Only checks if the service is out of bounds for the node)"""
            return service_from_action < len(node_info.services)
        action_number = action[0]
        if action_number == 0:
            # REIMAGE
            _, node_info = get_node_and_info(action[1])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            return node_exists_and_running(action[1]) and is_reimagable(node_info)

        elif action_number == 1:
            # block traffic
            _, node_info = get_node_and_info(action[2])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # The firewall rule needs to exist as well to block the traffic.
            return node_exists_and_running(action[2]) and firewall_rule_exists(node_info, action[3], bool(action[4]))

        elif action_number == 2:
            # allow traffic
            _, node_info = get_node_and_info(action[5])
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            return node_exists_and_running(action[5])

        elif action_number == 3:
            # stop service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to stop does not exist, this is invalid
            _, node_info = get_node_and_info(action[8])
            return node_exists_and_running(action[8]) and service_exists(node_info, action[9])

        elif action_number == 4:
            # start service
            # If the node does not exist, or if the node is not currently running, this action is invalid.
            # Also if the service to start does not exist, this is invalid
            _, node_info = get_node_and_info(action[10])
            return node_exists_and_running(action[10]) and service_exists(node_info, action[11])
        else:
            return False

    def reset(self) -> cyberbattle_env.Observation:
        logging.debug('Reset Defender')
        if not self.reset_request:
            self.event_source.notify_reset(last_reward=0)

        self.cyber_env.reset()

        self.reset_request = False
        self.__last_attacker_reward = None
        self.rewards = []
        self.timesteps = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0

        return self.observe()

    def on_reset(self, last_reward):
        logging.debug('on_reset Defender')
        self.reset_request = True
        self.__last_attacker_reward = last_reward

    def get_blank_defender_observation(self):
        """ Creates a empty defender observation. """
        obs = Defender_Observation(infected_nodes = [],
                                    incoming_firewall_status=[],
                                    outgoing_firewall_status=[],
                                    services_status=[])
        return obs

    def observe(self) -> Defender_Observation:
        """Gathers information directly from the environment to generate populate an observation for the defender agent to use."""

        # 更新观察，就是遍历每一个节点，便利每一个节点上的规则，然后进行统计。
        new_observation=self.get_blank_defender_observation()
        incoming_firewall_list = [0]*(len(self.cyber_env.environment.network.nodes)*len(self.firewall_rule_list))
        outgoing_firewall_list = [0]*(len(self.cyber_env.environment.network.nodes)*len(self.firewall_rule_list))
        all_services_list = [0]*self.num_services
        count_incoming_firewall = -1
        count_outgoing_firewall = -1
        count_services = -1

        # Iterates through all nodes in the environment.
        for _, node in model.iterate_network_nodes(self.cyber_env.environment.network):
            # Incoming Firewall rules section. Counts which incoming firewall rules are active.
            for rule in self.firewall_rule_list:
                count_incoming_firewall+=1
                for entry in node.firewall.incoming:
                    if rule == entry.port:
                        incoming_firewall_list[count_incoming_firewall] = 1

            # Outgoing Firewall rules section. Counts which outgoing firewall rules are active.
            for rule in self.firewall_rule_list:
                count_outgoing_firewall+=1
                for entry in node.firewall.outgoing:
                    if rule == entry.port:
                        outgoing_firewall_list[count_outgoing_firewall] = 1
                    
            # Services Section. Counts the currently running services.
            for service in node.services:
                count_services+=1
                if service.running:
                    all_services_list[count_services] = 1
                    
        # Take information from the environment and format it for defender agent observation.
        # Check all nodes and find which are infected. 1 if infected 0 if not.
        new_observation["infected_nodes"] = np.array([1 if node.agent_installed else 0 for _, node in model.iterate_network_nodes(self.cyber_env.environment.network)])
        # Lists all possible incoming firewall rules, 1 if active, 0 if not.
        new_observation['incoming_firewall_status'] = np.array(incoming_firewall_list)
        # Lists all possible outgoing firewall rules, 1 if active, 0 if not.
        new_observation['outgoing_firewall_status'] = np.array(outgoing_firewall_list)
        # Lists all possible services, 1 if active, 0 if not.
        new_observation['services_status'] = np.array(all_services_list)
        return new_observation

    def set_reset_request(self, reset_request):
        self.reset_request = reset_request

    def defender_constraints_broken(self):
        return self.cyber_env._defender_actuator.network_availability < self.cyber_env._CyberBattleEnv__defender_constraint.maintain_sla

    def close(self) -> None:
        return self.cyber_env.close()

    def render(self, mode: str = 'human') -> None:
        return self.cyber_env.render(mode)

    def render_as_fig(self) -> FigureWidget:
        return self.cyber_env.render_as_fig()
