import cyberbattle._env.defender as cyberDefender
from cyberbattle.simulation import model
class ReimageDefender(cyberDefender.DefenderAgent):
    """A reimage defender to test things out."""
    def __init__(self) -> None:
        pass
    def step(self, environment: cyberDefender.Environment, actions: cyberDefender.DefenderAgentActions, current_step: int):
        if current_step % 10 == 0:
            scanned_nodes = cyberDefender.random.choices(list(environment.network.nodes), k=1)
            for node_id in scanned_nodes:
                node_info = environment.get_node(node_id)
                if node_info.status == cyberDefender.model.MachineStatus.Running and node_info.agent_installed:
                    is_malware_detected = cyberDefender.random.random() <= 0.5
                    if is_malware_detected:
                        if node_info.reimagable:
                            cyberDefender.logging.error(f"Defender detected malware, reimaging node {node_id}")
                            actions.reimage_node(node_id)
                        else:
                            cyberDefender.logging.error(f"Defender detected malware, but node cannot be reimaged {node_id}")

class LearningDefender1():
    """A defender that in theory will link up into the defend_wrapper"""
    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]
    def __init__(self) -> None:
        #self.cyber_env = cyber_env
        pass
        
    def executeAction(self, next_action,environment: cyberDefender.Environment ,actions: cyberDefender.DefenderAgentActions)->int:
        actions.on_attacker_step_taken()#为什么更新网络可用性再做防御者动作？因为攻击者会改变网络可用性，若降低到某个标准攻击者胜利
        before=actions.network_availability
        defender_action_reward=0
        def get_node_from_action(node_from_action: int):
            """Converts from action number to node ID."""
            return list(environment.network.nodes)[node_from_action]

        def get_node_info(node_id: model.NodeID):
            """Gets node info from node ID."""
            return environment.get_node(node_id)

        def get_firewall_port_name_from_action(port_name_from_action: int):
            """Gets the name of the firewall port from the constant firewall rule list."""
            return self.firewall_rule_list[port_name_from_action]

        def get_service_port_name_from_action(node_id: model.NodeID, port_name_from_action: int):
            """Gets the service port name from the given node."""
            node_info = get_node_info(node_id)
            #return node_info.services[port_name_from_action]  zrf：感觉这里不对，应该是返回str形式 不然后面采取动作都是错的 都无法配对==
            return node_info.services[port_name_from_action].name

        def block_traffic(node_id: model.NodeID, port_name: model.PortName, incoming: bool):
            """Blocks traffic on a node to or from a port with port_name."""
            node_data = environment.get_node(node_id)
            node_info = get_node_info(node_id)
            rules = node_data.firewall.incoming if incoming else node_data.firewall.outgoing
            matching_rules = [r for r in rules if r.port == port_name]
            if matching_rules:
                for rule in matching_rules:
                    node_info.firewall.incoming.remove(rule) if incoming else node_info.firewall.outgoing.remove(rule)
        
        def allow_traffic(node_id: model.NodeID, port_name: model.PortName, incoming: bool):
            """Creates a new firewall rule if one does not exist."""
            node_data = environment.get_node(node_id)
            node_info = get_node_info(node_id)
            rules = node_data.firewall.incoming if incoming else node_data.firewall.outgoing
            matching_rules = [r for r in rules if r.port == port_name]
            if not matching_rules:
                rule_to_add = model.FirewallRule(port = port_name, permission=model.RulePermission.ALLOW)
                node_info.firewall.incoming.append(rule_to_add) if incoming else node_info.firewall.outgoing.append(rule_to_add)#小改了一个outgoing
        
        # If the action is invalid, the list will be empty. In this case the defender will skip its turn.
        if len(next_action) == 0 :
            return defender_action_reward
        # If the action is a reimage, reimage the node.
        if next_action[0] == 0:
            node_id=get_node_from_action(next_action[1])
            if get_node_info(node_id).agent_installed:
                node_info=get_node_info(node_id)
                value=node_info.value
                defender_action_reward+=100*value
            actions.reimage_node(get_node_from_action(next_action[1]))
            #print("rrrrrreeeee")
            #print(get_node_from_action(next_action[1]))
        # If the action is a block traffic.
        elif next_action[0] == 1:
            node_id = get_node_from_action(next_action[2])
            incoming = bool(next_action[4])
            port_name = get_firewall_port_name_from_action(next_action[3])
            block_traffic(node_id, port_name, incoming)
        
        # If the action is a allow traffic.
        elif next_action[0] == 2:
            node_id = get_node_from_action(next_action[5])
            incoming = bool(next_action[7])
            port_name = get_firewall_port_name_from_action(next_action[6])
            allow_traffic(node_id, port_name, incoming)
        
        # If the action is a stop service.
        elif next_action[0] == 3:
            node_id = get_node_from_action(next_action[8])
            actions.stop_service(node_id, get_service_port_name_from_action(node_id, next_action[9]))

        # If the action is a start service.
        elif next_action[0] == 4:
            node_id = get_node_from_action(next_action[10])
            actions.start_service(node_id, get_service_port_name_from_action(node_id, next_action[11]))
        #actions.on_attacker_step_taken()会改变网络可用性，不再是针对攻击者动作后的而是针对防御后的了，无法用来判断攻击者是否赢了！！
        after=actions.on_attacker_step_taken_for_defender_reward()
        #after=actions.network_availability
        if before<after:
            defender_action_reward+=500
            #print("NONONONOONONONOONONONONO")
        #if defender_action_reward>0:
           # print("")
        return defender_action_reward

'''lass LearningDefender():
    """A defender that in theory will link up into the defend_wrapper"""
    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]
    def __init__(self, cyber_env: CyberBattleEnv ) -> None:
        self.cyber_env = cyber_env
    def executeAction(self, next_action):
        self.cyber_env._defender_actuator.on_attacker_step_taken()
        def get_node_from_action(node_from_action: int):
            """Converts from action number to node ID."""
            return list(self.cyber_env.environment.network.nodes)[node_from_action]

        def get_node_info(node_id: model.NodeID):
            """Gets node info from node ID."""
            return self.cyber_env.environment.get_node(node_id)

        def get_firewall_port_name_from_action(port_name_from_action: int):
            """Gets the name of the firewall port from the constant firewall rule list."""
            return self.firewall_rule_list[port_name_from_action]

        def get_service_port_name_from_action(node_id: model.NodeID, port_name_from_action: int):
            """Gets the service port name from the given node."""
            node_info = get_node_info(node_id)
            return node_info.services[port_name_from_action]
             
        def block_traffic(node_id: model.NodeID, port_name: model.PortName, incoming: bool):
            """Blocks traffic on a node to or from a port with port_name."""
            node_data = self.cyber_env.environment.get_node(node_id)
            node_info = get_node_info(node_id)
            rules = node_data.firewall.incoming if incoming else node_data.firewall.outgoing
            matching_rules = [r for r in rules if r.port == port_name]
            if matching_rules:
                for rule in matching_rules:
                    node_info.firewall.incoming.remove(rule) if incoming else node_info.firewall.outgoing.remove(rule)
        
        def allow_traffic(node_id: model.NodeID, port_name: model.PortName, incoming: bool):
            """Creates a new firewall rule if one does not exist."""
            node_data = self.cyber_env.environment.get_node(node_id)
            node_info = get_node_info(node_id)
            rules = node_data.firewall.incoming if incoming else node_data.firewall.outgoing
            matching_rules = [r for r in rules if r.port == port_name]
            if not matching_rules:
                rule_to_add = model.FirewallRule(port = port_name, permission=model.RulePermission.ALLOW)
                node_info.firewall.incoming.append(rule_to_add) if incoming else node_info.firewall.outgoing.append(rule_to_add)#小改了一个outgoing
        
        # If the action is invalid, the list will be empty. In this case the defender will skip its turn.
        if len(next_action) == 0 :
            return
        # If the action is a reimage, reimage the node.
        if next_action[0] == 0:
            self.cyber_env._defender_actuator.reimage_node(get_node_from_action(next_action[1]))
        
        # If the action is a block traffic.
        elif next_action[0] == 1:
            node_id = get_node_from_action(next_action[2])
            incoming = bool(next_action[4])
            port_name = get_firewall_port_name_from_action(next_action[3])
            block_traffic(node_id, port_name, incoming)
        
        # If the action is a allow traffic.
        elif next_action[0] == 2:
            node_id = get_node_from_action(next_action[5])
            incoming = bool(next_action[7])
            port_name = get_firewall_port_name_from_action(next_action[6])
            allow_traffic(node_id, port_name, incoming)
        
        # If the action is a stop service.
        elif next_action[0] == 3:
            node_id = get_node_from_action(next_action[8])
            self.cyber_env._defender_actuator.stop_service(node_id, get_service_port_name_from_action(node_id, next_action[9]))

        # If the action is a start service.
        elif next_action[0] == 4:
            node_id = get_node_from_action(next_action[10])
            self.cyber_env._defender_actuator.start_service(node_id, get_service_port_name_from_action(node_id, next_action[11]))'''