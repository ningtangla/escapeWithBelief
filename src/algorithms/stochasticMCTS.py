import numpy as np
import itertools as it
import copy
import anytree
from anytree import AnyNode as Node
from anytree import RenderTree, findall

class CalculateScore:
    def __init__(self, c_init, c_base):
        self.c_init = c_init
        self.c_base = c_base
    
    def __call__(self, curr_node, child):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        action_prior = child.action_prior

        if self_visit_count == 0:
            u_score = np.inf
            q_score = 0
        else:
            exploration_rate = np.log((1 + parent_visit_count + self.c_base) / self.c_base) + self.c_init
            u_score = exploration_rate * action_prior * np.sqrt(parent_visit_count) / float(1 + self_visit_count) 
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score
        return score


class SelectChild:
    def __init__(self, calculate_score):
        self.calculate_score = calculate_score

    def __call__(self, curr_node):
        scores = [self.calculate_score(curr_node, child) for child in curr_node.children]
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = np.random.choice(maxIndex)
        selected_child = curr_node.children[selected_child_index]
        return selected_child


class GetActionPrior:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def __call__(self, curr_state):
        action_prior = {action: 1/len(self.action_space) for action in self.action_space}
        return action_prior 


class Expand:
    def __init__(self, is_terminal, initializeChildren):
        self.is_terminal = is_terminal
        self.initializeChildren = initializeChildren

    def __call__(self, leaf_node):
        curr_state = list(leaf_node.id.values())[0]
        if not self.is_terminal(curr_state):
            leaf_node.is_expanded = True
        if len(leaf_node.children) == 0:
            leaf_node = self.initializeChildren(leaf_node)
        return leaf_node


class RollOut:
    def __init__(self, rollout_policy, max_rollout_step, transition_func, reward_func, is_terminal):
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.max_rollout_step = max_rollout_step
        self.rollout_policy = rollout_policy
        self.is_terminal = is_terminal

    def __call__(self, leaf_node):
        curr_state = list(leaf_node.id.values())[0]
        sum_reward = 0
        for rollout_step in range(self.max_rollout_step):
            action = self.rollout_policy(curr_state)
            sum_reward += self.reward_func(curr_state, action)
            if self.is_terminal(curr_state):
                break

            next_state = self.transition_func(curr_state, action)
            curr_state = next_state
        return sum_reward

def backup(value, node_list):
    for node in node_list:
        node.sum_value += value
        node.num_visited += 1

class SelectAction():
    def __init__(self, numActionPlaned, actionSpace):
        self.numActionPlaned = numActionPlaned
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)
        self.actionIndexCombosForActionPlaned = list(it.product(range(self.numActionSpace), repeat = self.numActionPlaned))
        self.numActionIndexCombos = len(self.actionIndexCombosForActionPlaned)
    def __call__(self, roots):
        #aaa = [[child.num_visited for child in findall(root, lambda node: node.depth == self.numActionPlaned)] for root in roots]
        grandchildren_visit = np.sum([[child.num_visited for child in findall(root, lambda node: node.depth == self.numActionPlaned)] for root in roots], axis=0)
        #__import__('ipdb').set_trace()
        #for root in roots:
        #    for comboIndex in range(self.numActionIndexCombos):
        #        currParent = root
        #        for actionOrder in range(self.numActionPlaned):
        #            child = currParent.children[self.actionIndexCombosForActionPlaned[comboIndex][actionOrder]
        #            currParent = child
        #        numVisits[comboIndex] + = child.num_visited
        
        maxIndex = np.argwhere(grandchildren_visit == np.max(grandchildren_visit)).flatten()
        #print(maxIndex,grandchildren_visit)
        selectedActionIndexCombos = np.random.choice(maxIndex)
        action = [self.actionSpace[actionIndex] for actionIndex in self.actionIndexCombosForActionPlaned[selectedActionIndexCombos]]
        return action


class InitializeChildren:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, node):
        state = list(node.id.values())[0]
        initActionPrior = self.getActionPrior(state)
        for action in self.actionSpace:
            nextState = self.transition(state, action)
            Node(parent=node, id={action: nextState}, num_visited=0, sum_value=0, action_prior=initActionPrior[action], is_expanded=False)
        #print(state[0][1: 3], 'simulation', nextState[0][1:3])

        return node

class MCTS:
    def __init__(self, num_simulation, selectChild, expand, rollout, backup, selectAction, mctsRender, mctsRenderOn):
        self.num_simulation = num_simulation
        self.select_child = selectChild
        self.expand = expand
        self.rollout = rollout
        self.backup = backup
        self.selectAction = selectAction
        self.mctsRender = mctsRender
        self.mctsRenderOn = mctsRenderOn
    
    def __call__(self, curr_roots):
        num_tree = len(curr_roots)
        roots = []
        
        state = list(curr_roots[0].id.values())[0] 
        physicalState, beliefAndAttention = state 
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        for tree_index in range(num_tree):
            curr_tree_root = copy.deepcopy(curr_roots[tree_index])
            curr_tree_root = self.expand(curr_tree_root)
            backgroundScreen = None
            for explore_step in range(self.num_simulation):
                #print(explore_step)
                curr_node = curr_tree_root
                node_path = [curr_node]

                while curr_node.is_expanded:
                    next_node = self.select_child(curr_node)

                    if self.mctsRenderOn and timeStep > 8 and timeStep % 3 == 0:
                        backgroundScreen = self.mctsRender(curr_node, next_node, backgroundScreen)
                    node_path.append(next_node) 

                    curr_node = next_node

                leaf_node = self.expand(curr_node)
                value = self.rollout(leaf_node)
                self.backup(value, node_path)
            roots.append(curr_tree_root)
        action = self.selectAction(roots)
        return action

def main():
    pass

if __name__ == "__main__":
    main()
