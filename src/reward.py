import numpy as np
# import env

class RewardFunctionTerminalPenalty():
    def __init__(self, sheepId, aliveBouns, deathPenalty, isTerminal):
        self.sheepId = sheepId
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action):
       # physicalState, beliefAndAttention = state
       # agentStates, agentActions,timeStep, wolfIdAndSubtlety = physicalState
       # wolfId, wolfSubtlety = wolfIdAndSubtlety
       # sheepPosition = agentStates[self.sheepId]
       # wolfPosition = agentStates[wolfId]
       # distanceToWolfReward = 0 * (0.01 * np.power(np.sum(np.power(sheepPosition - wolfPosition, 2)), 0.5))
       # reward = distanceToWolfReward +  self.aliveBouns
        reward = self.aliveBouns
        if self.isTerminal(state):
            reward = reward + self.deathPenalty
        return reward
