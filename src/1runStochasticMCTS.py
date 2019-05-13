import numpy as np
import pygame as pg
import itertools as it
import math

from anytree import AnyNode as Node
from anytree import RenderTree

# Local import
# from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
from algorithms.stochasticMCTS import MCTS, CalculateScore, GetActionPrior, SelectAction, SelectChild, Expand, RollOut, backup, InitializeChildren

from simple1DEnv import TransitionFunction, RewardFunction, Terminal
from visualize import draw
import stochasticAgentsMotionSimulation as ag
import Attention
import calPosterior
import stochasticBeliefAndAttentionSimulation as ba
import env
import reward

class MakeDiffSimulationRoot():
    def __init__(self, isTerminal, updatePhysicalStateByBelief):
        self.isTerminal = isTerminal
        self.updatePhysicalStateByBelief = updatePhysicalStateByBelief
    def __call__(self, action, state):
        stateForSimulationRoot = self.updatePhysicalStateByBelief(state)
        while self.isTerminal(stateForSimulationRoot):
            stateForSimulationRoot = self.updatePhysicalStateByBelief(state)
        #print(state[0][3],stateForSimulationRoot[0][3]) 
        rootNode = Node(id={action: stateForSimulationRoot}, num_visited=0, sum_value=0, is_expanded = False)
        return rootNode

class RunMCTS:
    def __init__(self, maxRunningSteps, numTree, planFrequency, transitionFunctionInPlay, isTerminal, makeDiffSimulationRoot, render):
        self.maxRunningSteps = maxRunningSteps
        self.numTree = numTree
        self.planFrequency = planFrequency
        self.transitionFunctionInPlay = transitionFunctionInPlay
        self.isTerminal = isTerminal
        self.makeDiffSimulationRoot = makeDiffSimulationRoot
        self.render = render

    def __call__(self, mcts):
        # Running
        state, action = None, None
        currState = self.transitionFunctionInPlay(state, action)
        for runningStep in range(self.maxRunningSteps):
            #import datetime
            #print (datetime.datetime.now())
            
            if self.isTerminal(currState):
                break
            rootNodes = [self.makeDiffSimulationRoot(action, currState) for treeIndex in range(self.numTree)]
            if runningStep % self.planFrequency == 0:
                actions = mcts(rootNodes)
            action = actions[runningStep % self.planFrequency]
            
            nextState = self.transitionFunctionInPlay(currState, action) 
            currState = nextState
        
        # Output number of steps to reach the target.
        print(runningStep)
        return runningStep

def evaluate(numTree, chasingSubtlety, numTotalSimulationTimes, cInit, cBase):
    print(numTree, chasingSubtlety, numTotalSimulationTimes, cInit, cBase)
    numActionSpace = 8
    actionInterval = int(360/numActionSpace)
    actionSpace = [(np.cos(degreeInPolar), np.sin(degreeInPolar)) for degreeInPolar in np.arange(0, 360, actionInterval)/180 * math.pi]
    getActionPrior = GetActionPrior(actionSpace)

    # 2D Env
    initSheepPosition = np.array([320, 240]) 
    initSheepPositionNoise = np.array([0, 0])
    resetSheepState = ag.ResetAgentState(initSheepPosition, initSheepPositionNoise)
    initWolfOrDistractorPosition = np.array([320, 240])
    initWolfOrDistractorPositionNoise = np.array([125, 230])
    resetWolfOrDistractorState = ag.ResetAgentState(initWolfOrDistractorPosition, initWolfOrDistractorPositionNoise)
   
    numAgent = 25
    sheepId = 0
    suspectorIds = list(range(1, numAgent))

    resetWolfIdAndSubtlety = ag.ResetWolfIdAndSubtlety(suspectorIds, [chasingSubtlety])
    resetPhysicalState = ag.ResetPhysicalState(sheepId, numAgent, resetSheepState, resetWolfOrDistractorState, resetWolfIdAndSubtlety) 
   
    numFramePerSecond = 60
    numMDPTimeStepPerSecond = 5
    numFrameWithoutActionChange = int(numFramePerSecond/numMDPTimeStepPerSecond)
    
    sheepActionUpdateFrequency = 1 
    distanceToVisualDegreeRatio = 20
    minSheepSpeed = int(17.4 * distanceToVisualDegreeRatio/numFramePerSecond)
    maxSheepSpeed = int(23.2 * distanceToVisualDegreeRatio/numFramePerSecond)
    warmUpTimeSteps = int(10 * numMDPTimeStepPerSecond)
    sheepPolicy = ag.SheepPolicy(sheepActionUpdateFrequency, minSheepSpeed, maxSheepSpeed, warmUpTimeSteps)
    
    wolfActionUpdateFrequency = int(0.2 * numMDPTimeStepPerSecond)
    minWolfSpeed = int(8.7 * distanceToVisualDegreeRatio/numFramePerSecond)
    maxWolfSpeed = int(14.5 * distanceToVisualDegreeRatio/numFramePerSecond)
    wolfPolicy = ag.WolfPolicy(wolfActionUpdateFrequency, minWolfSpeed, maxWolfSpeed, warmUpTimeSteps)
    distractorActionUpdateFrequency = int(0.2 * numMDPTimeStepPerSecond)
    minDistractorSpeed = int(8.7 * distanceToVisualDegreeRatio/numFramePerSecond)
    maxDistractorSpeed = int(14.5 * distanceToVisualDegreeRatio/numFramePerSecond)
    distractorPolicy = ag.DistractorPolicy(distractorActionUpdateFrequency, minDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps)
    preparePolicy = ag.PreparePolicy(sheepId, numAgent, sheepPolicy, wolfPolicy, distractorPolicy)
    updatePhysicalState = ag.UpdatePhysicalState(numAgent, preparePolicy)

    xBoundary = [0, 640]
    yBoundary = [0, 480]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary) 
    transiteMultiAgentMotion = ag.TransiteMultiAgentMotion(checkBoundaryAndAdjust)
   
    minDistance = 2.5 * distanceToVisualDegreeRatio
    isTerminal = env.IsTerminal(sheepId, minDistance)
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    #screen = None
    screenColor = np.array([255,255,255])
    sheepColor = np.array([0, 255, 0])
    wolfColor = np.array([255, 0, 0])
    circleSize = 10
    saveImage = True
    saveImageFile = 'image1'
    render = env.Render(numAgent, screen, screenColor, sheepColor, wolfColor, circleSize, saveImage, saveImageFile)
    renderOnInSimulation = False
    transiteStateWithoutActionChangeInSimulation = env.TransiteStateWithoutActionChange(numFrameWithoutActionChange, isTerminal, transiteMultiAgentMotion, render, renderOnInSimulation)     
    renderOnInPlay = True
    transiteStateWithoutActionChangeInPlay = env.TransiteStateWithoutActionChange(numFrameWithoutActionChange, isTerminal, transiteMultiAgentMotion, render, renderOnInPlay)     
   
    attentionLimitation= 4
    precisionPerSlot=8.0
    precisionForUntracked=2.5
    memoryratePerSlot=0.7
    memoryrateForUntracked=0.45
    attention = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)    
    transferMultiAgentStatesToPositionDF = ba.TransferMultiAgentStatesToPositionDF(numAgent) 
    possibleSubtleties = [500, 11, 3.3, 1.83, 0.92, 0.31]
    resetBeliefAndAttention = ba.ResetBeliefAndAttention(sheepId, suspectorIds, possibleSubtleties, attentionLimitation, transferMultiAgentStatesToPositionDF, attention)
    
    maxDistance = 7.5 * distanceToVisualDegreeRatio
    numStandardErrorInDistanceRange = 2
    calDistancePriorOnAttentionSlot = Attention.CalDistancePriorOnAttentionSlot(minDistance, maxDistance, numStandardErrorInDistanceRange)
    attentionSwitch = Attention.AttentionSwitch(attentionLimitation, calDistancePriorOnAttentionSlot)    
    computePosterior = calPosterior.CalPosteriorLog(minDistance)

    attentionSwitchFrequencyInSimulation = np.inf
    beliefUpdateFrequencyInSimulation = np.inf
    updateBeliefAndAttentionInSimulation = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF,
            attentionSwitchFrequencyInSimulation, beliefUpdateFrequencyInSimulation)

    attentionSwitchFrequencyInPlay = int(0.6 * numMDPTimeStepPerSecond)
    beliefUpdateFrequencyInPlay = int(0.2 * numMDPTimeStepPerSecond)
    updateBeliefAndAttentionInPlay = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF, 
            attentionSwitchFrequencyInPlay, beliefUpdateFrequencyInPlay)

    updatePhysicalStateByBeliefFrequencyInSimulationRoot = int(0.2 * numMDPTimeStepPerSecond)
    updatePhysicalStateByBeliefInSimulationRoot = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInSimulationRoot)
    updatePhysicalStateByBeliefFrequencyInSimulation = np.inf
    updatePhysicalStateByBeliefInSimulation = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInSimulation)
    
    updatePhysicalStateByBeliefFrequencyInPlay = np.inf
    updatePhysicalStateByBeliefInPlay = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInPlay)

    transitionFunctionInSimulation = env.TransitionFunction(resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChangeInSimulation, 
            updateBeliefAndAttentionInSimulation, updatePhysicalStateByBeliefInSimulation)

    transitionFunctionInPlay = env.TransitionFunction(resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChangeInPlay, 
            updateBeliefAndAttentionInPlay, updatePhysicalStateByBeliefInPlay)
    
    maxRollOutSteps = 5
    aliveBouns = 1/maxRollOutSteps
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, aliveBouns, deathPenalty, isTerminal)  

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transitionFunctionInSimulation, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # Rollout
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunctionInSimulation, rewardFunction, isTerminal)
   
    numActionPlaned = 1
    selectAction = SelectAction(numActionPlaned, actionSpace)
    numSimulations = int(numTotalSimulationTimes/numTree)
    
    sheepColorInMcts = np.array([0, 255, 0])
    wolfColorInMcts = np.array([255, 0, 0])
    distractorColorInMcts = np.array([0, 0, 0])
    mctsRender = env.MctsRender(numAgent, screen, xBoundary[1], yBoundary[1], screenColor, sheepColorInMcts, wolfColorInMcts, distractorColorInMcts, circleSize, saveImage, saveImageFile)
    mctsRenderOn = True

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectAction, mctsRender, mctsRenderOn)
    
    maxRunningSteps = int(25 * numMDPTimeStepPerSecond)
    makeDiffSimulationRoot = MakeDiffSimulationRoot(isTerminal, updatePhysicalStateByBeliefInSimulationRoot)
    runMCTS = RunMCTS(maxRunningSteps, numTree, numActionPlaned, transitionFunctionInPlay, isTerminal, makeDiffSimulationRoot, render)

    rootAction = actionSpace[np.random.choice(range(numActionSpace))]
    numTestingIterations = 1
    episodeLengths = []
    escape = 0
    step = 1
    while step <= numTestingIterations:
        import datetime
        print (datetime.datetime.now())
        episodeLength = runMCTS(mcts)
        if episodeLength >= 1 * numMDPTimeStepPerSecond:
            step = step + 1
            episodeLengths.append(episodeLength)
            if episodeLength >= maxRunningSteps - 10:
                escape = escape + 1
    meanEpisodeLength = np.mean(episodeLengths)
    print("mean episode length is", meanEpisodeLength, escape/numTestingIterations)
    return [meanEpisodeLength, escape/numTestingIterations]

def main():
    
    cInit = [1]
    cBase = [50, 100]
    #modelResults = {(np.log10(init), np.log10(base)): evaluate(init, base) for init, base in it.product(cInit, cBase)}
    
    numTotalSimulationTimes = [64]
    numTrees = [1]
    subtleties = [500]
    #, 500, 11, 3.3, 1.83, 0.92, 0.31, 0.001]
    #, 50, 3.3, 0.92]
    precisionToSubtletyDict={500:0,50:5,11:30,3.3:60,1.83:90,0.92:120,0.31:150, 0.001: 180}
    modelResults = {(numTree, precisionToSubtletyDict[chasingSubtlety]): evaluate(numTree, chasingSubtlety, numTotalSimulation, init, base) for numTree, chasingSubtlety,
            numTotalSimulation, init, base in it.product(numTrees, subtleties, numTotalSimulationTimes, cInit, cBase)}

    print("Finished evaluating")
    # Visualize
    #independentVariableNames = ['cInit', 'cBase']
    independentVariableNames = ['numTree', 'chasingSubtlety']
    draw(modelResults, independentVariableNames)

if __name__ == "__main__":
    main()
    
