import numpy as np
import random
import itertools as it
import pygame as pg


# Local import
import env
import agentsEnv as ag
import reward
import policyGradient as PG
from model import GeneratePolicyNet
from evaluate import Evaluate
from visualize import draw


def main():
    # action space
    actionSpace = [[10, 0], [7, 7], [0, 10], [-7, 7], [-10, 0], [-7, -7], [0, -10], [7, -7]]
    numActionSpace = len(actionSpace)

    # state space
    numStateSpace = 4
    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary)

    initSheepPositionMean = np.array([180, 180])
    initWolfPositionMean = np.array([180, 180])
    initSheepPositionNoise = np.array([120, 120])
    initWolfPositionNoise = np.array([60, 60])
    sheepPositionReset = ag.SheepPositionReset(initSheepPositionMean, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionReset = ag.WolfPositionReset(initWolfPositionMean, initWolfPositionNoise, checkBoundaryAndAdjust)

    numOneAgentState = 2
    positionIndex = [0, 1]

    sheepPositionTransition = ag.SheepPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust)
    wolfPositionTransition = ag.WolfPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust)

    numAgent = 2
    sheepId = 0
    wolfId = 1
    transitionFunction = env.TransitionFunction(sheepId, wolfId, sheepPositionReset, wolfPositionReset,
                                                sheepPositionTransition, wolfPositionTransition)
    minDistance = 15
    isTerminal = env.IsTerminal(sheepId, wolfId, numOneAgentState, positionIndex, minDistance)

    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255, 255, 255]
    circleColorList = [[50, 255, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50],
                       [50, 50, 50], [50, 50, 50], [50, 50, 50]]
    circleSize = 8
    saveImage = False
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize,
                        saveImage, saveImageFile)

    aliveBouns = -1
    deathPenalty = 20
    rewardDecay = 0.99
    rewardFunction = reward.TerminalPenalty(sheepId, wolfId, numOneAgentState, positionIndex, aliveBouns, deathPenalty, isTerminal)
    accumulateRewards = PG.AccumulateRewards(rewardDecay, rewardFunction)

    maxTimeStep = 150
    sampleTrajectory = PG.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal)

    approximatePolicy = PG.ApproximatePolicy(actionSpace)
    trainPG = PG.TrainTensorflow(actionSpace)

    numTrajectory = 20
    maxEpisode = 1000

    # Generate models.
    learningRate = 1e-4
    hiddenNeuronNumbers = [128, 256, 512, 1024]
    hiddenDepths = [2, 4, 8]
    # hiddenNeuronNumbers = [128]
    # hiddenDepths = [2]
    generateModel = GeneratePolicyNet(numStateSpace, numActionSpace, learningRate)
    models = {(n, d): generateModel(d, round(n / d)) for n, d in it.product(hiddenNeuronNumbers, hiddenDepths)}
    print("Models generated")

    # Train.
    policyGradient = PG.PolicyGradient(numTrajectory, maxEpisode, render)
    trainModel = lambda model: policyGradient(model, approximatePolicy,
                                                             sampleTrajectory,
                                                             accumulateRewards,
                                                             trainPG)
    trainedModels = {key: trainModel(model) for key, model in models.items()}
    print("Finished training")

    # Evaluate
    modelEvaluate = Evaluate(numTrajectory, approximatePolicy, sampleTrajectory, rewardFunction)
    meanEpisodeRewards = {key: modelEvaluate(model) for key, model in trainedModels.items()}
    print("Finished evaluating")
    # print(meanEpisodeRewards)

    # Visualize
    independentVariableNames = ['NeuroTotalNumber', 'layerNumber']
    draw(meanEpisodeRewards, independentVariableNames)
    print("Finished visualizing", meanEpisodeRewards)

if __name__ == "__main__":
    main()
