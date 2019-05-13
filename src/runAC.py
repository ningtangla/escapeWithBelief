import tensorflow as tf
import numpy as np
import functools as ft
import env
import reward
import tensorflow_probability as tfp
import random
import agentsEnv as ag
import itertools as it
import pygame as pg


# Local import
import offlineA2CMonteCarloAdvantageDiscrete as A2CMC
from model import GenerateActorCriticModel
from evaluate import Evaluate
from visualize import draw



def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)
    
    actionSpace = [[10,0],[7,7],[0,10],[-7,7],[-10,0],[-7,-7],[0,-10],[7,-7]]
    numActionSpace = len(actionSpace)
    numStateSpace = 4

    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    
    initSheepPosition = np.array([180, 180]) 
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([120, 120])
    initWolfPositionNoise = np.array([60, 60])
    sheepPositionReset = ag.SheepPositionReset(initSheepPosition, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionReset = ag.WolfPositionReset(initWolfPosition, initWolfPositionNoise, checkBoundaryAndAdjust)
    
    numOneAgentState = 2
    positionIndex = [0, 1]

    sheepPositionTransition = ag.SheepPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust) 
    wolfPositionTransition = ag.WolfPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust) 
    
    numAgent = 2
    sheepId = 0
    wolfId = 1
    transitionFunction = env.TransitionFunction(sheepId, wolfId, sheepPositionReset, wolfPositionReset, sheepPositionTransition, wolfPositionTransition)
    minDistance = 15
    isTerminal = env.IsTerminal(sheepId, wolfId, numOneAgentState, positionIndex, minDistance) 
     
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255,255,255]
    circleColorList = [[50,255,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50]]
    circleSize = 8
    saveImage = False
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageFile)

    aliveBouns = -1
    deathPenalty = 20
    rewardDecay = 0.99
    rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, wolfId, numOneAgentState, positionIndex, aliveBouns, deathPenalty, isTerminal) 
    accumulateReward = A2CMC.AccumulateReward(rewardDecay, rewardFunction)
    
    maxTimeStep = 150
    sampleTrajectory = A2CMC.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal) 

    approximatePolicy = A2CMC.ApproximatePolicy(actionSpace)
    approximateValue = A2CMC.approximateValue
    trainCritic = A2CMC.TrainCriticMonteCarloTensorflow(accumulateReward) 
    estimateAdvantage = A2CMC.EstimateAdvantageMonteCarlo(accumulateReward) 
    trainActor = A2CMC.TrainActorMonteCarloTensorflow(actionSpace) 
    
    numTrajectory = 5
    maxEpisode = 1
    actorCritic = A2CMC.OfflineAdvantageActorCritic(numTrajectory, maxEpisode, render)

    # Generate models.
    learningRateActor = 1e-4
    learningRateCritic = 3e-4
    hiddenNeuronNumbers = [128, 256, 512, 1024]
    hiddenDepths = [2, 4, 8]
    generateModel = GenerateActorCriticModel(numStateSpace, numActionSpace, learningRateActor, learningRateCritic)
    modelDict = {(n, d): generateModel(d, round(n/d)) for n, d in it.product(hiddenNeuronNumbers, hiddenDepths)}

    print("Generated graphs")
    # Train.
    actorCritic = A2CMC.OfflineAdvantageActorCritic(numTrajectory, maxEpisode, render)
    modelTrain = lambda  actorModel, criticModel: actorCritic(actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic,
            approximateValue, estimateAdvantage, trainActor)
    trainedModelDict = {key: modelTrain(model[0], model[1]) for key, model in modelDict.items()}

    print("Finished training")
    # Evaluate
    modelEvaluate = Evaluate(numTrajectory, approximatePolicy, sampleTrajectory, rewardFunction)
    meanEpisodeRewards = {key: modelEvaluate(model[0], model[1]) for key, model in trainedModelDict.items()}

    print("Finished evaluating")
    # Visualize
    independentVariableNames = ['NeuroTotalNumber', 'layerNumber']
    draw(meanEpisodeRewards, independentVariableNames)

if __name__ == "__main__":
    main()
