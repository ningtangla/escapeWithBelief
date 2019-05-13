import os
import numpy as np
import pandas as pd
import pygame as pg
import itertools as it
import random 
#np.random.seed(123)

class TransitionFunction():
    def __init__(self, resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChange, updateBeliefAndAttention, updatePhysicalStateByBelief):
        self.resetPhysicalState = resetPhysicalState
        self.resetBeliefAndAttention = resetBeliefAndAttention
        self.updatePhysicalState = updatePhysicalState
        self.transiteStateWithoutActionChange = transiteStateWithoutActionChange
        self.updateBeliefAndAttention = updateBeliefAndAttention
        self.updatePhysicalStateByBelief = updatePhysicalStateByBelief

    def __call__(self, oldState, action):
        if oldState is None:
            newPhysicalState = self.resetPhysicalState()
            newBeliefAndAttention = self.resetBeliefAndAttention(newPhysicalState)
            newState = [newPhysicalState, newBeliefAndAttention] 
        else:
            #oldState = self.updatePhysicalStateByBelief(oldState)
            oldPhysicalState, oldBeliefAndAttention = oldState
            
            #newBeliefAndAttention = self.updateBeliefAndAttention(oldBeliefAndAttention, oldPhysicalState)
            #newPhysicalState = self.updatePhysicalState(oldPhysicalState, action)
        
            newPhysicalState = self.updatePhysicalState(oldPhysicalState, action)
            
            stateBeforeNoActionChangeTransition = [newPhysicalState, oldBeliefAndAttention]
            physicalStateAfterNoActionChangeTransition, beliefAndAttentionAfterNoActionChangeTransition = self.transiteStateWithoutActionChange(stateBeforeNoActionChangeTransition) 
            newBeliefAndAttention = self.updateBeliefAndAttention(oldBeliefAndAttention, physicalStateAfterNoActionChangeTransition)

            newState = [physicalStateAfterNoActionChangeTransition, newBeliefAndAttention]
            newState = self.updatePhysicalStateByBelief(newState)
            #print(newBeliefAndAttention[0]['logP'])
            #__import__('ipdb').set_trace()
        return newState

class TransiteStateWithoutActionChange():
    def __init__(self, maxFrame, isTerminal, transiteMultiAgentMotion, render, renderOn):
        self.maxFrame = maxFrame
        self.isTerminal = isTerminal
        self.transiteMultiAgentMotion = transiteMultiAgentMotion
        self.render = render
        self.renderOn = renderOn
    def __call__(self, state):
        for frame in range(self.maxFrame):
            physicalState, beliefAndAttention = state 
            agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
            if self.isTerminal(state):
                break
            if self.renderOn == True:
                self.render(state)
            newAgentStates, newAgentActions = self.transiteMultiAgentMotion(agentStates, agentActions) 
            newPhysicalState = [newAgentStates, newAgentActions, timeStep, wolfIdAndSubtlety]
            stateAfterNoActionChangeTransition = [newPhysicalState, beliefAndAttention]
            state = stateAfterNoActionChangeTransition
        return state

class IsTerminal():
    def __init__(self, sheepId, minDistance):
        self.sheepId = sheepId
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        physicalState, beliefAndAttention = state
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        wolfId, wolfSubtlety = wolfIdAndSubtlety
        sheepPosition = agentStates[self.sheepId]
        wolfPosition = agentStates[wolfId]
        if np.sum(np.power(sheepPosition - wolfPosition, 2)) ** 0.5 <= self.minDistance:
            terminal = True
        return terminal   


class Render():
    def __init__(self, numAgent, screen, screenColor, sheepColor, wolfColor, circleSize, saveImage, saveImageFile):
        self.numAgent = numAgent
        self.screen = screen
        self.screenColor = screenColor
        self.sheepColor = sheepColor
        self.wolfColor = wolfColor
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageFile = saveImageFile
    def __call__(self, state):
        physicalState, beliefAndAttention = state 
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        
        hypothesisInformation, positionOldTimeDF = beliefAndAttention
        posteriorAllHypothesesBeforeNormalization = np.exp(hypothesisInformation['logP'])
        posteriorAllHypotheses = posteriorAllHypothesesBeforeNormalization / (np.sum(posteriorAllHypothesesBeforeNormalization))
        posteriorAllWolf = posteriorAllHypotheses.groupby(['wolfIdentity']).sum().values
        wolfColors = [self.wolfColor * wolfBelief for wolfBelief in posteriorAllWolf]
        circleColorList = [self.sheepColor] + wolfColors
        
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit
            self.screen.fill(self.screenColor)
            for i in range(self.numAgent):
                oneAgentState = agentStates[i]
                oneAgentPosition = np.array(oneAgentState)
                pg.draw.circle(self.screen, circleColorList[i], [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], self.circleSize)
                #pg.draw.line(self.screen, np.zeros(3), [np.int(positionOldTimeDF.loc[i].values[0]), np.int(positionOldTimeDF.loc[i].values[1])], [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], self.circleSize)
            pg.display.flip()
            currentDir = os.getcwd()
            parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
            saveImageDir=parentDir+'/src/data/'+self.saveImageFile
            if self.saveImage==True:
                filenameList = os.listdir(saveImageDir)
                pg.image.save(self.screen,saveImageDir+'/'+str(len(filenameList))+'.png')
            pg.time.wait(1)

class MctsRender():
    def __init__(self, numAgent, screen, surfaceWidth, surfaceHeight, screenColor, sheepColor, wolfColor, distractorColor, circleSize, saveImage, saveImageFile):
        self.numAgent = numAgent
        self.screen = screen
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.screenColor = screenColor
        self.sheepColor = sheepColor
        self.wolfColor = wolfColor
        self.distractorColor = distractorColor
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageFile = saveImageFile
    def __call__(self, currNode, nextNode, backgroundScreen):

        surfaceToDraw = pg.Surface((self.surfaceWidth, self.surfaceHeight))
        surfaceToDraw.fill(self.screenColor)
        #surfaceToDraw.set_colorkey(np.zeros(3))
        surfaceToDraw.set_alpha(80)
        if backgroundScreen == None:
            backgroundScreen = pg.Surface((self.surfaceWidth, self.surfaceHeight))
            backgroundScreen.fill(self.screenColor)
            self.screen.fill(self.screenColor)
        surfaceToDraw.blit(backgroundScreen, (0,0))
        
        pg.display.flip()
        pg.time.wait(1)
        state = list(currNode.id.values())[0] 
        physicalState, beliefAndAttention = state 
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        wolfId, wolfSubtlety = wolfIdAndSubtlety 
        
        nextState = list(nextNode.id.values())[0]
        nextPhysicalState, nextBeliefAndAttention = nextState 
        nextAgentStates, nextAgentActions, nextTimeStep, nextWolfIdAndSubtlety = nextPhysicalState
        
        lineWidth = nextNode.num_visited + 1 
        circleColorList = [self.sheepColor] + [self.distractorColor] * (self.numAgent - 1)
        circleColorList[wolfId] = self.wolfColor
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit
                
            for i in range(self.numAgent):
                oneAgentState = agentStates[i]
                oneAgentNextState = nextAgentStates[i]
                oneAgentPosition = np.array(oneAgentState)
                oneAgentNextPosition = np.array(oneAgentNextState)
                if i == 0:
                    line = pg.draw.line(surfaceToDraw, np.zeros(3), [np.int(oneAgentPosition[0]), np.int(oneAgentPosition[1])], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], lineWidth)
                    circles = pg.draw.circle(surfaceToDraw, circleColorList[i], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], self.circleSize)
                if i == wolfId:
                    circles = pg.draw.circle(surfaceToDraw, circleColorList[i], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], self.circleSize)
            
            self.screen.blit(surfaceToDraw, (0, 0)) 
            pg.display.flip()
            pg.time.wait(1)
            backgroundScreenToReturn = self.screen.copy()
            
            if self.saveImage==True:
                currentDir = os.getcwd()
                parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
                saveImageDir=parentDir+'/src/data/'+self.saveImageFile
                filenameList = os.listdir(saveImageDir)
                pg.image.save(self.screen,saveImageDir+'/'+str(len(filenameList))+'.png')
        return self.screen

if __name__ == '__main__':
    a = TransitionFunction
    __import__('ipdb').set_trace()
