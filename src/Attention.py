import numpy as np 
import pandas as pd 

def modifyPrecisionForUntracked(attentionStatus,precisionPerSlot,precisionForUntracked):
    attentionStatus = attentionStatus
    if attentionStatus==0:
        return precisionForUntracked/precisionPerSlot
    else:
        return attentionStatus

def modifyDecayForUntracked(attentionStatus,memoryratePerSlot,memoryrateForUntracked):
    attentionStatus = attentionStatus
    if attentionStatus==0:
        return (1 - memoryratePerSlot)/((1 - memoryrateForUntracked)+0.00000001)
    else:
        return attentionStatus

class CalDistancePriorOnAttentionSlot():
    def __init__(self, minDistance, maxDistance, numStandardErrorInDistanceRange):
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        self.numStandardErrorInDistanceRange = numStandardErrorInDistanceRange
        self.midDistance = (self.minDistance + self.maxDistance) / 2
        self.rangeDistance = (self.maxDistance - self.minDistance)
    def __call__(self, distances):
        centeredDistance = (distances - self.midDistance) / (self.rangeDistance / self.numStandardErrorInDistanceRange)
        prior = (np.tanh(-centeredDistance) + 1) / 2
        #adjustedPrior = np.array([probability if distance > self.minDistance else 0 for probability, distance in zip(prior, distances)])
        adjustedPrior = prior
        return adjustedPrior

class AttentionSwitch():
    def __init__(self,attentionLimitation, calDistancePriorOnAttentionSlot):
        self.attentionLimitation=attentionLimitation
        self.calDistancePriorOnAttentionSlot = calDistancePriorOnAttentionSlot
    def __call__(self,hypothesisInformation, positionDF):
        newHypothesisInformation=hypothesisInformation.copy()
        hypothesis = hypothesisInformation.index
        
        wolfObjNums = hypothesis.get_level_values('wolfIdentity')
        sheepObjNums = hypothesis.get_level_values('sheepIdentity')
        wolfLoc = positionDF.loc[wolfObjNums]
        sheepLoc = positionDF.loc[sheepObjNums]
        
        distanceBetweenWolfAndSheep = np.sqrt(np.sum(np.power(wolfLoc.values - sheepLoc.values, 2), axis = 1))
        distancePriorOnHypothesisAttention = self.calDistancePriorOnAttentionSlot(distanceBetweenWolfAndSheep)
        probabilityOnHypothesisAttention = np.exp(hypothesisInformation['logP']) * distancePriorOnHypothesisAttention
        posteriorOnHypothesisAttention = probabilityOnHypothesisAttention/probabilityOnHypothesisAttention.sum()
        posteriorOnAttentionSlot = posteriorOnHypothesisAttention.groupby(['wolfIdentity','sheepIdentity']).sum().values
        
        numOtherCondtionBeyondPair = hypothesisInformation.groupby(['wolfIdentity','sheepIdentity']).size().values[0]
        newAttentionStatus=list(np.random.multinomial(self.attentionLimitation, posteriorOnAttentionSlot))*numOtherCondtionBeyondPair
        newHypothesisInformation['attentionStatus']=np.array(newAttentionStatus)
        return newHypothesisInformation

class AttentionToPrecisionAndDecay():
    def __init__(self,precisionPerSlot,precisionForUntracked,memoryratePerSlot,memoryrateForUntracked):
        self.precisionPerSlot = precisionPerSlot
        self.precisionForUntracked = precisionForUntracked
        self.memoryratePerSlot = memoryratePerSlot
        self.memoryrateForUntracked = memoryrateForUntracked
    def __call__(self,attentionStatus):
        attentionForPrecision = list(map(lambda x: modifyPrecisionForUntracked(x,self.precisionPerSlot,self.precisionForUntracked),attentionStatus))
        attentionForDecay = list(map(lambda x: modifyDecayForUntracked(x,self.memoryratePerSlot,self.memoryrateForUntracked),attentionStatus))
        precisionHypothesis = np.multiply(self.precisionPerSlot , attentionForPrecision)+0.00000001
        decayHypothesis = 1 - np.divide((1 - self.memoryratePerSlot),np.add(attentionForDecay,0.00000001))
        # decayHypothesis = 1 - np.divide((1 - self.memoryratePerSlot),(np.power(2,np.array(attentionForDecay)-1))+0.00000001)
        precisionHypothesisDF = pd.DataFrame(precisionHypothesis,index=attentionStatus.index,columns=['perceptionPrecision'])
        decayHypothesisDF = pd.DataFrame(decayHypothesis,index=attentionStatus.index,columns=['memoryDecay'])
        return precisionHypothesisDF, decayHypothesisDF
