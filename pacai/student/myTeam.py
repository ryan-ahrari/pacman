from pacai.agents.capture.defense import DefensiveReflexAgent
from pacai.agents.capture.offense import OffensiveReflexAgent
from pacai.core.directions import Directions

class firstAgent(OffensiveReflexAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.index = index

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        maxVal = max(values)

        bestActions = [a for a, v in zip(actions, values) if v == maxVal]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bDist = float('inf')
            bestAction = ''
            for i in actions:
                successor = self.getSuccessor(gameState, i)
                dist = self.getMazeDistance(self.start, successor.getAgentPosition(self.index))
                if dist > bDist:
                    continue
                else:
                    bestAction = i
                    bDist = dist
            return bestAction
        return (bestActions[0])

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        if (action == Directions.STOP):
            features['stop'] = 1

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        cList = self.getCapsules(successor)
        if len(cList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in cList])
            features['capsule'] = minDistance

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        else:
            features['invaderDistance'] = -1

        return features

    def getWeights(self, gameState, action):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        state = self.getCurrentObservation()
        # print('Current observation: ',state)
        agent = state.isOnRedTeam(self.index)
        dict = {}
        if ((agent and state.isOnRedSide(myPos)) or (not agent and state.isOnBlueSide(myPos))):
            dict['successorScore'] = 500
            dict['distanceToFood'] = -100
            dict['invaderDistance'] = -1
            dict['stop'] = -100
            dict['capsule'] = -50
        else:
            dict['successorScore'] = 1000
            dict['distanceToFood'] = -15
            dict['invaderDistance'] = 7
            dict['stop'] = -100
            dict['capsule'] = -8
        return dict

class secondAgent(DefensiveReflexAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.index = index

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]

        maxVal = max(values)

        bestActions = [a for a, v in zip(actions, values) if v == maxVal]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bDist = float('inf')
            bestAction = ''
            for i in actions:
                successor = self.getSuccessor(gameState, i)
                dist = self.getMazeDistance(self.start, successor.getAgentPosition(self.index))
                if dist > bDist:
                    continue
                else:
                    bestAction = i
                    bDist = dist
            return bestAction
        return (bestActions[0])

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        foodList = self.getFoodYouAreDefending(successor).asList()
        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToProtectFood'] = minDistance

        cList = self.getCapsulesYouAreDefending(successor)
        if len(cList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in cList])
            features['capsule'] = minDistance

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        else:
            features['invaderDistance'] = -1

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -500,
            'onDefense': 100,
            'distanceToProtectFood': -5,
            'capsule': -5,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -11
        }

def createTeam(firstIndex, secondIndex, isRed):
    # first = 'pacai.agents.capture.offense.OffensiveReflexAgent',
    # second = 'pacai.agents.capture.defense.DefensiveReflexAgent'):
    # first = firstAgent(0),
    # second = secondAgent(0)):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # firstAgent = firstAgent()
    # secondAgent = secondAgent()

    return [
        firstAgent(firstIndex), secondAgent(secondIndex)
    ]
