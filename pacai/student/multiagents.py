# from cmath import inf
# from operator import le
import random

from pacai.core.directions import Directions
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Score variable for return value
        score = 0

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # Get some useful information
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]

        # Loop through the positions of the ghosts to calculate distance
        for x, y in ghostPositions:
            # dist is triangle distance of ghost to pacman
            dist = (abs(x - newPosition[0]) + abs(y - newPosition[1])) / 2
            # if one is very close, decrement score by a factor of dist^3
            if dist < 2:
                score -= dist * dist * dist

        # If the next action is stop, decrement score by a huge amount
        if action == 'Stop':
            score -= 900

        # Loop through positions of food to calculate distance
        for x, y in oldFood:
            dist = (abs(x - newPosition[0]) + abs(y - newPosition[1])) / 2
            # As dist gets larger, the addition to score gets smaller
            if dist != 0:
                score += 1 / dist

        # Return the score accumulated added with actual game score
        return (successorGameState.getScore() + score)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Minimax function, recursively called by min, max, and getAction, returns value
    def minimax(self, state, depth, agent):

        # Check if state is win/lose state, or if you have reached terminal depth
        if state.isLose() or state.isWin() or depth == self.getTreeDepth():

            # return evaluationFunction value of state
            func = self.getEvaluationFunction()
            return func(state)

        # If agent is 0, do max because agent is pacman
        if agent == 0:
            return self.max(state, depth, agent)

        # otherwise agent is 1 to (numAgents-1), do min because agent is ghost
        else:
            return self.min(state, depth, agent)

    # Min function for minimax, called recursively by minimax, and calls minimax rescursively
    def min(self, state, depth, agent):

        # Set min to a very high number
        mini = 999999
        # Get number of agents
        numAgents = state.getNumAgents()
        # Get legal actions for agent calling min
        legal = state.getLegalActions(agent)

        # Remove 'STOP' action from possible actions
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))

        # Go through possible legal actions
        for actions in legal:

            # Get successor state using agent and action
            succ = state.generateSuccessor(agent, actions)

            # Get minimax value for successor state, increment agent and modulo with numAgents
            new = self.minimax(succ, depth, ((agent + 1) % numAgents))

            # If minimax value for successor state less than min, replace min with it
            if new is not None and new < mini:
                mini = new

        # return min
        return mini

    # Max function for minimax, called recursively by minimax, and calls minimax rescursively
    def max(self, state, depth, agent):

        # Increment depth, because each time max is called, it is pacmans turn again
        depth += 1
        # Set max to a very low number
        maxi = -999999
        # Get legal actions for agent calling max
        legal = state.getLegalActions(agent)
        # Get number of agents
        numAgents = state.getNumAgents()

        # Remove 'STOP' action from possible actions
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))

        # Go through possible legal actions
        for actions in legal:

            # Get successor state using agent and action
            succ = state.generateSuccessor(agent, actions)

            # Get minimax value for successor state, increment agent and modulo with numAgents
            new = self.minimax(succ, depth, ((agent + 1) % numAgents))

            # If minimax value for successor state more than max, replace max with it
            if new is not None and new > maxi:
                maxi = new

        # return max
        return maxi

    # Returns best action determined by minimax
    def getAction(self, state):

        # Get legal actions for pacman when it first calls getAction
        legal = state.getLegalActions(0)

        # Remove 'STOP' action from possible actions
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))

        # Set max variable to track which action is best
        max = 0
        # Default return direction is STOP
        ret = Directions.STOP

        # Go through possible legal actions
        for actions in legal:

            # Get successor state using agent and action
            succ = state.generateSuccessor(0, actions)

            # Kickoff minimax recursion for a possible action
            new = self.minimax(succ, 0, 0)

            # If max variable is less than the minimax value, replace max and set the return string
            if new is not None and new > max:
                max = new
                ret = actions

        # Return best action determined by minimax
        return ret

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Exactly the same as minimax above, only 2 extra args passed in
    def minimax(self, state, depth, agent, alpha, beta):
        if state.isLose() or state.isWin() or depth == self.getTreeDepth():
            func = self.getEvaluationFunction()
            return func(state)
        if agent == 0:
            return self.max(state, depth, agent, alpha, beta)
        else:
            return self.min(state, depth, agent, alpha, beta)

    # Almost exactly the same as min in class above, only changes are commented
    def min(self, state, depth, agent, alpha, beta):
        mini = 999999
        numAgents = state.getNumAgents()
        legal = state.getLegalActions(agent)
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))
        for actions in legal:
            succ = state.generateSuccessor(agent, actions)
            new = self.minimax(succ, depth, ((agent + 1) % numAgents), alpha, beta)

            # If new value greater than beta, just return beta
            if new > beta:
                return beta
            if new is not None and new < mini:
                mini = new

                # Set beta when smallest value is found
                beta = new
        return mini

    # Almost exactly the same as max in class above, only changes are commented
    def max(self, state, depth, agent, alpha, beta):
        depth += 1
        maxi = -999999
        legal = state.getLegalActions(agent)
        numAgents = state.getNumAgents()
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))
        for actions in legal:
            succ = state.generateSuccessor(agent, actions)
            new = self.minimax(succ, depth, ((agent + 1) % numAgents), alpha, beta)

            # If new value less than alpha, return alpha
            if new < alpha:
                return alpha
            if new is not None and new > maxi:
                maxi = new

                # Set alpha when largest value is found
                alpha = new
        return maxi

    # Basically exact same as getAction for minimax, just a different call to minimax
    def getAction(self, state):
        legal = state.getLegalActions(0)
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))
        max = 0
        ret = Directions.STOP
        for actions in legal:
            succ = state.generateSuccessor(0, actions)
            new = self.minimax(succ, 0, 0, -999999, 999999)
            if new is not None and new > max:
                max = new
                ret = actions
        return ret

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Exactly the same as minimax function from Minimax class
    def minimax(self, state, depth, agent):
        if state.isLose() or state.isWin() or depth == self.getTreeDepth():
            func = self.getEvaluationFunction()
            return func(state)
        if agent == 0:
            return self.max(state, depth, agent)
        else:
            return self.min(state, depth, agent)

    # Basically exactly the same as mn function from Minimax class, changes commented
    def min(self, state, depth, agent):

        # Set variable to be averaged
        mini = 0
        numAgents = state.getNumAgents()
        legal = state.getLegalActions(agent)
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))

        # Set variable to calculate average
        i = len(legal)
        for actions in legal:
            succ = state.generateSuccessor(agent, actions)
            new = self.minimax(succ, depth, ((agent + 1) % numAgents))

            # Compound to variable to be averaged using calculated minimax function
            mini += new

        # Return the average by dividing the sum with the number of possible actions
        return mini / i

    # Exactly the same as max function from Minimax class
    def max(self, state, depth, agent):
        depth += 1
        maxi = -999999
        legal = state.getLegalActions(agent)
        numAgents = state.getNumAgents()
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))
        for actions in legal:
            succ = state.generateSuccessor(agent, actions)
            new = self.minimax(succ, depth, ((agent + 1) % numAgents))
            if new is not None and new > maxi:
                maxi = new
        return maxi

    # Exactly the same as getAction function from Minimax class
    def getAction(self, state):
        legal = state.getLegalActions(0)
        for action in legal:
            if action == Directions.STOP:
                legal.pop(legal.index(action))
        max = 0
        ret = Directions.STOP
        for actions in legal:
            succ = state.generateSuccessor(0, actions)
            new = self.minimax(succ, 0, 0)
            if new is not None and new > max:
                max = new
                ret = actions
        return ret


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: Very similar to previous evaluation function, just uses currentGameState
    as opposed to successorGameState, also does a simpler calulation to add and subtract from
    the evaluation
    """
    # Score variable for return value
    score = 0

    # Useful information you can extract.
    # newPosition = successorGameState.getPacmanPosition()
    # oldFood = currentGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

    # Get some useful information
    newPosition = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]

    # Loop through the positions of the ghosts to calculate distance
    for x, y in ghostPositions:
        dist = (abs(x - newPosition[0]) + abs(y - newPosition[1])) / 2
        # if one is very close, decrement score by a factor of dist
        if dist < 2:
            score -= dist

    # Loop through positions of food to calculate distance
    for x, y in oldFood:
        dist = (abs(x - newPosition[0]) + abs(y - newPosition[1])) / 2
        # As dist gets larger, the addition to score gets smaller
        if dist != 0:
            score += 1 / dist

    # Return the score accumulated added with actual game score
    return (currentGameState.getScore() + score)

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
