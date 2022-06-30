"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
# from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
from pacai.core.distance import euclidean
from pacai.core.distance import manhattan
# from pacai.student.search import depthFirstSearch
from pacai.student.search import uniformCostSearch


class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """
    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

    # Starting state function, returns a position (x, y) and a list of bools for each corner
    def startingState(self):

        # get starting position (x, y)
        start = self.startingPosition

        # list of bools for each corner, preset to False
        lista = [False, False, False, False]

        # if the starting position is a corner, mark that corner as True
        if start in self.corners:
            lista[self.corners.index(start)] = True

        # now return the tuple (position, cornersVisitedlist)
        return (start, lista)

    # Check to see if the pair is a goal state
    def isGoal(self, pair):

        # if you're not in the starting position, and all the corners have been visited, return True
        if ((pair == self.startingPosition) is False and (pair[1] == [True, True, True, True])):
            return True

        # return false
        return False

    # Return the successor states for the pair passed in
    def successorStates(self, pair):

        # Copy the list of visited corners, and construct a list for the successors
        lista = pair[1].copy()
        successors = []

        # For all actions: East, West, North, South
        for action in Directions.CARDINAL:

            # get the coordinates from the first item in the pair
            x, y = pair[0]
            dx, dy = Actions.directionToVector(action)

            # Here we ensure if a corner is being reached, it is marked as so in the bool list
            if pair[0] in self.corners:
                lista[self.corners.index(pair[0])] = True

            # set the next coordinates, and position
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            nextState = (nextx, nexty)

            # The next coordinate is not a wall
            if (not hitsWall):

                # Construct the successor.
                state = (nextState, lista)
                successors.append((state, action, 1))

        self._numExpanded += 1

        # return list of successors
        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """
    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # Make a corners list of all the corner positions, and get the passed in position
    corners = list(problem.corners)
    curr = state[0]

    # Remove corners already visited from corners positions list
    for cor in state[1]:
        if cor is True:
            corners.pop(state[1].index(cor))

    # If we are currently at a corner, also remove it from the list
    if curr in corners:
        corners.pop(corners.index(state[0]))

    # heuristic variable
    heur = 0

    # while there are corners in the positions list
    while(corners):

        # Make index and minimum variables for heuristic tracking
        index = 0
        min = 10000

        # For the corners in the positions list
        for corns in corners:

            # get the euclidean distance from current position to corner
            new = euclidean(curr, corns)

            # if this corner is the closest corner seen, update that information
            if new < min:
                index = corners.index(corns)
                min = new

        # reset current position to be in closest corner, and remove corner from the positions list
        curr = corners[index]
        corners.pop(index)

        # add the distance from current position to the closest corner to heuristic
        heur += min

    # return heuristic
    return heur


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """
    # get position, and foodGrid from state, make foodGrid into list fg
    position, foodGrid = state
    fg = foodGrid.asList()

    # Heuristic variable
    max = 0

    # For all food in the list fg
    for food in fg:

        # get euclidean and manhattan distance for node to food
        euc = euclidean(food, position)
        man = manhattan(food, position)

        # new is average of distances
        new = ((man + euc) / 2)

        # if new is larger than max, replace max with new
        if new > max:
            max = new

    # return the furthest food distance
    return max

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """
        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # get problem, and return UCS of problem to closest food
        problem = AnyFoodSearchProblem(gameState)
        return uniformCostSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):

        # if there is a food at position (x, y) = state, return True
        if(self.food[state[0]][state[1]] is True):
            return True

        # Otherwise return false
        return False


class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
