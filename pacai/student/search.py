"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    # Get first position and check if it is the goal
    node = problem.startingState()
    if problem.isGoal(node):
        return []

    # otherwise build a stack and get sucessor tuples
    frontier = Stack()
    suc = problem.successorStates(node)

    # list for already visited nodes
    reached = []

    # put successor tuples with empty path lists onto queue
    for children in suc:

        listofn = []
        pair = (children, listofn)
        frontier.push(pair)

        # print('Just pushed: '+str(pair))
        reached.append(children[0])

    # while the frontier has something on it
    while (not frontier.isEmpty()):

        # get the (tuple, list) pair
        pair = frontier.pop()

        # isolate tuple, list and position
        tup = pair[0]
        listo = pair[1]
        node = tup[0]

        # check if you have reached the goal
        if problem.isGoal(node):

            # append the last action to path list then return path list
            lista = listo.copy()
            lista.append(tup[1])

            return lista

        # for all sucessor tuples of the current position
        for child in problem.successorStates(node):

            # You have reached a position not seen before
            if child[0] not in reached:

                # copy path list, append new action, and push new tuple and path list to queue
                lista = listo.copy()
                lista.append(tup[1])
                npair = (child, lista)
                frontier.push(npair)

                # update visited posiitons list
                reached.append(child[0])


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    # Get first position and check if it is the goal
    node = problem.startingState()
    if problem.isGoal(node):
        return []

    # otherwise build a queue and get sucessor tuples
    frontier = Queue()
    suc = problem.successorStates(node)

    # list for already visited nodes
    reached = []

    # put successor tuples with empty path lists onto queue
    for children in suc:

        listofn = []
        pair = (children, listofn)
        frontier.push(pair)

        # print('Just pushed: '+str(pair))
        reached.append(children[0])
    # variable for debugging
    #  n=0

    # while the frontier has something on it
    while (not frontier.isEmpty()):

        # get the (tuple, list) pair
        pair = frontier.pop()

        # isolate tuple, list and position
        tup = pair[0]    # (state, action, cost)
        listo = pair[1]  # list of actions
        node = tup[0]    # state (x, y) pair

        # for all sucessor tuples of the current position
        for child in problem.successorStates(node):

            # You have reached the goal state
            if problem.isGoal(child[0]):

                # for some reason append the last action twice to path list then return path list
                lista = listo.copy()
                lista.append(tup[1])

                return lista

            # You have reached a position not seen before
            if child[0] not in reached:

                # copy path list, append new action, and push new tuple and path list to queue
                lista = listo.copy()
                lista.append(tup[1])
                npair = (child, lista)
                frontier.push(npair)

                # update visited posiitons list
                reached.append(child[0])

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    # Get first position and check if it is the goal
    node = problem.startingState()
    if problem.isGoal(node):
        return []

    # otherwise build a priority queue and get sucessor tuples
    frontier = PriorityQueue()
    suc = problem.successorStates(node)

    # list for already visited nodes
    reached = []

    # put successor tuples with empty path lists onto queue
    for children in suc:

        # construct sucessor
        listofn = []
        pair = (children, listofn)

        # priority is a sum of cost and path length
        frontier.push(pair, (len(listofn) + children[2]))

        # push successor onto queue
        reached.append(children[0])

    # while the frontier has something on it
    while (not frontier.isEmpty()):

        # get the (tuple, list) pair
        pair = frontier.pop()

        # isolate tuple, list and position
        tup = pair[0]
        listo = pair[1]
        node = tup[0]

        # check if you have reached the goal
        if problem.isGoal(node):

            # append the last action to path list then return path list
            lista = listo.copy()
            lista.append(tup[1])
            return lista

        # for all sucessor tuples of the current position
        for child in problem.successorStates(node):

            # You have reached a position not seen before
            if child[0] not in reached:

                # copy path list, append new action, and push new tuple and path list to queue
                lista = listo.copy()
                lista.append(tup[1])
                npair = (child, lista)

                # priority is a sum of cost and path length
                frontier.push(npair, (len(lista) + child[2]))

                # update visited posiitons list
                reached.append(child[0])

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # Get first position and check if it is the goal
    node = problem.startingState()
    if problem.isGoal(node):
        return []

    # otherwise build a priority queue and get sucessor tuples
    frontier = PriorityQueue()
    suc = problem.successorStates(node)

    # list for already visited nodes
    reached = []

    # put successor tuples with empty path lists onto queue
    for children in suc:

        # construct sucessor
        listofn = []
        pair = (children, listofn)

        # priotity is a sum of the heuristic, cost, and path length
        frontier.push(pair, (len(listofn) + children[2] + heuristic(children[0], problem)))

        # push successor onto queue
        reached.append(children[0])

    # while the frontier has something on it
    while (not frontier.isEmpty()):

        # get the (tuple, list) pair
        pair = frontier.pop()

        # isolate tuple, list and position
        tup = pair[0]
        listo = pair[1]
        node = tup[0]

        # check if you have reached the goal
        if problem.isGoal(node):

            # append the last action to path list then return path list
            lista = listo.copy()

            # print('\nReached goal, length'+str(len(lista))+' list is: '+str(lista)+'\n')
            return lista

        # for all sucessor tuples of the current position
        for child in problem.successorStates(node):

            # You have reached a position not seen before
            if child[0] not in reached:

                # copy path list, append new action, and push new tuple and path list to queue
                lista = listo.copy()
                lista.append(tup[1])
                npair = (child, lista)

                # priotity is a sum of the heuristic, cost, and path length
                frontier.push(npair, (len(lista) + child[2] + heuristic(child[0], problem)))

                # update visited posiitons list
                reached.append(child[0])
