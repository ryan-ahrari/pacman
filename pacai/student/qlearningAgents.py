from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util.probability import flipCoin
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: getQValue will get the QValue of ((x, y), 'action') state, action pair from
    the qValues dictionary, or set the QValue to 0.0 otherwise. getValue finds the best actions
    for the state using getPolicy, which in turn call getQValue to find the QValues of each next
    state. Similarly getValue will then use the action gotten from getPolicy to getQValue of the
    best action from the state passed to it. Finally the update function serves as the main method
    of updating qValues{}, and will also call getValue and getQValue.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        # choose best action according to getPolicy
        action = self.getPolicy(state)

        # if that action is not None, return the qValue of the state, action pair
        if action:
            value = self.getQValue(state, action)
            return value

        # else return 0.0
        return 0.0

    def update(self, state, action, nextState, reward):
        """
        According to reward, create a sample and then using alpha, and the previous
        qValue{(state, action)}, calculate the new qValue and set it, then return.
        """
        # calculate new qValue using oldValue, alpha, and the sample gathered
        al = self.getAlpha()
        sample = (reward + (self.getDiscountRate() * self.getValue(nextState)))
        self.qValues[(state, action)] = (((1 - al) * self.getQValue(state, action)) + (al * sample))

    def getAction(self, state):
        """
        Using probablity epsilon, either take the best action for a state, using
        getPolicy, or else take a random action.
        """
        # Choose the best action according to getPolicy
        policy = self.getPolicy(state)
        if policy:

            # Get a random action for state, and a boolean that is True with probabilty of epsilon
            rand = random.choice(self.getLegalActions(state))
            bool = flipCoin(self.getEpsilon())

            # Return either best action or random action
            if not bool:
                return policy
            return rand
        return None

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.getLegalActions(state)
        # if there are legal actions to take
        if actions:
            values = []
            acts = []

            # go through possible action and put them on lists
            for action in actions:
                values.append(self.getQValue(state, action))
                acts.append(action)

            # return best action
            best = max(values)
            return acts[values.index(best)]
        return None

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # Initialize weights
        self.weight = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen. Returns
        featureVector*weights dot product
        """
        # Making a dot-product sum
        sum = 0.0

        # Get features for state
        features = self.featExtractor.getFeatures(self, state, action)

        # Looping through the features
        for feat in features:

            # If theres something of value to add, add to the dot product
            if features[feat] != 0.0:
                sum += (features[feat] * self.weight.get(feat, 0.0))

        # return dot product
        return sum

    def update(self, state, action, nextState, reward):
        """
        According to reward, create a sample and then using alpha, and the previous
        weight{feature}, calculate the new weight and set it, then return.
        """
        # Get features for state
        features = self.featExtractor.getFeatures(self, state, action)

        # Get learning rate, alpha, 1 time
        al = self.getAlpha()

        # Looping through the features
        for feat in features:

            # Update feature weights
            corr = (reward + (self.getDiscountRate() * (self.getValue(nextState))))
            corr -= self.getQValue(state, action)
            weit = self.weight.get(feat, 0.0)
            self.weight[feat] = (weit + (al * corr * features[feat]))

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            print(self.weight)
            pass
