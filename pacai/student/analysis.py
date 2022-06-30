"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I made the noise extremely small, 0.01 so
    that the agent would be encouraged to act non stochastically
    and move across the bridge
    """

    answerDiscount = 0.9
    answerNoise = 0.01

    return answerDiscount, answerNoise

def question3a():
    """
    Lowered noise, and high penalty for living to encourage agent to move along cliff,
    but finish as soon as possible
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = -5.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Lowered noise, and a fairly high living penalty,
    forcing agent away from the cliff, but encouraging it to
    stop as quick as possible
    """

    answerDiscount = 0.1
    answerNoise = 0.01
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Lowered noise, agent acts non stochastically, moves along cliff to high reward
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Agent behaves like this by default, no change necessary
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Agent behaves like this by default, no change necessary
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    50 iterations not enough, not a possible thing
    """
    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
