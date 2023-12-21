# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        for _ in range(self.iterations):
            new_values = util.Counter()
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                max_value = -float('inf')
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    cur_value = self.computeQValueFromValues(state, action)
                    if cur_value >= max_value:
                        max_value = cur_value
                new_values[state] = max_value
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        ans = 0
        TAP = self.mdp.getTransitionStatesAndProbs(state, action)
        for pair in TAP:
            nextState, prob = pair[0], pair[1]
            ans += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return ans

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        arg, max_value = None, -float('inf')
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        for action in actions:
            cur_value = self.computeQValueFromValues(state, action)
            if cur_value >= max_value:
                arg, max_value = action, cur_value
        return arg

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        iterate = self.iterations
        while iterate > 0:
            for state in states:
                if iterate == 0:
                    break
                iterate -= 1
                if self.mdp.isTerminal(state):
                    continue
                max_value = -float('inf')
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    cur_value = self.computeQValueFromValues(state, action)
                    if cur_value >= max_value:
                        max_value = cur_value
                self.values[state] = max_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        #compute the predecessors
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            predecessors[state] = set()
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                TAP = self.mdp.getTransitionStatesAndProbs(state, action)
                for pair in TAP:
                    nextState, prob = pair[0], pair[1]
                    if prob > 0:
                        predecessors[nextState].add(state)
        #initialize a pq and build it
        pq = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                cur_value = self.values[state]
                actions = self.mdp.getPossibleActions(state)
                max_value = -float('inf')
                for action in actions:
                    cur_qvalue = self.computeQValueFromValues(state, action)
                    max_value = cur_qvalue if cur_qvalue > max_value else max_value
                diff = abs(max_value - cur_value)
                pq.push(state, -diff)            
        #iterate the pq
        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = self.updateValue(state)
            for prv_state in predecessors[state]:
                cur_value = self.values[prv_state]
                actions = self.mdp.getPossibleActions(prv_state)
                max_value = -float('inf')
                for action in actions:
                    cur_qvalue = self.computeQValueFromValues(prv_state, action)
                    max_value = cur_qvalue if cur_qvalue > max_value else max_value
                diff = abs(max_value - cur_value)
                if diff > self.theta:
                    pq.update(prv_state, -diff)

    def updateValue(self, state):
        cur_value = self.values[state]
        actions = self.mdp.getPossibleActions(state)
        max_value = -float('inf')
        for action in actions:
            cur_qvalue = self.computeQValueFromValues(state, action)
            max_value = cur_qvalue if cur_qvalue > max_value else max_value        
        return max_value