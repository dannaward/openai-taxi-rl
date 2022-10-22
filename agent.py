import random

import numpy as np


class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6

        self.epsilon = 0
        self.gamma = 1.0
        self.alpha = 0.2

    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if self.mode == "q_learning":
            return self.q_learning_select_action(state)
        elif self.mode == "mc_control":
            return self.mc_control_select_action(state)

    def step(self, state, action, reward, next_state, done):

        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if self.mode == "q_learning":
            self.q_learning_step(state, action, reward, next_state, done)
        elif self.mode == "mc_control":
            self.mc_control_step(state, action, reward, next_state, done)

    def q_learning_select_action(self, state):
        action = self.epsilon_greedy(self.Q, state, self.epsilon)
        return action

    def mc_control_select_action(self, state):
        pass

    def q_learning_step(self, state, action, reward, next_state, done):
        next_action = self.select_action(state)

        current = self.Q[state][action]
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + (self.gamma * Qsa_next)
        new_value = current + (self.alpha * (target - current))
        self.Q[state][action] = new_value

    def mc_control_step(self, state, action, reward, next_state, done):
        pass

    def epsilon_greedy(self, Q, state, epsilon):
        if random.random() > epsilon:
            return np.argmax(Q[state])
        else:
            return random.choice(np.arange(self.n_actions))