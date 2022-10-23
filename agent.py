import random
import numpy as np


class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6

        self.epsilon = 0

        if mode == "q_learning":
            self.alpha = 0.2
            self.gamma = 1.0
        elif mode == "mc_control":
            self.alpha = 0.05
            self.gamma = 0.95
            self.episode = []
            self.epsilon_var = 1
            self.G = 0

    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if self.mode == "mc_control":
            self.epsilon = 1 / self.epsilon_var

        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.n_actions))

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

    def q_learning_step(self, state, action, reward, next_state, done):
        current = self.Q[state][action]

        if next_state:
            next_q = np.max(self.Q[next_state])
        else:
            next_q = 0

        updated_value = current + (self.alpha * (reward + self.gamma * next_q - current))

        self.Q[state][action] = updated_value

    def mc_control_step(self, state, action, reward, next_state, done):
        if done:
            for state, action, reward in reversed(self.episode):
                current = self.Q[state][action]

                self.G = reward + self.gamma * self.G
                updated_value = current + self.alpha * (self.G - current)

                self.Q[state][action] = updated_value

            self.episode.clear()
            self.epsilon_var += 0.001
            self.G = 0

        else:
            self.episode.append((state, action, reward))
