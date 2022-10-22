import numpy as np


class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6

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
            pass
        elif self.mode == "mc_control":
            pass

        return np.random.choice(self.n_actions)
        # return action

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

    def q_learning(self, state):
        pass

    def mc_control(self, state):
        pass
