import numpy as np


class GreedyQAgent:
    def __init__(self, alpha=0.8, gamma=0.2, epsilon=0.0, actions=2, strict=False):
        self.q_map = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.strict = strict

    @property
    def is_strict(self):
        return self.strict

    def update_experience(self, cur_state, new_state, action, reward):
        old = self.q_map.get((cur_state, action), 0.0)
        self.q_map[cur_state, action] = old + self.alpha * (reward + self.gamma * self.max_in_q(new_state) - old)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(self.actions))
        if self.q_map.get((state, 0), 0.0) == self.q_map.get((state, 1), 0.0):
            return -1
        return self.argmax_in_q(state)

    def max_in_q(self, state):
        maximum = -1
        for i in range(self.actions):
            maximum = max(maximum, self.q_map.get((state, i), 0.0))
        return maximum

    def argmax_in_q(self, state):
        if self.q_map.get((state, 0), 0) == self.q_map.get((state, 1), 0):
            return np.random.choice([0, 1])
        if self.q_map.get((state, 0), 0) > self.q_map.get((state, 1), 0):
            return 0
        else:
            return 1