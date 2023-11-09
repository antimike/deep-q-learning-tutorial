from gymnasium import Env
from typing import Callable, Union
import numpy as np

from .utils import calculate_bellman_update, Number
from .env import State


class QAgent:

    def __init__(self, env: Env, discount_factor: float, learning_rate: float,
                 max_episodes: int, max_actions: int):
        self.env = env
        self.gamma = discount_factor
        self.alpha = learning_rate
        self._max_episodes = max_episodes
        self._max_actions = max_actions
        self._decay_rate = 1.0 / max_episodes

        self.exploration_rate = 1.0
        self._Q = np.zeros((env.observation_space.n, env.action_space.n))

    def _get_idx(self, state: Union[State, int]):
        try:
            return state[0]
        except TypeError:
            return state

    def _exploit(self, state: State):
        return np.argmax(self._Q[self._get_idx(state), :])

    def _explore(self, state: State):
        return self.env.action_space.sample()

    def _update_decay_rate(self):
        if self.exploration_rate > self._decay_rate:
            self.exploration_rate -= self._decay_rate

    def get_action(self, state: State):
        if np.random.random() > self.exploration_rate:
            return self._exploit(state)
        else:
            return self._explore(state)

    def _apply_update(self, prev_state: State, action: int, new_state_idx: int,
                      reward: Number) -> None:
        prev_state_idx = self._get_idx(prev_state)
        self._Q[prev_state_idx, action] = calculate_bellman_update(
            self._Q[prev_state_idx, action], np.max(self._Q[new_state_idx, :]),
            reward, self.gamma, self.alpha)

    def _run_episode(self, callback: Callable[[State, int, int],
                                              None]) -> float:
        state = self.env.reset()
        total_reward = 0

        for _ in range(self._max_actions):
            action = self.get_action(state)
            new_state, reward, terminated, truncated, info = self.env.step(
                action)
            callback(state, action, new_state, reward)
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        return total_reward

    def train(self) -> None:
        for _ in range(self._max_episodes):
            # TODO: Short-circuit if episode reward is optimal for k consecutive runs
            self._run_episode(self._apply_update)
            self._update_decay_rate()

    def test(self):

        def show_transition(prev_state, action, new_state, reward):
            print(f"==================")
            print(
                f"(prev_state, action, new_state) = {(prev_state, action, new_state)}"
            )
            self.env.render()
            print(f"==================")

        reward = self._run_episode(show_transition)
        print()
        print(f"total_reward = {reward}")
