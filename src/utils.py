from typing import Union, TypeAlias

Number: TypeAlias = Union[int, float]


def calculate_bellman_update(q_prev: float, q_curr: float, reward: Number,
                             gamma: float, alpha: float):
    return q_prev + alpha * (reward + gamma * q_curr - q_prev)
