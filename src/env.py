from gymnasium import register, make

register(
    reward_threshold=0.82,
    max_episode_steps=100,
    kwargs={
        "map_name": "4x4",
        "is_slippery": False
    },
    entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
    id="NonSlipFrozenLake-v0",
)

env = make("NonSlipFrozenLake-v0")
