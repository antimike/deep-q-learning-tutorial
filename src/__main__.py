from .env import env
from .agent import QAgent

if __name__ == "__main__":
    agent = QAgent(
        env,
        discount_factor=0.93,
        learning_rate=0.83,
        max_actions=99,
        max_episodes=1000,
    )
    agent.train()
    agent.test()
