import numpy as np
from src.environment.game_env import SuperHeroEnv
from src.model.dqn_agent import DQNAgent
from src.utils.config import config

def train_dqn():
    env = SuperHeroEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)

    episodes = config["episodes"]
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(config["max_steps"]):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            agent.train(config["batch_size"])
            if done:
                break

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        # Update target network periodically
        if episode % config["target_update_freq"] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

if __name__ == "__main__":
    train_dqn()
