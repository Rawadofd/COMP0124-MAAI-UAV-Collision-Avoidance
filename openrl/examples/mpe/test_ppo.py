# ======== Modified by Xiao for multi-drones collision scenarios ======== #
# test_ppo.py
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.envs.wrappers import GIFWrapper # used to generate gif
def test():
    # Create MPE environment.
    env = make("simple_spread", env_num=9)
    # Use GIFWrapper to generate gifs.
    env = GIFWrapper(env, "ppo.gif")
    agent = Agent(Net(env))  # Create an intelligent agent.
    # Load the trained model.
    agent.load('./ppo_agent/')
    # Begin to test.
    obs, _ = env.reset()
    while True:
        action, _ = agent.act(obs)
        obs, r, done, info = env.step(action)
        if done.any():
            break
    env.close()
if __name__ == "__main__":
    test()