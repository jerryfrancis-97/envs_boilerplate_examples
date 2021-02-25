import gym
import torch
from agent import TRPOAgent
import simple_driving
import time
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="agent.pth",\
                            help="enter pathof the model")
    args = parser.parse_args()

    #Extract params
    path = args.path

    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)

    agent.load_model(path)
    
    env = gym.make('SimpleDriving-v0')
    ob = env.reset()

    
    no_of_episodes = 100

    for i in range(no_of_episodes):
        timesteps = 0
        cum_reward = 0
        done = False
    
        while not done:
            action = agent(ob)
            ob, reward, done, _ = env.step(action)
            
            # print(f"Episode : {i}; Timestep : {timesteps}; Reward : {reward}")           
            timesteps+=1
            cum_reward+=reward
            env.render()
        
        print(f"Episode {i} completed in {timesteps} timsteps, total reward: {cum_reward}")
        ob = env.reset()
        time.sleep(1/30)


if __name__ == '__main__':
    main()
