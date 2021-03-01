import gym
import torch
from agent import TRPOAgent, Data_recorder
import simple_driving
import time
import argparse
import pickle
import os
from logger import Logger


def main():

    """
    Experiment obs :
    1 iters 10K batch_size works better than 100 iters 5K batch size!
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size : default = 5000')
    parser.add_argument('--iterations', type=int, default=100,
                        help='iterations : default = 100')
    parser.add_argument('--max_episode_length', type=int, default=250,
                        help='max_episode_length : default = 250')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='to show working info : default = True')
    parser.add_argument('--seed', type=int, default=0,
                        help='to set seed : default = 0')
    args = parser.parse_args()

    # Extract params
    seed = args.seed
    batch_size = args.batch_size
    iterations = args.iterations
    max_episode_length = args.max_episode_length
    verbose = args.verbose

    # File structuring
    dir_path = "./agent_TRPO"+str(iterations)+"itrs_"+str(batch_size)+"batch_size"+str(seed)+"_seed"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    
    # Activate logger
    logger = Logger(dir_path)

    #Training part
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(policy=nn)
    recorder = Data_recorder()

    # agent.load_model("agent_.pth")
    training_recordings = agent.train("SimpleDriving-v0", logger, seed=seed, batch_size=batch_size, iterations=iterations,
                max_episode_length=max_episode_length, verbose=True)

    # Save data and model
    data_path = os.path.join(dir_path,"training_data.pkl")
    model_path = os.path.join(dir_path,"model.pth")

    recorder.save_recordings(training_recordings, data_path)
    agent.save_model(model_path)

    # Sample eval Run
    env = gym.make('SimpleDriving-v0')
    ob = env.reset()
    while True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)


if __name__ == '__main__':
    main()
