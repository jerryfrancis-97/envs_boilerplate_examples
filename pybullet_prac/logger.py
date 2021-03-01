from torch.utils.tensorboard import SummaryWriter
import time


#logger
class Logger(object):

    def __init__ (self, path, experiment_name= "Experiment"):
        self.writer = SummaryWriter(f"runs/{experiment_name}_for_{path[2:]}_at_{int(time.time())}")
        self.total_no_of_episodes = 0
    
    def per_episode_log(self, per_episode_reward, episode_length, per_episode_returns, episode_no):
        self.total_no_of_episodes += 1
        self.writer.add_scalar("charts/Episode reward", per_episode_reward, self.total_no_of_episodes)
        self.writer.add_scalar("charts/Length of an episode", episode_length, self.total_no_of_episodes)
        self.writer.add_scalar("charts/Episodic returns", per_episode_returns, self.total_no_of_episodes)
    
    def per_iteration(self,average_reward_per_iter, kl_delta, kl_value, num_episode, iteration):
        # print(num_episode,"episodes")
        self.writer.add_scalar("charts/Avg Reward per iter", average_reward_per_iter, iteration)
        if kl_delta != 0:
            # print("True")
            # print("Values: ", kl_delta, kl_value)
            kl_diff = kl_value - kl_delta
            self.writer.add_scalar("charts/KL Divergence diff from threshold", kl_diff, iteration)
            self.writer.add_scalar("charts/KL Divergence per iter ", kl_value, iteration)
        # print("No. ofe pisods: ",num_episode)
        self.writer.add_scalar("charts/Number of episodes per iter", num_episode, iteration)

    def close(self):
        self.writer.close()

if __name__ == '__main__':
    logger = Logger()