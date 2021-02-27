from torch.utils.tensorboard import SummaryWriter
import time


#logger
class Logger(object):

    def __init__ (self, path="./tmp", experiment_name= f"'Experiment_at_{int(time.time())}"):
        self.writer = SummaryWriter(f"{path}/{experiment_name}")
    
    def per_episode_log(self, per_episode_reward, episode_length, per_episode_returns, episode_no):
        self.writer.add_scalar("charts/Episode reward", per_episode_reward, episode_no)
        self.writer.add_scalar("charts/Length of an episode", episode_length, episode_no)
        self.writer.add_scalar("charts/Episodic returns", per_episode_returns, episode_no)
    
    def per_iteration(self,average_reward_per_iter, num_episode, iteration):
        self.writer.add_scalar("charts/Avg Reward per iter", average_reward_per_iter, iteration)
        # self.writer.add_scalar("charts/KL Divergence diff from threshold", kl_diff, iteration)
        # self.writer.add_scalar("charts/KL Divergence per iter ", kl_value, iteration)
        self.writer.add_scalar("charts/Number of episodes per iter", num_episode, iteration)

    def close(self):
        self.writer.close()

logger = Logger()