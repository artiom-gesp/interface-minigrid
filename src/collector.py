import random
import sys
from typing import List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent, InterfaceAgent
from dataset import EpisodesDataset
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import EpisodeDirManager, RandomHeuristic
import torch.nn.functional as F
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX


class Collector:
    def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager) -> None:
        self.env = env
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.obs = self.env.reset()
        self.episode_ids = [None] * self.env.num_envs
        self.heuristic = RandomHeuristic(self.env.num_actions)
        self.alpha = 0.99
        self.step = 1

    @torch.no_grad()
    def collect(self, agent: Union[Agent, InterfaceAgent], epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, *, num_steps: Optional[int] = None, num_episodes: Optional[int] = None, skip_frames: Optional[int] = 0):
        assert 0 <= epsilon <= 1

        assert (num_steps is None) != (num_episodes is None)
        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

        to_log = []
        steps, episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = [], [], [], []

        burnin_obs_rec, mask_padding = None, None
        # if set(self.episode_ids) != {None} and burn_in > 0:
        #     current_episodes = [self.dataset.get_episode(episode_id) for episode_id in self.episode_ids]
        #     segmented_episodes = [episode.segment(start=len(episode) - burn_in, stop=len(episode), should_pad=True) for episode in current_episodes]
        #     mask_padding = torch.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
        #     burnin_obs = torch.stack([episode.observations for episode in segmented_episodes], dim=0).float().div(255).to(agent.device)
        #     burnin_obs_rec = torch.clamp(agent.tokenizer.encode_decode(burnin_obs, should_preprocess=True, should_postprocess=True), 0, 1)

        agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=burnin_obs_rec, mask_padding=mask_padding)
        obs = torch.IntTensor(self.obs).to(agent.device)
        if isinstance(agent, InterfaceAgent):
            agent.reset()
        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout)
        while not should_stop(steps, episodes):
            
            act, modified_obs, f_obs = agent.act(obs, self.env.env, epsilon)
            padded_obs = F.pad(obs, (0, 0, 0, f_obs.size(2) - obs.size(2), 0, f_obs.size(1) - obs.size(1)))
            padded_mod_obs = F.pad(modified_obs, (0, 0, 0, f_obs.size(2) - obs.size(2), 0, f_obs.size(1) - obs.size(1)))
            observations.append(torch.cat([padded_obs.cpu(), padded_mod_obs.cpu(), f_obs.cpu()], dim=0).unsqueeze(0))

            self.obs, reward, done, _ = self.env.step(act)

            # alpha_ = max(self.alpha ** (self.step / 15), 0.0001)
            alpha_ = 1
            self.step += 1


            # reconstruction reward is added here for simplicity
            reward += alpha_ / (1 + torch.sqrt(torch.sum((modified_obs - obs) ** 2))).item()

            for _ in range(skip_frames):
                self.env.step(act)
                
 
            actions.append(act)
            rewards.append(reward)
            dones.append(done)

            new_steps = len(self.env.mask_new_dones)
            steps += new_steps
            pbar.update(new_steps if num_steps is not None else 0)

            # Warning: with EpisodicLifeEnv + MultiProcessEnv, reset is ignored if not a real done.
            # Thus, segments of experience following a life loss and preceding a general done are discarded.
            # Not a problem with a SingleProcessEnv.
            obs = torch.IntTensor(self.obs).to(agent.device)

            if self.env.should_reset():
                self.add_experience_to_dataset(observations, actions, rewards, dones)

                new_episodes = self.env.num_envs
                episodes += new_episodes
                pbar.update(new_episodes if num_episodes is not None else 0)

                for episode_id in self.episode_ids:
                    episode = self.dataset.get_episode(episode_id)
                    self.episode_dir_manager.save(episode, episode_id, epoch)
                    metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                    metrics_episode['episode_num'] = episode_id
                    metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                    to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})
                    returns.append(metrics_episode['episode_return'])

                self.obs = self.env.reset()
                self.episode_ids = [None] * self.env.num_envs
                agent.actor_critic.reset(n=self.env.num_envs)

                if isinstance(agent, InterfaceAgent):
                    agent.reset(is_end=True)
                observations, actions, rewards, dones = [], [], [], []

        # Add incomplete episodes to dataset, and complete them later.
        if len(observations) > 0:
            self.add_experience_to_dataset(observations, actions, rewards, dones)

        agent.actor_critic.clear()

        metrics_collect = {
            '#episodes': len(self.dataset),
            '#steps': sum(map(len, self.dataset.episodes)),
        }
        if len(returns) > 0:
            metrics_collect['return'] = np.mean(returns)
        metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)

        return to_log

    def add_experience_to_dataset(self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray]) -> None:
        
        # assert len(observations) == len(actions) == len(rewards) == len(dones)
        for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(
                observations=torch.ByteTensor(o).contiguous(),  # channel-first
                actions=torch.LongTensor(a),
                rewards=torch.FloatTensor(r),
                ends=torch.LongTensor(d),
                mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
            )
            if self.episode_ids[i] is None:
                self.episode_ids[i] = self.dataset.add_episode(episode)
            else:
                self.dataset.update_episode(self.episode_ids[i], episode)
