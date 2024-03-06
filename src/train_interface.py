"""
replay buffer must have: 
- discrete latent space + action taken world agent
- action taken by interface agent
- reward received by interface agent (:= reward received by world model agent)
- done received by interface agent (:= done received by world model agent)
"""


from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.world_object import Goal

import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from agent import Agent, InterfaceAgent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.interface import InterfaceMLP, InterfaceTransformer
from utils import configure_optimizer, EpisodeDirManager, set_seed


from stable_baselines3 import PPO

torch.set_num_threads(10)

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.nb_step = 0
        

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        goal_x, goal_y = self.goal_pos
        agent_x, agent_y = self.env.agent_pos
        reward_pos = 1 / (1 + np.sqrt((goal_x - agent_x)**2 + (goal_y - agent_y)**2))
        reward_time = self.nb_step / self.env.max_steps

        self.nb_step += 1

        reward = reward_pos - reward_time
        if terminated:
            reward += sum([x / 500 for x in range(1, self.env.max_steps)]) + 1

        return obs, reward, terminated, truncated, info


    def reset(self):

        obs = self.env.reset()
        
        self.nb_step = 0
        for i in range(self.env.grid.width):
            for j in range(self.env.grid.height):
                tile = self.env.grid.get(i, j)
                if isinstance(tile, Goal):
                    self.goal_pos = (i, j)
        
        return obs


class InterfaceTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)


        
        def env_fn():
            env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array", max_steps=500, agent_view_size=5)
            env = ImgObsWrapper(env)
            env = CustomReward(env)

            return env

        def create_env(cfg_env, num_envs):
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

        if self.cfg.evaluation.should:
            test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = train_env if self.cfg.training.should else test_env

        user_actor_critic = PPO.load("/local/home/argesp/interface-minigrid/model_best_5", device=self.device)
        if cfg.interface.general.interface_type == 'mlp':
            interface_actor_critic = InterfaceMLP(user_actor_critic, obs_vocab_size=10, **self.cfg.interface.mlp, **cfg.actor_critic)
        elif cfg.interface.general.interface_type == 'transformer':
            interface_actor_critic = InterfaceTransformer(user_actor_critic, obs_vocab_size=10, config=instantiate(cfg.interface.transformer))
        else:
            raise NotImplementedError(f"Interface type {cfg.interface.general.interface_type} not implemented")
        self.interface_agent = InterfaceAgent(user_actor_critic, interface_actor_critic, cfg.training.actor_critic.reset_horizon).to(self.device)
        print(f'{sum(p.numel() for p in self.interface_agent.actor_critic.parameters())} parameters in interface_agent.actor_critic')

        self.optimizer_interface = torch.optim.Adam(self.interface_agent.actor_critic.parameters(), lr=cfg.training.learning_rate)

        # if cfg.initialization.path_to_checkpoint is not None:
        #     self.interface_agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

    def run(self) -> None:

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.interface_agent, epoch, **self.cfg.collection.train.config)
                to_log += self.train_agent(epoch)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                self.test_dataset.clear()
                to_log += self.test_collector.collect(self.interface_agent, epoch, **self.cfg.collection.test.config)
                try:
                    to_log += self.eval_agent(epoch)
                except:
                    print("for some reason testing failed uwu")

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()

    def train_agent(self, epoch: int) -> None:
        self.interface_agent.train()
        self.interface_agent.zero_grad()

        component = self.interface_agent.actor_critic
        optimizer = self.optimizer_interface
        sequence_length = 1 + self.cfg.training.actor_critic.imagine_horizon
        sample_from_start = False
        steps_per_epoch = self.cfg.training.actor_critic.steps_per_epoch
        batch_num_samples = self.cfg.training.actor_critic.batch_num_samples
        grad_acc_steps = self.cfg.training.actor_critic.grad_acc_steps
        max_grad_norm = self.cfg.training.actor_critic.max_grad_norm

        metrics = {}

        cfg_actor_critic = self.cfg.training.actor_critic

        if epoch > cfg_actor_critic.start_after_epochs:
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)

            for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout):
                optimizer.zero_grad()
                for _ in range(grad_acc_steps):
                    batch = self.train_dataset.sample_batch(batch_num_samples, sequence_length, sample_from_start)
                    batch = self._to_device(batch)

                    losses = component.compute_loss(batch, **cfg_actor_critic) / grad_acc_steps
                    loss_total_step = losses.loss_total
                    loss_total_step.backward()
                    loss_total_epoch += loss_total_step.item() / steps_per_epoch

                    for loss_name, loss_value in losses.intermediate_losses.items():
                        intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

                optimizer.step()

            metrics = {f'interface/train/total_loss': loss_total_epoch, **intermediate_losses}
        self.interface_agent.actor_critic.eval()

        return [{'epoch': epoch, **metrics}]

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.interface_agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}
    
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_actor_critic.start_after_epochs:
            mode_str = 'imagination'
            batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.imagine_horizon, sample_from_start=True)

            to_log = []
            for i, (o, a, r, d) in enumerate(zip(batch["observations"].cpu(), batch["actions"].cpu(), batch["rewards"].cpu(), batch["ends"].long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
                episode = Episode(o, a, r, d, torch.ones_like(d))
                episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * batch["observations"].size(0) + i
                self.episode_manager_imagination.save(episode, episode_id, epoch)

                metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                metrics_episode['episode_num'] = episode_id
                metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.interface_agent.world_model.act_vocab_size)
                to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

            return to_log


        return [metrics_tokenizer, metrics_world_model]


    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.interface_agent.actor_critic.state_dict(), self.ckpt_dir / 'interface_last.pt')
        torch.save(epoch, self.ckpt_dir / 'interface_epoch.pt')
        torch.save(self.optimizer_interface.state_dict(), self.ckpt_dir / 'interface_optimizer.pt')
        ckpt_dataset_dir = self.ckpt_dir / 'dataset'
        ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
        self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
        if self.cfg.evaluation.should:
            torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'interface_epoch.pt') + 1
        latest_interface = torch.load(self.ckpt_dir / 'interface_last.pt', map_location=self.device)
        self.interface_agent.actor_critic.load_state_dict(latest_interface)
        ckpt_opt = torch.load(self.ckpt_dir / 'interface_optimizer.pt', map_location=self.device)
        self.optimizer_interface.load_state_dict(ckpt_opt)
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        wandb.finish()

