from collections import OrderedDict
import cv2
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from episode import Episode
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.grid import Grid


class MultiCategorical():

    def __init__(self, logits):


        assert logits.ndim == 4 or logits.ndim == 3
        self.variables = []

        if logits.ndim == 4:
            for i in range(logits.shape[2]):
                self.variables.append(Categorical(logits=logits[:, :, i]))
        else:
             for i in range(logits.shape[1]):
                self.variables.append(Categorical(logits=logits[:, i]))

        self.dim = logits.ndim

    def sample(self):
        return torch.cat([c.sample().unsqueeze(self.dim - 2) for c in self.variables], dim=self.dim - 2)
    
    def log_prob(self, values):
        assert values.ndim == self.dim - 1 and values.shape[self.dim - 2] == len(self.variables), f"{values.ndim}, {self.dim - 1}, {values.shape[self.dim - 2]}, {len(self.variables)}"
        t = torch.cat([self.variables[i].log_prob(values[:, :, i]).unsqueeze(-1) for i in range(values.shape[-1])], dim=-1)
        return torch.sum(t, dim=-1)
    
    def entropy(self):
        return sum([c.entropy() for c in self.variables]) / len(self.variables)


def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)

    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class EpisodeDirManager:
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames, vscode_codec=False):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3
    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


def get_top(agent_dir, agent_pos, short_side, long_side):
    x, y = agent_pos
    if agent_dir == 0:
        top = (x, max(y - short_side, 0))
    elif agent_dir == 1:
        top = (max(x - short_side, 0), y)
    elif agent_dir == 2:
        top = (max(x - long_side, 0), max(y - short_side, 0))
    elif agent_dir == 3:
        top = (max(x - short_side, 0), max(y - long_side, 0))
    
    return top

def update_full_view(env, view):
    agent_dir = env.agent_dir
    agent_pos = env.agent_pos

    x, y = agent_pos

    view_size = view.shape[0]

    short_side = view_size // 2
    long_side = view_size - 1

    y_top, x_top = get_top(agent_dir, agent_pos, short_side, long_side)

    end_x = env.size - x_top
    end_y = env.size - y_top


    if agent_dir == 0:
        start_y = short_side - y if y - short_side < 0 else 0
        start_x = 4 - x if x - long_side < 0 else 0
    elif agent_dir == 1:
        start_y = 0
        start_x = short_side - x if x - short_side < 0 else 0
    elif agent_dir == 2:
        start_y = short_side - y if y - short_side < 0 else 0
        start_x = 0
    elif agent_dir == 3:
        start_y = long_side - y if y - long_side < 0 else 0
        start_x = short_side - x if x - short_side < 0 else 0

    cropped_view = view[start_y:end_y, start_x:end_x]

    full_view = env.grid.encode()

    width, height, _ = cropped_view.shape

    full_view[y_top:y_top+width, x_top:x_top+height] = cropped_view

    return full_view


def full_obs(env):
    full_grid = env.grid.encode()
    full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
        [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
    )

    return full_grid[None, ...]


def decode_full_img(repr, agent_pos, agent_dir):
    d, e = Grid.decode(repr)
    return d.render(
        16,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        # highlight_mask=False,
    )