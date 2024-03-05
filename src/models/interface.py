from dataclasses import dataclass
from typing import Any, Optional, Union
import sys

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import InterfaceWorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses, init_weights, MultiCategorical
from models.actor_critic import ActorCritic
from models.kv_caching import KeysValues
from models.transformer import Transformer, TransformerConfig
from models.slicer import Embedder, Head


@dataclass
class InterfaceOutput:
    logits_actions: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    user_actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    true_obs: torch.LongTensor
    ends: torch.BoolTensor


class InterfaceActorCritic(nn.Module):
    def __init__(self, user_actor_critic: ActorCritic, use_original_obs: bool = False) -> None:
        super().__init__()

        self.use_original_obs = use_original_obs
        self.user_actor_critic = user_actor_critic
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.step = 1
        self.alpha = 0.99
        self.agent_loss_scale = 0.1

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.n_full = self.cnn(torch.rand((1, 3, 19, 19))).shape[1]


    def __repr__(self) -> str:
        return "interface_actor_critic"

    def clear(self) -> None:
        self.user_actor_critic.hx, self.user_actor_critic.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        pass

    def prune(self, mask: np.ndarray) -> None:
        self.user_actor_critic.hx = self.user_actor_critic.hx[mask]
        self.user_actor_critic.cx = self.user_actor_critic.cx[mask]

    def mask_output(self, obs):
        
        masked_logits = obs.clone().detach()
        masked_logits[..., 0] %= 10
        masked_logits[..., 1] %= 6
        masked_logits[..., 2] %= 3

        return masked_logits
        
    def compute_loss(self, batch: Batch, gamma: float, lambda_: float, entropy_weight: float, burn_in: int, reset_horizon: int, **kwargs: Any) -> LossWithIntermediateLosses:
        assert not self.use_original_obs

        true_obs = batch["observations"][:, :, 0, :5, :5].squeeze(2)
        modified_obs = rearrange(batch["observations"][:, :, 1, :5, :5].squeeze(2), 'b t w h c -> b t (w h c)')
        full_obs = rearrange(batch["observations"][:, :, 2].squeeze(2), 'b t w h c -> (b t) c w h')

        tokens = rearrange(true_obs, 'b t w h c -> (b t) w h c').int()
        outputs = self(tokens, full_obs)

        b = true_obs.size(0)
        t = true_obs.size(1)
    
        values = rearrange(outputs.means_values, '(b t) 1 -> b t', b=b, t=t)
        logits = rearrange(outputs.logits_actions, '(b t) n d -> b t n d', b=b, t=t)
        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=batch["rewards"],
                values=values,
                ends=batch["ends"],
                gamma=gamma,
                lambda_=lambda_,
            )

        values = values

        d = MultiCategorical(logits=logits)

        # alpha_ = self.alpha ** (self.step / 20)
        alpha_ = max(self.alpha ** (self.step / 20), 0.0001)
        agent_scale = self.agent_loss_scale * (1 - alpha_)

    
        log_probs = d.log_prob(modified_obs)
        loss_actions = -1 * agent_scale * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = -1 * entropy_weight * d.entropy().mean()
        loss_values = agent_scale * F.mse_loss(values, lambda_returns)

        input = outputs.logits_actions

        target = rearrange(true_obs, 'b t w h c -> (b t) (w h c)').int()

        target = target.type(torch.LongTensor).to(input.device)
        self.step += 1

        loss_curriculum = alpha_ * sum([self.cross_entropy_loss(input[:, idx], target[:, idx]) for idx in range(input.size(1))])

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy, loss_curriculum=loss_curriculum)
        # return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy)


class InterfaceMLP(InterfaceActorCritic):
    def __init__(self, user_actor_critic: ActorCritic, obs_vocab_size: int, wm_token_dim: int, embed_dim: int, num_layers: int, use_original_obs: bool = False, **kwargs) -> None:
        
        super().__init__(user_actor_critic, use_original_obs)

        self.non_lin = nn.ELU()

        with torch.no_grad():
            self.n_partial = self.cnn(torch.rand((1, 3, 5, 5))).shape[1]

        self.convert = nn.Linear(self.n_full, self.n_partial)

        self.critic_layers = [nn.Linear(2 * self.n_partial, embed_dim), self.non_lin]
        self.actor_layers = [nn.Linear(2 * self.n_partial, embed_dim), self.non_lin]

        for _ in range(num_layers):
            self.critic_layers.append(nn.Linear(embed_dim, embed_dim))
            self.critic_layers.append(self.non_lin)
            self.actor_layers.append(nn.Linear(embed_dim, embed_dim))
            self.actor_layers.append(self.non_lin)
        self.critic_layers.append(nn.Linear(embed_dim, 1))
        self.actor_layers.append(nn.Linear(embed_dim, wm_token_dim * obs_vocab_size))
        
        self.critic = nn.Sequential(*self.critic_layers)
        self.actor = nn.Sequential(*self.actor_layers)



        self.apply(init_weights)



    def __repr__(self) -> str:
        return "interface_actor_critic_mlp"
    
    def forward(self, tokens: torch.LongTensor, full_tokens: torch.LongTensor) -> InterfaceOutput:
        
        tokens = rearrange(tokens, 'b w h c -> b c w h').int()

        x_full = self.convert(self.cnn(full_tokens.float()))
        x_partial = self.cnn(tokens.float())

        x_combined = torch.concat([x_partial, x_full], dim=-1)

        logits_actions = self.actor(x_combined)
        means_values = self.critic(x_combined)

        logits_actions = rearrange(logits_actions, 'b (h w) -> b h w', h=tokens.size(1) * tokens.size(2) * tokens.size(3))

        return InterfaceOutput(logits_actions, means_values)


class InterfaceTransformer(InterfaceActorCritic):
    def __init__(self, user_actor_critic: ActorCritic, obs_vocab_size: int, config: TransformerConfig, use_original_obs: bool = False) -> None:

        super().__init__(user_actor_critic, use_original_obs)

        self.obs_vocab_size = obs_vocab_size
        self.config = config
        self.transformer = Transformer(config)

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern


        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(obs_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.full_linear = nn.Sequential(nn.Linear(self.n_full, config.embed_dim), nn.ReLU())

        self.critic_linear = nn.Sequential(
                nn.Linear(2 * config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 1)
            )
        self.actor_linear = nn.Sequential(
                nn.Linear(2 * config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )

        self.apply(init_weights)



    def __repr__(self) -> str:
        return "interface_actor_critic_transformer"
    
    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None):

        super().reset(n, burnin_observations, mask_padding)
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
        self.n = n
    
    def forward(self, tokens: torch.LongTensor, full_tokens: torch.LongTensor) -> InterfaceOutput:

        tokens = rearrange(tokens, 'b w h c -> b (w h c)').int()

        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if self.keys_values_wm is None else self.keys_values_wm.size

        x_full = self.cnn(full_tokens.float())
        x_full = self.full_linear(x_full)
        if self.keys_values_wm.size + num_steps > self.transformer.config.max_tokens:
           self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.n, max_tokens=self.config.max_tokens)

        sequences = self.embedder(tokens, num_steps, prev_steps)
        sequences += self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, None)
        x_combined = torch.concat([x, x_full.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        logits_actions = self.actor_linear(x_combined)

        means_values = self.critic_linear(x_combined)
        means_values = means_values.sum(-2)
        return InterfaceOutput(logits_actions, means_values)
