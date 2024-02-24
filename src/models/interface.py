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
        self.agent_loss_scale = 10

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
            self.n_flatten = self.cnn(torch.rand((1, 3, 19, 19))).shape[1]


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
        masked_logits[..., 0] %= 11
        masked_logits[..., 1] %= 5
        masked_logits[..., 2] %= 3

        return masked_logits
        
    def compute_loss(self, batch: Batch, gamma: float, lambda_: float, entropy_weight: float, burn_in: int, reset_horizon: int, **kwargs: Any) -> LossWithIntermediateLosses:
        assert not self.use_original_obs

        true_obs = batch["observations"][:, :, 0].squeeze(2)
        modified_obs = rearrange(batch["observations"][:, :, 1].squeeze(2), 'b t w h c -> b t (w h c)')

        tokens = rearrange(true_obs, 'b t w h c -> (b t) (w h c)').int()
        outputs = self(tokens)

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
            )[:, :-1]

        values = values[:, :-1]

        d = MultiCategorical(logits=logits[:, :-1])
        # alpha_ = max(self.alpha ** (self.step / 80), 0.25)
        # agent_scale = self.agent_loss_scale * (1 - alpha_)
        agent_scale = 1
    
        log_probs = d.log_prob(modified_obs[:, :-1])
        loss_actions = -1 * agent_scale * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - agent_scale * entropy_weight * d.entropy().mean()
        loss_values = agent_scale * F.mse_loss(values, lambda_returns)

        # input = rearrange(outputs.logits_actions[:, :-1], 'b s l d -> (b s) l d')
        # target = rearrange(outputs.true_obs[:, :-1], 'b s l -> (b s) l')
        # self.step += 1
        # loss_curriculum = alpha_ * sum([self.cross_entropy_loss(input[:, idx], target[:, idx]) for idx in range(input.size(1))])

        # return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy, loss_curriculum=loss_curriculum)
        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy)


    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int, reset_horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        episode_observations = batch["observations"]
        device = episode_observations.device
        wm_env = InterfaceWorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_modified_obs = []
        all_user_actions = []
        all_true_obs = []

        self.reset(n=episode_observations.size(0))

        episode_observations = torch.clamp(tokenizer.encode_decode(episode_observations[:, :-1], should_preprocess=True, should_postprocess=True), 0, 1) if episode_observations.size(1) > 1 else None
        obs_tokens = wm_env.reset_from_initial_observations(episode_observations[:, 0])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            outputs_interface_ac = self(obs_tokens)
            latent_obs = MultiCategorical(logits=outputs_interface_ac.logits_actions).sample()
            # wm_env.obs_tokens = latent_obs

            with torch.no_grad():
                embedded_tokens = tokenizer.embedding(latent_obs)     # (B, K, E)
                z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(wm_env.num_observations_tokens)))
                rec = tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
                modified_obs = torch.clamp(rec, 0, 1)
                # modified_obs = wm_env.decode_obs_tokens()
                all_modified_obs.append(modified_obs)

                outputs_user_ac = self.user_actor_critic(modified_obs)
                user_action = Categorical(logits=outputs_user_ac.logits_actions).sample()


            if k > 0 and k % reset_horizon == 0:
                wm_env.reset_from_initial_observations(episode_observations[:, k])

            all_true_obs.append(obs_tokens)
            obs_tokens, reward, done, _ = wm_env.step(user_action, should_predict_next_obs=(k < horizon - 1))
            all_actions.append(latent_obs.unsqueeze(1))
            all_user_actions.append(user_action)
            all_logits_actions.append(outputs_interface_ac.logits_actions.unsqueeze(1))
            all_values.append(outputs_interface_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()
        return ImagineOutput(
            observations=torch.stack(all_modified_obs, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            user_actions=torch.cat(all_user_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            # values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            values=torch.cat(all_values, dim=1),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
            true_obs=torch.stack(all_true_obs, dim=1)                               # (B, T, K)
        )


class InterfaceMLP(InterfaceActorCritic):
    def __init__(self, user_actor_critic: ActorCritic, obs_vocab_size: int, wm_token_dim: int, embed_dim: int, num_layers: int, use_original_obs: bool = False, **kwargs) -> None:
        
        super().__init__(user_actor_critic, use_original_obs)

        self.non_lin = nn.ELU()
        self.critic_layers = [nn.Linear(wm_token_dim, embed_dim), self.non_lin]
        self.actor_layers = [nn.Linear(wm_token_dim, embed_dim), self.non_lin]
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
    
    def forward(self, tokens: torch.FloatTensor) -> InterfaceOutput:

        logits_actions = self.actor(tokens.float())
        means_values = self.critic(tokens.float())

        logits_actions = rearrange(logits_actions, 'b (h w) -> b h w', h=tokens.size(1))

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

        self.full_linear = nn.Sequential(nn.Linear(self.n_flatten, config.embed_dim), nn.ReLU())

        print(self.n_flatten)
        
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
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if self.keys_values_wm is None else self.keys_values_wm.size

        print(full_tokens.shape)
        x_full = self.cnn(full_tokens.float())
        print(x_full.shape)
        x_full = self.full_linear(x_full)
        if self.keys_values_wm.size + num_steps > self.transformer.config.max_tokens:
           self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.n, max_tokens=self.config.max_tokens)

        sequences = self.embedder(tokens, num_steps, prev_steps)
        sequences += self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, None)
        logits_actions = self.actor_linear(torch.concat([x, x_full], dim=-1))

        means_values = self.critic_linear(x)
        means_values = means_values.sum(-2)
        return InterfaceOutput(logits_actions, means_values)
