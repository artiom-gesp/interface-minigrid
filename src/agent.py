from pathlib import Path
from einops import rearrange

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.interface import InterfaceActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict, MultiCategorical, make_video
from envs.world_model_env import InterfaceWorldModelEnv
import numpy as np

i = 0
class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token


class InterfaceAgent(nn.Module):
    def __init__(self, user_actor_critic: ActorCritic, interface_actor_critic: InterfaceActorCritic, reset_wm_every: int):
        super().__init__()
        self.user_actor_critic = user_actor_critic
        self.actor_critic = interface_actor_critic
        
        self.reset_wm_every = reset_wm_every
        self.step = 0
        self.representations = []

    @property
    def device(self):
        return self.user_actor_critic.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_interface: bool = True, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        # if load_interface:
        #     self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'interface_actor_critic'))

    def reset(self, obs, is_end = False):
        if is_end and self.representations:
            torch.cat(self.representations)
            repr = rearrange(torch.cat(self.representations), 'f c h w -> f h w c').to(torch.uint8)
            global i
            make_video(f"./media/test_{i}.mp4", 15, np.array(repr))
            self.representations = []
            i += 1
        self.step = 0
    
    def act(self, obs: torch.IntTensor, full_obs: torch.IntTensor) -> torch.LongTensor:

        # if self.user_actor_critic.use_original_obs:
        #     input_ac = obs
        # else:
        #     input_ac = torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        _, c, w, h = obs.size()

        tokens = rearrange(obs, 'b w h c -> b (w h c)').int()
        full_tokens = rearrange(full_obs, 'b w h c -> b c w h').int()
        outputs_interface_ac = self.actor_critic(tokens, full_tokens)
        modified_obs = MultiCategorical(logits=outputs_interface_ac.logits_actions).sample()
        modified_obs = rearrange(modified_obs, 'b (c w h) -> b c w h', c=c, w=w, h=h)
        masked_modified_obs = self.actor_critic.mask_output(modified_obs)
        # self.wm_env.obs_tokens = latent_obs

        # logits_actions = self.user_actor_critic(modified_obs).logits_actions[:, -1] / temperature
        # act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        # self.wm_env.step(act_token)

        # self.step += 1
        action, _ = self.user_actor_critic.predict(masked_modified_obs.cpu().numpy())
        return action, modified_obs
