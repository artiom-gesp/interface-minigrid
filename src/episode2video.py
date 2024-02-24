from utils import make_video
import torch
from einops import rearrange
import numpy as np

if __name__ == "__main__":

    path = "/local/home/argesp/iris/outputs/2024-02-04/18-01-49/media/episodes/train/best_episode_19_epoch_5.pt"
    v = torch.load(path)
    make_video("test.mp4", 15, np.array(v['observations']))
    # make_video("test.mp4", 15, np.array(rearrange(v['observations'], 'f c h w -> f h w c')))
