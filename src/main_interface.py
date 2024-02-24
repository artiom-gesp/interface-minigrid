import hydra
from omegaconf import DictConfig

from train_interface import InterfaceTrainer as Trainer


@hydra.main(config_path="../config", config_name="interface_trainer")
def main(cfg: DictConfig):

    print(cfg.wandb.project)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
