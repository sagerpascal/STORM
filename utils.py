import torch
import os
import numpy as np
import random
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN
import wandb


def seed_np_torch(seed=20010105):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, logger_args):
        run = wandb.init(**logger_args)
        self.tag_step = {}
        self._run = run

    @property
    def run_name(self):
        return self._run.name

    def log(self, tag, value):
        if "video" in tag:
            # Ensure value is a numpy array of shape [T, C, H, W], normalized to [0, 255], dtype=uint8
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            wandb.log({tag: wandb.Video(value, fps=15, format="mp4")})
        elif "images" in tag:
            # value is expected to be a tensor or numpy array of shape [N, C, H, W]
            images = [wandb.Image(img) for img in value]
            wandb.log({tag: images})
        elif "hist" in tag:
            # value can be a list or numpy array
            wandb.log({tag: wandb.Histogram(value)})
        else:
            wandb.log({tag: value})

    def log_dict(self, dict):
        wandb.log(dict)


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
