import os
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


class LaMa:
    def __init__(self, ckpt, config):
        self.model, self.config, self.device = self.load_config(ckpt, config)

    @staticmethod
    def load_config(ckpt, config):
        predict_config = OmegaConf.load(config)
        predict_config.model.path = ckpt
        device = torch.device("cuda")
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)
        return model, predict_config, device

    def create_batch(self, image: np.ndarray, mask: np.ndarray):
        batch = dict()
        batch['image'] = image.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1
        return batch

    @torch.no_grad()
    def inpaint(self, image: np.ndarray, mask: np.ndarray):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        image = torch.from_numpy(image).float().div(255.)
        mask = torch.from_numpy(mask).float()

        batch = self.create_batch(image, mask)
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]

        batch = self.model(batch)
        cur_res = batch[self.config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res