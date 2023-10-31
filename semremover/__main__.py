import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from .models import SemanticObjectRemover
from .models.utils import package_path, save_image


default_labels = [
     'grass', 'rug', 'field', 'sand', 'land', 'floor', 'windowpane', 'glass'
]
input_path = "input/input.jpg"

lama_ckpt = package_path("../models/weights/big-lama")

lama_config = package_path("models/config/lama_default.yaml")

maskformer_ckpt = "facebook/maskformer-swin-large-ade"

label_file = package_path("models/config/ade20k_labels.json")

output = "outputs/"

dilate_kernel_size = 15

output_type = ".png"

def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--floor", type=int, default=0,
        help="0: remove furniture, 1: create floor",
    )
    return parser


parser = create_parser()
args = parser.parse_args(sys.argv[1:])

sem_obj_remover = SemanticObjectRemover(lama_ckpt, lama_config, maskformer_ckpt, label_file)

if args.floor == 0:
     inpainted_image = sem_obj_remover.remove_objects_from_image(input_path, default_labels, dilate_kernel_size)
     save_image(inpainted_image, "ouput.png", output, output_type)
else:
     inpainted_image = sem_obj_remover.create_mask_from_image(input_path, default_labels, dilate_kernel_size)
     save_image(inpainted_image, "ouput.png", output, output_type)

