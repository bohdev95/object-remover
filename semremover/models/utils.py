import os
from pathlib import Path

from PIL import Image


def package_path(path: str):
    package_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    return os.path.join(package_directory, path)


def save_image(image: Image, image_path: str, output_dir: str, output_type: str):
    img_stem = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{img_stem}{output_type}")
    image.save(output_path)
