#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

import re
from typing import Tuple

import numpy as np
from PIL import Image, ImageChops


def add_tint_to_rgb(image: Image.Image, tint_color: Tuple) -> Image.Image:
    r, g, b, alpha = image.split()
    tint = Image.new("RGB", image.size, tint_color)
    tinted_rgb = ImageChops.screen(tint.convert("RGB"), image.convert("RGB"))

    # Return the composite image with original alpha channel
    return Image.merge(
        "RGBA",
        (
            tinted_rgb.split()[0],
            tinted_rgb.split()[1],
            tinted_rgb.split()[2],
            alpha,
        ),
    )


def wrap_text(text: str, max_chars_per_line: int, split_on_period: bool = False) -> str:
    if split_on_period:
        text = text.split(".")[0]
    # Remove digits which are preceded by `_`.
    text = re.sub(r"_(\d+)", "", text)
    # Remove underscores and slashes
    text = text.replace("/", "_")
    text = text.replace(" ", "_")
    names = text.split("_")

    current_line = ""
    wrapped_text = []
    for name in names:
        name = name.strip()
        if len(current_line + name) <= max_chars_per_line:
            current_line += name + " "
        else:
            wrapped_text.append(current_line.strip())
            current_line = name + " "
    wrapped_text.append(current_line.strip())
    output_text = "\n".join(wrapped_text).strip()
    return output_text


def resize_icon_height(icon: Image.Image, target_height: float) -> Image.Image:
    width, height = icon.size
    scaling_factor = target_height / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_icon = icon.resize((new_width, new_height))
    return resized_icon


def load_and_resize_icon(path: str, target_height: float) -> Image.Image:
    icon = Image.open(path).convert("RGBA")
    return np.array(resize_icon_height(icon, target_height))
