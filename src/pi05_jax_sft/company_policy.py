from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openpi import transforms


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        # CHW → HWC
        image = image.transpose(1, 2, 0)
    return image


@dataclass(frozen=True)
class CompanyWristInputs(transforms.DataTransformFn):
    """Maps dataset sample to pi0.5 Observation format.

    Supports either the legacy single-wrist dataset or the Tonglu three-view
    dataset. OpenPI's pi0.5 image slots are fixed, so the project mapping is:

    - chest_image -> base_0_rgb
    - wrist_image -> left_wrist_0_rgb
    - top_image   -> right_wrist_0_rgb
    """

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(data["wrist_image"])
        zero_image = np.zeros_like(wrist_image)
        chest_image = _parse_image(data["chest_image"]) if "chest_image" in data else zero_image
        top_image = _parse_image(data["top_image"]) if "top_image" in data else zero_image

        inputs: dict = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": chest_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": top_image,
            },
            "image_mask": {
                "base_0_rgb": np.bool_("chest_image" in data),
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.bool_("top_image" in data),
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclass(frozen=True)
class CompanyWristOutputs(transforms.DataTransformFn):
    """Slices model output actions back to the 7-DOF robot action space."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
