from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from torch import nn


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _to_torch_batch(tree: Any, device: torch.device | str) -> Any:
    def convert(value: Any) -> torch.Tensor:
        tensor = torch.from_numpy(np.array(value)).to(device)[None, ...]
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor

    return jax.tree.map(convert, tree)


def _as_bool_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.bool)


def _as_nchw_image(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] != 3 and value.shape[-1] == 3:
        value = value.permute(0, 3, 1, 2)
    if value.dtype == torch.uint8:
        value = value.to(torch.float32) / 255.0 * 2.0 - 1.0
    elif value.dtype != torch.float32:
        value = value.to(torch.float32)
    return value


def _make_att_2d_masks_no_checks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = torch.logical_and(pad_masks[:, None, :], pad_masks[:, :, None])
    return torch.logical_and(att_2d_masks, pad_2d_masks)


class Pi05SampleActionsTRTWrapper(nn.Module):
    """Tensor-only wrapper around PI0Pytorch.sample_actions.

    The OpenPI policy does Python-side transforms before model inference. TensorRT
    should only see fixed-shape tensors after those transforms, so this wrapper
    reconstructs the Observation object from a flat tensor signature.
    """

    def __init__(self, model: nn.Module, device: torch.device | str, num_steps: int):
        super().__init__()

        self.model = model
        self._device = torch.device(device)
        self._num_steps = int(num_steps)

    def forward(
        self,
        base_0_rgb: torch.Tensor,
        left_wrist_0_rgb: torch.Tensor,
        right_wrist_0_rgb: torch.Tensor,
        base_0_rgb_mask: torch.Tensor,
        left_wrist_0_rgb_mask: torch.Tensor,
        right_wrist_0_rgb_mask: torch.Tensor,
        state: torch.Tensor,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        images = [
            _as_nchw_image(base_0_rgb),
            _as_nchw_image(left_wrist_0_rgb),
            _as_nchw_image(right_wrist_0_rgb),
        ]
        img_masks = [
            _as_bool_tensor(base_0_rgb_mask),
            _as_bool_tensor(left_wrist_0_rgb_mask),
            _as_bool_tensor(right_wrist_0_rgb_mask),
        ]
        lang_tokens = tokenized_prompt
        lang_masks = _as_bool_tensor(tokenized_prompt_mask)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        prefix_att_2d_masks = _make_att_2d_masks_no_checks(
            prefix_pad_masks,
            prefix_att_masks,
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        bsize = state.shape[0]
        dt = torch.tensor(-1.0 / self._num_steps, dtype=torch.float32, device=state.device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=state.device)
        for _ in range(self._num_steps):
            v_t = self._denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                time.expand(bsize),
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    def _denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = _make_att_2d_masks_no_checks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self.model._prepare_attention_masks_4d(full_att_2d_masks)

        outputs_embeds, _ = self.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.model.action_out_proj(suffix_out)


def flattened_trt_inputs(inputs: Mapping[str, Any], noise: torch.Tensor) -> tuple[torch.Tensor, ...]:
    image = inputs["image"]
    image_mask = inputs["image_mask"]
    return (
        image["base_0_rgb"],
        image["left_wrist_0_rgb"],
        image["right_wrist_0_rgb"],
        image_mask["base_0_rgb"],
        image_mask["left_wrist_0_rgb"],
        image_mask["right_wrist_0_rgb"],
        inputs["state"],
        inputs["tokenized_prompt"],
        inputs["tokenized_prompt_mask"],
        noise,
    )


def build_example_inputs(policy: Any, obs: dict[str, Any], device: torch.device | str) -> tuple[Any, tuple[torch.Tensor, ...]]:
    transformed = policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
    inputs = _to_torch_batch(transformed, device)
    action_horizon = int(policy._model.config.action_horizon)  # noqa: SLF001
    action_dim = int(policy._model.config.action_dim)  # noqa: SLF001
    noise = torch.zeros((1, action_horizon, action_dim), dtype=torch.float32, device=device)
    return inputs, flattened_trt_inputs(inputs, noise)


def compile_sample_actions_to_tensorrt(
    policy: Any,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    device: torch.device | str = "cuda:0",
    num_steps: int = 10,
    engine_path: str | Path | None = None,
    fp16: bool = True,
) -> nn.Module:
    import torch_tensorrt

    wrapper = Pi05SampleActionsTRTWrapper(policy._model, device, num_steps).eval().to(device)  # noqa: SLF001
    with torch.no_grad():
        compiled = torch_tensorrt.compile(
            wrapper,
            ir="dynamo",
            inputs=example_inputs,
            enabled_precisions={torch.float16} if fp16 else {torch.float32},
            truncate_double=True,
            min_block_size=5,
        )

    if engine_path is not None:
        Path(engine_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        torch_tensorrt.save(compiled, str(engine_path), inputs=example_inputs)
    return compiled


def load_tensorrt_module(engine_path: str | Path, device: torch.device | str = "cuda:0") -> nn.Module:
    loaded = torch.export.load(str(engine_path)).module()
    loaded.eval()
    return loaded.to(device)


class TensorRTPolicyAdapter:
    """Drop-in Policy-like object using TensorRT for sample_actions."""

    def __init__(
        self,
        policy: Any,
        trt_module: nn.Module,
        *,
        device: torch.device | str = "cuda:0",
    ):
        self._policy = policy
        self._trt_module = trt_module.eval().to(device)
        self._device = torch.device(device)
        self._model = policy._model  # noqa: SLF001
        self._input_transform = policy._input_transform  # noqa: SLF001
        self._output_transform = policy._output_transform  # noqa: SLF001
        self._metadata = policy.metadata
        self._action_horizon = int(self._model.config.action_horizon)
        self._action_dim = int(self._model.config.action_dim)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None, sample_kwargs: dict[str, Any] | None = None) -> dict:
        if sample_kwargs:
            raise ValueError("TensorRTPolicyAdapter currently supports the non-RTC path only; sample_kwargs must be empty.")

        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = _to_torch_batch(inputs, self._device)

        if noise is None:
            noise_tensor = torch.randn(
                (1, self._action_horizon, self._action_dim),
                dtype=torch.float32,
                device=self._device,
            )
        else:
            noise_tensor = torch.from_numpy(np.asarray(noise, dtype=np.float32)).to(self._device)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor[None, ...]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        actions = self._trt_module(*flattened_trt_inputs(inputs, noise_tensor))
        end.record()
        torch.cuda.synchronize(self._device)
        infer_ms = float(start.elapsed_time(end))

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().float().cpu()), outputs)
        model_actions = np.asarray(outputs["actions"], dtype=np.float32)
        outputs = self._output_transform(outputs)
        outputs["model_actions"] = model_actions
        outputs["policy_timing"] = {"infer_ms": infer_ms}
        return outputs


def attach_tensorrt_policy(
    policy: Any,
    *,
    engine_path: str | Path | None,
    build_engine_path: str | Path | None,
    example_obs: dict[str, Any],
    device: torch.device | str = "cuda:0",
    num_steps: int = 10,
    fp16: bool = True,
) -> TensorRTPolicyAdapter:
    _, example_inputs = build_example_inputs(policy, example_obs, device)
    if engine_path is not None:
        module = load_tensorrt_module(engine_path, device)
    else:
        module = compile_sample_actions_to_tensorrt(
            policy,
            example_inputs,
            device=device,
            num_steps=num_steps,
            engine_path=build_engine_path,
            fp16=fp16,
        )
    return TensorRTPolicyAdapter(policy, module, device=device)

