"""
Lightweight helper to load Wan's text-to-video model for inference demos.

All config and the WanT2V class plus its dependencies are consolidated here so
this file can be used standalone inside the assignment folder.
"""
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from easydict import EasyDict
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from wan.distributed.fsdp import shard_model  # type: ignore  # noqa: E402
from wan.modules.model import WanModel  # type: ignore  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # type: ignore  # noqa: E402
from wan.modules.vae import WanVAE  # type: ignore  # noqa: E402
from wan.utils.fm_solvers import (  # type: ignore  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore  # noqa: E402


def _shared_cfg() -> EasyDict:
    cfg = EasyDict()
    # t5
    cfg.t5_model = 'umt5_xxl'
    cfg.t5_dtype = torch.bfloat16
    cfg.text_len = 512
    # transformer
    cfg.param_dtype = torch.bfloat16
    # inference
    cfg.num_train_timesteps = 1000
    cfg.sample_fps = 16
    cfg.sample_neg_prompt = (
        '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，'
        '最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，'
        '画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，'
        '杂乱的背景，三条腿，背景人很多，倒着走'
    )
    return cfg


def _t2v_1_3B_cfg() -> EasyDict:
    cfg = EasyDict(__name__='Config: Wan T2V 1.3B')
    cfg.update(_shared_cfg())
    cfg.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
    cfg.t5_tokenizer = 'google/umt5-xxl'
    cfg.vae_checkpoint = 'Wan2.1_VAE.pth'
    cfg.vae_stride = (4, 8, 8)
    cfg.patch_size = (1, 2, 2)
    cfg.dim = 1536
    cfg.ffn_dim = 8960
    cfg.freq_dim = 256
    cfg.num_heads = 12
    cfg.num_layers = 30
    cfg.window_size = (-1, -1)
    cfg.qk_norm = True
    cfg.cross_attn_norm = True
    cfg.eps = 1e-6
    return cfg


def _t2v_14B_cfg() -> EasyDict:
    cfg = EasyDict(__name__='Config: Wan T2V 14B')
    cfg.update(_shared_cfg())
    cfg.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
    cfg.t5_tokenizer = 'google/umt5-xxl'
    cfg.vae_checkpoint = 'Wan2.1_VAE.pth'
    cfg.vae_stride = (4, 8, 8)
    cfg.patch_size = (1, 2, 2)
    cfg.dim = 5120
    cfg.ffn_dim = 13824
    cfg.freq_dim = 256
    cfg.num_heads = 40
    cfg.num_layers = 40
    cfg.window_size = (-1, -1)
    cfg.qk_norm = True
    cfg.cross_attn_norm = True
    cfg.eps = 1e-6
    return cfg


def load_wan_t2v(
    checkpoint_dir: str,
    model_size: Literal["1.3B", "14B"] = "1.3B",
    device: str = "cuda:0",
    t5_cpu: bool = False,
):
    """
    Build a WanT2V model instance with the desired config and checkpoint path.

    Args:
        checkpoint_dir: Directory containing Wan checkpoints (T5, VAE, DiT).
        model_size: Select between the smaller 1.3B or larger 14B text-to-video.
        device: Torch device string, e.g. ``cuda:0``.
        t5_cpu: If True, keeps the T5 encoder on CPU to save VRAM.
    """
    device_id = 0
    if device.startswith("cuda") and ":" in device:
        try:
            device_id = int(device.split(":")[1])
        except (ValueError, IndexError):
            device_id = 0

    config = _t2v_1_3B_cfg() if model_size == "1.3B" else _t2v_14B_cfg()
    return WanT2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        t5_cpu=t5_cpu,
    )


class WanT2V:
    """
    Wan text-to-video generator. Adapted from `wan.text2video` with imports consolidated here.
    """

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from wan.distributed.xdit_context_parallel import (  # type: ignore
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
