"""
Minimal text-to-video inference entrypoint with bundled dependencies.

Example:
    python src/sample.py \\
        --ckpt_dir /path/to/Wan2.1-T2V-1.3B \\
        --prompt "A cat astronaut floating through neon clouds, cinematic lighting."
"""
import argparse
import binascii
import os
import os.path as osp
from pathlib import Path
import sys

import imageio
import torch
import torchvision

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ASSIGNMENT_ROOT.parent
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

try:  # Works when executed as a module
    from .model import load_wan_t2v  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001  # Fallback when running as a script
    from model import load_wan_t2v  # type: ignore  # noqa: E402


def _rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    """Save a video tensor to disk using imageio/torchvision."""
    cache_file = osp.join('/tmp', _rand_name(
        suffix=suffix)) if save_file is None else save_file

    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:  # noqa: BLE001
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan text-to-video inference.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./Wan2.1-T2V-1.3B/",
        help="Directory containing Wan checkpoints. Defaults to env WAN_T2V_CKPT.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage.",
        help="Text prompt to drive generation.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Output width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output height in pixels.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="Number of frames (must be 4n+1).",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="Diffusion sampling steps.",
    )
    parser.add_argument(
        "--guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="Sampler backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (-1 for random).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string, e.g. cuda:0.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        help="Keep T5 encoder on CPU to reduce VRAM.",
    )
    parser.add_argument(
        "--no_offload",
        action="store_true",
        help="Disable model offloading between steps (uses more VRAM, faster).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ASSIGNMENT_ROOT / "results" / "video.mp4"),
        help="Where to save the generated video.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.ckpt_dir:
        raise SystemExit("Please provide --ckpt_dir or set WAN_T2V_CKPT.")

    model = load_wan_t2v(
        checkpoint_dir=args.ckpt_dir,
        model_size="1.3B",  # fixed to T2V-1.3B per homework requirement
        device=args.device,
        t5_cpu=args.t5_cpu,
    )
    video = model.generate(
        input_prompt=args.prompt,
        size=(args.width, args.height),
        frame_num=args.frame_num,
        shift=5.0,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=not args.no_offload,
    )
    if video is None:
        raise RuntimeError("Model returned no video (check distributed rank and device).")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = getattr(model.config, "sample_fps", 16)
    cache_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
