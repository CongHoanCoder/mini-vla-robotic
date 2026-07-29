"""Test trained VLA policy on Meta-World MT1"""

import argparse
import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.metaworld_env import MetaWorldMT1Wrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from models.vision.registry import VisionEncoderCfg
from utils.tokenizer import SimpleTokenizer


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    d_model = ckpt["d_model"]
    diffusion_T = ckpt["diffusion_T"]
    use_flow_matching = ckpt.get("use_flow_matching", False)
    obs_history_len = ckpt.get("obs_history_len", 1)
    action_head_hidden_dim = ckpt.get("action_head_hidden_dim", 128)
    fusion_type = ckpt.get("fusion_type", "mlp")

    vision_cfg = None
    if "vision_cfg" in ckpt:
        vision_cfg = VisionEncoderCfg(**ckpt["vision_cfg"])

    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_T,
        vision_cfg=vision_cfg,
        use_flow_matching=use_flow_matching,
        obs_history_len=obs_history_len,
        action_head_hidden_dim=action_head_hidden_dim,
        fusion_type=fusion_type,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(vocab=vocab)

    action_stats = ckpt.get("action_stats")
    if action_stats is None:
        action_mean = np.zeros(action_dim, dtype=np.float32)
        action_std = np.ones(action_dim, dtype=np.float32)
    else:
        action_mean = np.asarray(action_stats["mean"], dtype=np.float32)
        action_std = np.asarray(action_stats["std"], dtype=np.float32)
        action_std = np.clip(action_std, float(action_stats.get("eps", 1e-6)), None)

    return model, tokenizer, action_mean, action_std, obs_history_len, fusion_type


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained VLA on Meta-World MT1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--env-name", type=str, default="push-v3")
    parser.add_argument("--camera-name", type=str, default="topview")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--instruction", type=str, default="push the object to the goal")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[test] Loading checkpoint: {args.checkpoint}")
    model, tokenizer, action_mean, action_std, obs_history_len, fusion_type = load_checkpoint(
        args.checkpoint, device
    )

    instr_tokens = tokenizer.encode(args.instruction)
    text_ids = (
        torch.tensor(instr_tokens, dtype=torch.long).unsqueeze(0).to(device)
    )

    env = MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )

    print(f"[test] Env: {args.env_name}, state_dim={env.state_dim}, action_dim={env.action_dim}")
    print(f"[test] action_mean={action_mean.tolist()}, action_std={action_std.tolist()}")
    print(f"[test] obs_history_len={obs_history_len}, fusion_type={fusion_type}")

    episode_rewards = []
    episode_steps = []
    episode_successes = []

    for ep in range(args.episodes):
        img, state, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_success = 0

        step = 0
        done = False
        while not done and step < args.max_steps:
            img_t = (
                torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            ).to(device)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action_t = model.act(img_t, text_ids, state_t)

            action_np = action_t.squeeze(0).cpu().numpy()
            action_np = action_np * action_std + action_mean
            action_np = np.clip(action_np, env.action_low, env.action_high)

            img, state, reward, done, info = env.step(action_np)
            ep_reward += reward
            ep_success = max(ep_success, int(info.get("success", 0)))
            step += 1

        episode_rewards.append(ep_reward)
        episode_steps.append(step)
        episode_successes.append(ep_success)
        print(
            f"[test] Ep {ep+1}/{args.episodes}: reward={ep_reward:.3f}, "
            f"steps={step}, success={ep_success}"
        )

    env.close()

    rewards = np.array(episode_rewards, dtype=np.float32)
    successes = np.array(episode_successes, dtype=np.float32)
    steps = np.array(episode_steps, dtype=np.float32)
    print()
    print(f"[test] Success rate: {100.0 * successes.mean():.1f}%")
    print(f"[test] Reward: mean={rewards.mean():.3f}, std={rewards.std():.3f}")
    print(f"[test] Steps:  mean={steps.mean():.1f},  std={steps.std():.1f}")
    print(f"[test] Total episodes: {len(episode_rewards)}")
    print("[test] Done.")


if __name__ == "__main__":
    main()