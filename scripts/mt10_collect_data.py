"""Collect demonstration data from Meta-World MT1 environments using expert policies"""

import os
import argparse
import time
import numpy as np
import gymnasium as gym
import metaworld
from metaworld.policies import ENV_POLICY_MAP
from utils.tokenizer import SimpleTokenizer
from envs.env_list import CHOSEN_IDX, CHOSEN_ENV_NAME, CHOSEN_INSTRUCTION

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default=CHOSEN_ENV_NAME)
    parser.add_argument("--camera-name", type=str, default="corner2",
                        help="Meta-World camera: corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output-path", type=str, default=f"data/{CHOSEN_ENV_NAME}.npz")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional sleep between steps for visualization (seconds)")
    parser.add_argument("--instruction", type=str, default=CHOSEN_INSTRUCTION,
                        help="Fixed instruction for all episodes")
    return parser.parse_args()


def extract_state(obs):
    """
    Meta-World MT1 observations are already flat numpy arrays.
    """
    return np.asarray(obs, dtype=np.float32).ravel()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    SEED = args.seed
    envs = gym.make_vec("Meta-World/MT10", vector_strategy="sync", seed=SEED)
    obs, info = envs.reset()

    env = envs.unwrapped.envs[CHOSEN_IDX]
    env.unwrapped.render_mode = "rgb_array"   # live viewer for only this sub-env

    # Expert (teacher) policy for drawer-open
    policy = ENV_POLICY_MAP[CHOSEN_ENV_NAME]()
    obs, info = env.reset(seed=args.seed)
    # policy = ENV_POLICY_MAP[args.env_name]()

    images = []
    states = []
    actions = []
    texts = []

    # fixed instruction for this dataset
    instruction = args.instruction

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < args.max_steps:
            # expert policy action on raw obs
            action = policy.get_action(obs)  # shape (action_dim,)

            # log current transition
            img = env.render() # (H, W, 3) uint8
            state = extract_state(obs) # (state_dim,)

            images.append(img.copy())
            states.append(state.copy())
            actions.append(np.asarray(action, dtype=np.float32).copy())
            texts.append(instruction)

            # step env
            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {ep+1}/{args.episodes} finished after {steps} steps, success={int(info.get('success', 0))}")

    env.close()

    # stack arrays
    images = np.stack(images, axis=0)   # (N, H, W, 3)
    states = np.stack(states, axis=0)   # (N, state_dim)
    actions = np.stack(actions, axis=0) # (N, action_dim)

    # tokenize instructions
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=tokenizer.vocab,
    )

    print("Saved Meta-World", CHOSEN_ENV_NAME, "dataset to", args.output_path)
    print("  images:", images.shape)
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  text_ids:", text_ids.shape)


if __name__ == "__main__":
    main()
