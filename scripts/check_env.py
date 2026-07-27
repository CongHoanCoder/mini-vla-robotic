"""Continuously view the drawer-open sub-env from MT10, driven by the expert policy."""

import numpy as np
import gymnasium as gym
import metaworld
from metaworld.policies import ENV_POLICY_MAP
from envs.env_list import CHOSEN_IDX, CHOSEN_ENV_NAME, CHOSEN_INSTRUCTION
SEED = 42

envs = gym.make_vec("Meta-World/MT10", vector_strategy="sync", seed=SEED)
obs, info = envs.reset()

sub = envs.unwrapped.envs[CHOSEN_IDX]
sub.unwrapped.render_mode = "human"   # live viewer for only this sub-env
sub.unwrapped.camera_name = "corner4"  # or "topview", "behindGripper", etc.

# Expert (teacher) policy for drawer-open
policy = ENV_POLICY_MAP[CHOSEN_ENV_NAME]()

print("obs shape per env:", obs[CHOSEN_IDX].shape)

try:
    while True:                                  # run forever until Ctrl-C
        # Start from a zero batch, then set the expert action for our sub-env.
        actions = np.zeros_like(envs.action_space.sample())

        drawer_obs = obs[CHOSEN_IDX]
        # MT10 obs may have a one-hot task-ID appended; the scripted policy
        # expects the raw 39-dim obs. Slice it off if present.
        expert_obs = drawer_obs[:39] if drawer_obs.shape[0] > 39 else drawer_obs

        actions[CHOSEN_IDX] = policy.get_action(expert_obs)

        obs, reward, terminated, truncated, info = envs.step(actions)
        sub.render()

        # SyncVectorEnv auto-resets sub-envs when they finish, so the drawer-open
        # episode restarts automatically and the expert solves it again.
except KeyboardInterrupt:
    print("\nStopping (Ctrl-C).")
finally:
    envs.close()


