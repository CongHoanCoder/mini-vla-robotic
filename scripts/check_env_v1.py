"""Inspect cameras in PickCubeSO100-v1."""
import gymnasium as gym
import mani_skill.envs  # noqa: F401

env = gym.make("PickCubeSO100-v1", num_envs=1,
               obs_mode="rgbd", control_mode="pd_joint_delta_pos",
               render_mode="rgb_array")
obs, _ = env.reset(seed=0)
base = env.unwrapped

print("SENSOR CAMERAS:", list(base.scene.sensors.keys()))
print("HUMAN-RENDER CAMERAS:", list(base.scene.human_render_cameras.keys()))

if isinstance(obs, dict) and "sensor_data" in obs:
    for cam, textures in obs["sensor_data"].items():
        for tex, t in textures.items():
            print(f"sensor_data.{cam}.{tex}: {tuple(t.shape)} {t.dtype}")
env.close()
