import gymnasium as gym
import numpy as np
import metaworld



from envs.env_list import CHOSEN_IDX, CHOSEN_ENV_NAME, CHOSEN_INSTRUCTION

class MetaWorldMT10DrawerOpenWrapper:
    """
    Wraps a Metaworld MT10 environment into a simple interface:
    - reset() -> (image, state)
    - step(action) -> (image, state, reward, done, info)
    """
    def __init__(self, env_name=CHOSEN_ENV_NAME, seed=42, render_mode='rgb_array', camera_name='corner1'):


        
        # Use MT10 vectorized env to access the drawer-open sub-env
        SEED = seed
        envs = gym.make_vec("Meta-World/MT10", vector_strategy="sync", seed=SEED)
        self.env = envs.unwrapped.envs[CHOSEN_IDX]
        self.env.unwrapped.render_mode = render_mode
    
        
        self.render_mode = render_mode

        obs, _ = self.env.reset()
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.obs_shape = self._get_image().shape

    def _extract_state(self, obs):
        """
        Adapt this to your env's observation structure.
        Examples:
          - obs might be a dict with keys ["robot_state", "object_state"].
          - or it might already be a flat vector.
        """
        if isinstance(obs, dict):
            if "observation" in obs:
                state = obs["observation"]
            elif "robot_state" in obs or "object_state" in obs:
                state_parts = []
                if "robot_state" in obs:
                    state_parts.append(obs["robot_state"])
                if "object_state" in obs:
                    state_parts.append(obs["object_state"])
                state = np.concatenate(state_parts, axis=-1)
            else:
                raise KeyError(
                    f"No suitable state keys in observation dict. "
                    f"Available keys: {list(obs.keys())}"
                )
        else:
            state = obs
        return np.asarray(state, dtype=np.float32)

    def _get_image(self):
        img = self.env.render()
        img = img.astype(np.uint8)
        return img

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, info

    def step(self, action):
        obs, reward, truncate, terminate, info = self.env.step(action)
        done = truncate or terminate
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, reward, done, info

    def close(self):
        self.env.close()