"""Exercise CNN training and persistence without a desktop or game."""
import tempfile
import unittest
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch

class VisualFixture(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
    action_space = gym.spaces.Discrete(16)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros((84, 84, 1), np.uint8), {}
    def step(self, action):
        self.count += 1
        return np.full((84, 84, 1), self.count % 256, np.uint8), float(action == 1), False, self.count >= 8, {}

class LearningTests(unittest.TestCase):
    def test_visual_policy_updates_and_checkpoint_resumes(self):
        torch.set_num_threads(2)
        env = VecFrameStack(DummyVecEnv([VisualFixture]), n_stack=4)
        try:
            model = PPO('CnnPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu')
            before = [p.detach().clone() for p in model.policy.parameters()]
            model.learn(16)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.policy.parameters())))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'policy'
                obs = env.reset()
                expected, _ = model.predict(obs, deterministic=True)
                model.save(path)
                resumed = PPO.load(path, env=env, device='cpu')
                actual, _ = resumed.predict(obs, deterministic=True)
                np.testing.assert_array_equal(expected, actual)
                resumed.learn(8, reset_num_timesteps=False)
                self.assertEqual(resumed.num_timesteps, 24)
        finally:
            env.close()
