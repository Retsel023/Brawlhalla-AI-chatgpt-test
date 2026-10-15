"""Desktop controls and screen observations for the original visual workflow."""

from contextlib import suppress
import time

import gymnasium as gym
import numpy as np
from PIL import Image

from controls import ACTIONS


class KeyboardActions:
    def __init__(self, bindings, *, controller=None, key_type=None):
        if controller is None or key_type is None:
            from pynput.keyboard import Controller, Key
            controller = Controller() if controller is None else controller
            key_type = Key if key_type is None else key_type
        self.controller = controller
        self.keys = {name: getattr(key_type, key, key) for name, key in bindings.items()}
        missing = {name for action in ACTIONS for name in action} - self.keys.keys()
        if missing:
            raise ValueError(f'Missing key bindings: {sorted(missing)}')
        self.movement = {self.keys[name] for name in ('left', 'right', 'down')}
        self.held = set()

    def _release(self, keys):
        first_error = None
        for key in tuple(keys):
            try:
                self.controller.release(key)
                self.held.discard(key)
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def apply(self, action):
        wanted = {self.keys[name] for name in ACTIONS[int(action)]}
        self._release(self.held - (wanted & self.movement))
        for key in wanted - self.held:
            self.held.add(key)
            self.controller.press(key)

    def release_buttons(self):
        self._release(self.held - self.movement)

    def release(self):
        self._release(self.held)


class GameEnv(gym.Env):
    """Real-time visual environment. Episode boundaries do not reset the game."""

    def __init__(self, config, feedback, *, camera=None, controls=None):
        self.config, self.feedback = config, feedback
        self.controls = None
        if camera is None:
            import mss
            camera = mss.mss()
        self.camera = camera
        try:
            self.controls = controls if controls is not None else KeyboardActions(config['keys'])
        except BaseException:
            self.camera.close()
            raise
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
        self.deadline = 0

    def release(self):
        self.controls.release()

    def observe(self):
        shot = self.camera.grab(self.config['capture'])
        frame = Image.frombytes('RGB', shot.size, shot.rgb).convert('L').resize((84, 84))
        return np.asarray(frame, dtype=np.uint8)[..., None]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.release()
        self.feedback.consume()
        self.deadline = time.monotonic() + self.config['episode_seconds']
        return self.observe(), {}

    def step(self, action):
        try:
            if self.feedback.stop:
                raise KeyboardInterrupt
            if not self.action_space.contains(action):
                raise ValueError(f'Invalid action: {action}')
            self.controls.apply(action)
            deadline = time.monotonic() + self.config['step_seconds']
            while time.monotonic() < deadline:
                if self.feedback.stop:
                    raise KeyboardInterrupt
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
            self.controls.release_buttons()
            obs = self.observe()
            reward, done = self.feedback.consume()
            return obs, reward, done, time.monotonic() >= self.deadline, {}
        except BaseException:
            with suppress(Exception):
                self.release()
            raise

    def close(self):
        try:
            self.release()
        finally:
            self.camera.close()
