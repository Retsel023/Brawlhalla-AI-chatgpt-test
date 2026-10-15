"""Visual PPO agent for a locally running Brawlhalla training session."""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from controls import ACTIONS, Feedback


class GameEnv(gym.Env):
    def __init__(self, config, feedback):
        # Import desktop tools lazily so --help and unit checks work without a display.
        import mss
        from pynput.keyboard import Controller, Key
        self.camera = mss.mss()
        self.keyboard = Controller()
        self.config, self.feedback = config, feedback
        self.keys = {name: getattr(Key, key, key) for name, key in config['keys'].items()}
        self.held = set()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
        self.deadline = 0

    def release(self):
        for key in tuple(self.held):
            try:
                self.keyboard.release(key)
            finally:
                self.held.discard(key)

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
        if self.feedback.stop:
            self.release()
            raise KeyboardInterrupt
        wanted = {self.keys[name] for name in ACTIONS[int(action)]}
        # Pulse action buttons; movement can remain held across steps.
        movement = {self.keys[name] for name in ('left', 'right', 'down')}
        for key in self.held - (wanted & movement):
            self.keyboard.release(key)
        self.held &= wanted & movement
        try:
            for key in wanted - self.held:
                self.keyboard.press(key)
                self.held.add(key)
            time.sleep(self.config['step_seconds'])
            obs = self.observe()
        finally:
            for key in tuple(self.held - movement):
                self.keyboard.release(key)
                self.held.discard(key)
        reward, done = self.feedback.consume()
        return obs, reward, done, time.monotonic() >= self.deadline, {}

    def close(self):
        self.release()
        self.camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'train', 'play'])
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model', default='models/ppo_brawlhalla')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='AMD ROCm PyTorch also uses the cuda device name')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    import torch
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('GPU unavailable. Install the AMD-supported Windows ROCm PyTorch build, or use --device cpu.')
        if torch.version.hip is None:
            parser.error('Expected AMD ROCm PyTorch for this RX 9070 XT workflow.')
        probe = torch.randn((32, 32), device='cuda', requires_grad=True)
        (probe @ probe.T).sum().backward()
        torch.cuda.synchronize()
        print('GPU verified:', torch.cuda.get_device_name(0))
    else:
        torch.set_num_threads(4)
        print('Using CPU; AMD GPU support has not been activated.')
    if config['step_seconds'] <= 0 or config['episode_seconds'] <= 0:
        parser.error('step_seconds and episode_seconds must be positive')
    if args.steps <= 0:
        parser.error('--steps must be positive')
    if args.mode == 'capture':
        import mss
        with mss.mss() as camera:
            shot = camera.grab(config['capture'])
            Image.frombytes('RGB', shot.size, shot.rgb).save('capture.png')
        print('Saved capture.png. Verify it contains only the game view before training.')
        return
    from pynput.keyboard import Listener, Key
    feedback = Feedback()
    pressed = set()

    def on_press(key):
        if key in pressed:
            return
        pressed.add(key)
        if key == Key.f8:
            feedback.add(1)
        elif key == Key.f9:
            feedback.add(-1)
        elif key == Key.f10:
            with feedback.lock:
                feedback.done = True
        elif key == Key.f12:
            feedback.stop = True

    listener = Listener(on_press=on_press, on_release=lambda key: pressed.discard(key))
    listener.start()
    env = None
    model = None
    try:
        print('Focus Brawlhalla offline training. F8 reward; F9 penalty; F10 episode boundary; F12 stop.')
        print('Episode boundaries do not reset the game. Keep the training session running.')
        time.sleep(config['countdown_seconds'])
        if feedback.stop:
            return
        env = VecFrameStack(DummyVecEnv([lambda: GameEnv(config, feedback)]), n_stack=4)
        path = Path(args.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.with_suffix('.zip').exists():
            model = PPO.load(str(path), env=env, device=device)
            print('Loaded saved progress.')
        elif args.mode == 'play':
            raise FileNotFoundError('Train a model before using play mode.')
        else:
            model = PPO('CnnPolicy', env, n_steps=256, batch_size=64, verbose=1, device=device)
        if args.mode == 'train':
            model.learn(total_timesteps=args.steps, reset_num_timesteps=False,
                        callback=CheckpointCallback(save_freq=1000, save_path='models/checkpoints'))
        else:
            obs = env.reset()
            while not feedback.stop:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = env.step(action)
    except KeyboardInterrupt:
        print('Stopped; releasing controls and saving the latest model.')
    finally:
        if env is not None:
            env.close()
        listener.stop()
        if model is not None and args.mode == 'train':
            model.save(args.model)


if __name__ == '__main__':
    main()
