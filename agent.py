"""Brawlhalla visual training, game-state observation, and controlled-bridge PPO."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from controls import Feedback
from telemetry import PLAYER_KEY, positive_seconds


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('config must be a JSON object')
    for section in ('state', 'bridge', 'game'):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f'config {section} must be an object')
    return config


def _positive_config(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a positive finite number')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return value


def choose_device(requested):
    import torch
    device = 'cuda' if requested == 'auto' and torch.cuda.is_available() else requested
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('GPU unavailable. Install the AMD-supported Windows ROCm PyTorch build, or use --device cpu.')
        if torch.version.hip is None:
            raise ValueError('Expected AMD ROCm PyTorch for this RX 9070 XT workflow.')
        probe = torch.randn((32, 32), device='cuda', requires_grad=True)
        (probe @ probe.T).sum().backward()
        torch.cuda.synchronize()
        print('GPU verified:', torch.cuda.get_device_name(0))
    else:
        torch.set_num_threads(4)
        print('Using CPU.')
    return device


def checkpoint_path(value, source):
    path = Path(value or ('models/ppo_state' if source == 'bridge' else 'models/ppo_brawlhalla'))
    return path if str(path).endswith('.zip') else Path(str(path) + '.zip')


def make_env(source, config, feedback, *, client=None, self_slot='Player1', opponent_slot='Player2'):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    if source == 'visual':
        from desktop import GameEnv
        return VecFrameStack(DummyVecEnv([lambda: GameEnv(config, feedback)]), n_stack=4), 'CnnPolicy'
    if source != 'bridge':
        raise ValueError('The spectator telemetry source is observation-only; use observe mode.')
    from state_bridge import ControlledStateEnv
    state = config.get('state', {})
    factory = lambda: ControlledStateEnv(
        client, self_slot=self_slot, opponent_slot=opponent_slot,
        hold_seconds=config['step_seconds'],
        position_scale=state.get('position_scale', 1000.0),
        damage_scale=state.get('damage_scale', 100.0),
        damage_reward_scale=state.get('damage_reward_scale', 0.01))
    return DummyVecEnv([factory]), 'MlpPolicy'


def load_or_create_model(env, policy, path, *, mode, device):
    from stable_baselines3 import PPO
    if path.exists():
        try:
            model = PPO.load(str(path), env=env, device=device)
        except ValueError as error:
            raise ValueError(
                f'Checkpoint {path} is incompatible with this observation space. '
                'Visual CNN and structured-state models require separate checkpoints. '
                f'{error}') from error
        print('Loaded saved model:', path)
        return model
    if mode == 'play':
        raise FileNotFoundError(f'Train a model before play mode: {path}')
    return PPO(policy, env, n_steps=256, batch_size=64, verbose=1, device=device)


def start_visual_hotkeys(feedback):
    from pynput.keyboard import Listener, Key
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
    return listener


def run_policy(args, config, source, self_slot, opponent_slot):
    from stable_baselines3.common.callbacks import CheckpointCallback
    from state_bridge import StateBridgeClient
    feedback = Feedback()
    client = None
    listener = None
    env = None
    model = None
    path = checkpoint_path(args.model, source)
    device = choose_device(args.device)
    try:
        if source == 'visual':
            listener = start_visual_hotkeys(feedback)
            print('Focus Brawlhalla offline training. F8 reward; F9 penalty; F10 episode boundary; F12 stop.')
            print('Visual episode boundaries do not reset the game. Keep the training session running.')
        else:
            settings = config.get('bridge', {})
            print('Connecting to the offline game-state bridge. Ctrl+C stops and saves.')
            print('This requires a separate game-side provider; the spectator exporter cannot serve as that provider.')
            client = StateBridgeClient(
                host=args.bridge_host or settings.get('host', '127.0.0.1'),
                port=args.bridge_port or settings.get('port', 8765),
                timeout=args.timeout or settings.get('timeout_seconds', 3.0),
                expected_build=settings.get('expected_build', '11.00.36'),
                stop_requested=lambda: feedback.stop)
            client.connect()
        countdown = config.get('countdown_seconds', 5)
        if isinstance(countdown, bool) or not isinstance(countdown, (int, float)) or not math.isfinite(countdown) or countdown < 0:
            raise ValueError('countdown_seconds must be a finite nonnegative number')
        deadline = time.monotonic() + countdown
        while time.monotonic() < deadline:
            if feedback.stop:
                return 0
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        if feedback.stop:
            return 0
        env, policy = make_env(source, config, feedback, client=client,
                               self_slot=self_slot, opponent_slot=opponent_slot)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = load_or_create_model(env, policy, path, mode=args.mode, device=device)
        if args.mode == 'train':
            model.learn(total_timesteps=args.steps, reset_num_timesteps=False,
                        callback=CheckpointCallback(
                            save_freq=1000, save_path=str(path.parent / 'checkpoints' / source),
                            name_prefix=path.stem))
        else:
            obs = env.reset()
            while not feedback.stop:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = env.step(action)
        return 0
    finally:
        try:
            if env is not None:
                env.close()
            elif client is not None:
                client.close()
        finally:
            try:
                if listener is not None:
                    listener.stop()
            finally:
                if model is not None and args.mode == 'train':
                    model.save(str(path))
                    print('Saved model:', path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['doctor', 'capture', 'observe', 'train', 'play'])
    parser.add_argument('--source', choices=['visual', 'telemetry', 'bridge'])
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model', help='defaults to separate visual/state checkpoint paths')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='AMD ROCm PyTorch also uses the cuda device name')
    parser.add_argument('--self-player', help='explicit Player<number> key; defaults to config state.self_player')
    parser.add_argument('--opponent-player', help='explicit Player<number> key; defaults to config state.opponent_player')
    parser.add_argument('--telemetry-dir', type=Path, default=Path.home() / 'BrawlhallaStatsLive')
    parser.add_argument('--duration', type=positive_seconds, default=60.0, help='observe duration in seconds')
    parser.add_argument('--timeout', type=positive_seconds, help='seconds without a valid feed/bridge response')
    parser.add_argument('--bridge-host', help='local controlled bridge hostname/address')
    parser.add_argument('--bridge-port', type=int)
    parser.add_argument('--game-dir', type=Path, help='installed Windows game directory for doctor mode')
    args = parser.parse_args(argv)
    source = args.source or ('telemetry' if args.mode == 'observe' else 'visual')
    if args.mode in ('train', 'play') and source == 'telemetry':
        parser.error('The built-in telemetry feed is spectator-only and cannot acknowledge actions or resets. '
                     'Use observe mode, or --source bridge with an actual offline-controlled provider.')
    if args.mode == 'observe' and source != 'telemetry':
        parser.error('observe reads the built-in telemetry source; use --source telemetry.')
    if args.mode == 'capture' and source != 'visual':
        parser.error('capture requires the visual source.')
    if args.steps <= 0:
        parser.error('--steps must be positive')
    if args.bridge_port is not None and not 1 <= args.bridge_port <= 65535:
        parser.error('--bridge-port must be between 1 and 65535')
    try:
        config = load_config(args.config)
        if args.mode == 'doctor':
            from game_install import DEFAULT_GAME_DIRECTORY, INSPECTED_SWF_SHA256, inspect_install
            settings = config.get('game', {})
            report = inspect_install(
                args.game_dir or settings.get('directory', DEFAULT_GAME_DIRECTORY),
                expected_swf_sha256=settings.get('expected_swf_sha256', INSPECTED_SWF_SHA256))
            print(json.dumps(report, indent=2))
            return 0 if report['ready'] else 1
        state = config.get('state', {})
        own = args.self_player or state.get('self_player', 'Player1')
        other = args.opponent_player or state.get('opponent_player', 'Player2')
        if source != 'visual':
            if not all(isinstance(slot, str) and PLAYER_KEY.fullmatch(slot) for slot in (own, other)) or own == other:
                raise ValueError('Select two different Player<number> keys with --self-player and --opponent-player.')
        if args.mode == 'observe':
            from state_monitor import monitor
            print(f'Observing {own} versus {other}. Reward projections are diagnostics; no controls or learning.', file=sys.stderr)
            stats = monitor(args.telemetry_dir, self_slot=own, opponent_slot=other,
                            duration=args.duration, timeout=args.timeout or 3.0,
                            position_scale=state.get('position_scale', 1000.0),
                            damage_scale=state.get('damage_scale', 100.0),
                            damage_reward_scale=state.get('damage_reward_scale', 0.01))
            print(f'Observed {stats.recorded} fresh updates.', file=sys.stderr)
            return 0
        if args.mode == 'capture':
            import mss
            from PIL import Image
            with mss.mss() as camera:
                shot = camera.grab(config['capture'])
                Image.frombytes('RGB', shot.size, shot.rgb).save('capture.png')
            print('Saved capture.png. Verify it contains only the game view before training.')
            return 0
        _positive_config(config['step_seconds'], 'step_seconds')
        if source == 'visual':
            _positive_config(config['episode_seconds'], 'episode_seconds')
        return run_policy(args, config, source, own, other)
    except KeyboardInterrupt:
        print('Stopped. Any initialized training model has been saved.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        print(f'Cannot run {args.mode}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
