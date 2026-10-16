"""Run selectable, continuous game sessions through a verified v2 provider."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

from agent import choose_device, load_config
from session_profiles import PROFILE_PRESETS, SessionOptions, default_options


class StopControl:
    def __init__(self, path=None, max_seconds=0):
        self.path = Path(path) if path is not None else None
        self.deadline = time.monotonic() + max_seconds if max_seconds else None
        self.reason = None

    def __call__(self):
        if self.path is not None and self.path.exists():
            self.reason = 'stop_requested'
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.reason = self.reason or 'time_limit'
        return self.reason is not None


def state_contract(config):
    state = config.get('state', {})
    history = config.get('sessions', {}).get('history_frames', 4)
    if type(history) is not int or not 1 <= history <= 16:
        raise ValueError('sessions.history_frames must be an integer between 1 and 16')
    contract = {'observation_schema': 'team-v2', 'history_frames': history,
                'position_scale': state.get('position_scale', 1000.0),
                'damage_scale': state.get('damage_scale', 100.0),
                'damage_reward_scale': state.get('damage_reward_scale', 0.01)}
    for name in ('position_scale', 'damage_scale', 'damage_reward_scale'):
        from game_state import _positive_scale
        _positive_scale(contract[name], name)
    return contract


def _progress(model, options, lineage, *, session_steps, episodes):
    return {'profile': options.profile['profile'], 'match_format': options.profile['match_format'],
            'learning': options.learning, 'session_steps': session_steps,
            'model_timesteps': int(model.num_timesteps), 'episodes': episodes,
            'model_origin': lineage.get('strategy', 'unknown')}


def _make_callback(store, branch, lineage, options, contract, stop, start_steps):
    from stable_baselines3.common.callbacks import BaseCallback

    class SessionCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = 0
            self.next_save = options.checkpoint_interval

        def save(self):
            steps = int(self.model.num_timesteps - start_steps)
            lineage['last_session'] = _progress(self.model, options, lineage,
                                                 session_steps=steps, episodes=self.episodes)
            path = store.save(self.model, branch, lineage, contract=contract)
            print('Checkpoint saved:', path, json.dumps(lineage['last_session']), flush=True)

        def _on_step(self):
            if stop():
                raise KeyboardInterrupt
            self.episodes += sum(bool(done) for done in self.locals.get('dones', ()))
            steps = int(self.model.num_timesteps - start_steps)
            if steps >= self.next_save:
                self.save()
                self.next_save = steps + options.checkpoint_interval
            return True

    return SessionCallback()


def run_session(options, config, *, stop_file=None, check_only=False, model_kwargs=None):
    """Run client capabilities; a real Windows game-side provider is required."""
    from model_store import ModelStore
    from session_bridge import SessionBridgeClient, SessionEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    stop = StopControl(stop_file, options.max_seconds)
    bridge = config.get('bridge', {})
    client = SessionBridgeClient(
        host=bridge.get('host', '127.0.0.1'), port=bridge.get('port', 8765),
        timeout=bridge.get('timeout_seconds', 3.0),
        episode_timeout=bridge.get('episode_timeout_seconds', 120.0),
        expected_build=bridge.get('expected_build', '11.00.36'), stop_requested=stop)
    env = None
    model = None
    store = None
    lineage = None
    contract = None
    callback = None
    start_steps = 0
    steps_played = 0
    branch = 'teams' if options.profile['match_format'] == '2v2' else 'duel'
    try:
        if stop():
            print('Session stopped before connecting.', flush=True)
            return 130 if check_only else 0
        print(f'Mode: {PROFILE_PRESETS[options.profile["profile"]]["label"]}; '
              f'{options.profile["match_format"]}; learning={options.learning}', flush=True)
        print('Connecting to game-side session provider. The built-in spectator files cannot run these modes.', flush=True)
        client.connect()
        expected_mode = 'online_controlled' if options.profile['online'] else 'offline_controlled'
        if client.provider_mode != expected_mode:
            raise ValueError(f'Selected mode requires provider {expected_mode}; got {client.provider_mode}.')
        if client.controlled_player != options.profile['self_player']:
            raise ValueError('Provider controlled_player does not match the selected self_player.')
        if check_only:
            print('Provider handshake passed. No session was configured and no game actions were requested.', flush=True)
            return 0
        contract = state_contract(config)
        hold_seconds = config.get('step_seconds', 0.1)
        factory = lambda: SessionEnv(client, options.profile, hold_seconds=hold_seconds,
                                     position_scale=contract['position_scale'], damage_scale=contract['damage_scale'],
                                     damage_reward_scale=contract['damage_reward_scale'])
        env = VecFrameStack(DummyVecEnv([factory]), n_stack=contract['history_frames'])
        store = ModelStore(options.model_root, model_kwargs=model_kwargs)
        model, lineage = store.load_or_create(
            env, branch, device=choose_device(options.device), initialize_from=options.initialize_from,
            evaluation=not options.learning, contract=contract)
        start_steps = int(model.num_timesteps)
        print('Model:', store.path_for(branch), 'origin:', json.dumps(lineage), flush=True)
        if options.learning:
            callback = _make_callback(store, branch, lineage, options, contract, stop, start_steps)
            rollout = max(1, int(model.n_steps) * int(model.n_envs))
            while True:
                if stop():
                    break
                used = int(model.num_timesteps - start_steps)
                remaining = options.steps - used if options.steps else rollout * 40
                if options.steps and remaining <= 0:
                    break
                chunk = min(remaining, rollout * 40)
                model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=callback)
                callback.save()
        else:
            observation = env.reset()
            while not stop() and (not options.steps or steps_played < options.steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, _, _ = env.step(action)
                steps_played += 1
        print('Session finished:', stop.reason or 'step_limit', flush=True)
        return 0
    except KeyboardInterrupt:
        print('Session stopped:', stop.reason or 'keyboard_interrupt', flush=True)
        return 0 if stop.reason is not None and not check_only else 130
    finally:
        try:
            if env is not None:
                env.close()
            else:
                client.close()
        finally:
            if model is not None and options.learning:
                if callback is not None:
                    callback.save()
                else:
                    store.save(model, branch, lineage, contract=contract)
            elif model is not None:
                print(f'Evaluation ended after {steps_played} steps; model weights were not saved or changed.', flush=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--session-file', type=Path, help='saved menu settings JSON')
    parser.add_argument('--profile', choices=tuple(PROFILE_PRESETS))
    parser.add_argument('--format', dest='match_format', choices=['1v1', '2v2'])
    parser.add_argument('--self-player')
    parser.add_argument('--teammate-player')
    parser.add_argument('--opponents', nargs='+')
    parser.add_argument('--friendly-fire', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--learn', dest='learning', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--steps', type=int, help='0 runs until stopped; learning rounds up to a PPO rollout')
    parser.add_argument('--checkpoint-interval', type=int)
    parser.add_argument('--model-root')
    parser.add_argument('--initialize-from', help='compatible source checkpoint for a new branch')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--max-seconds', type=float, help='0 has no time limit')
    parser.add_argument('--stop-file', type=Path, help='stop cleanly if this file exists')
    parser.add_argument('--check', action='store_true', help='handshake only; no session/game actions')
    parser.add_argument('--dry-run', action='store_true', help='validate and show settings without dependencies or a provider')
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if 'sessions' in config and not isinstance(config['sessions'], dict):
            raise ValueError('config sessions must be an object')
        fields = ('match_format', 'self_player', 'teammate_player', 'opponents', 'friendly_fire',
                  'learning', 'steps', 'checkpoint_interval', 'model_root', 'initialize_from', 'device', 'max_seconds')
        overrides = {name: getattr(args, name) for name in fields if getattr(args, name) is not None}
        if args.session_file is not None:
            if args.profile is not None or overrides:
                raise ValueError('Use saved session settings or command-line profile fields, not both.')
            options = SessionOptions.from_dict(json.loads(args.session_file.read_text(encoding='utf-8')))
        else:
            options = default_options(args.profile or 'bot_training', **overrides)
        if args.dry_run:
            print(json.dumps(options.to_dict(), indent=2))
            return 0
        # Windows Ctrl+Break is also a graceful stop for console sessions.
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        return run_session(options, config, stop_file=args.stop_file, check_only=args.check)
    except KeyboardInterrupt:
        print('Session interrupted.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(f'Cannot start session: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
