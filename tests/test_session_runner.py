"""Exercise real PPO session orchestration against synthetic loopback providers.

These tests validate this project's protocol and learning code, not integration
with a running Brawlhalla installation or real matchmaking.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from session_profiles import PROFILE_PRESETS, SessionOptions, default_options, validate_profile
from session_runner import main, run_session
from test_session_bridge import SessionProviderFixture


ROOT = Path(__file__).resolve().parents[1]
MODEL_KWARGS = {
    'n_steps': 8, 'batch_size': 8, 'n_epochs': 1, 'verbose': 0, 'seed': 29,
    'policy_kwargs': {'net_arch': {'pi': [16], 'vf': [16]}},
}
DRY_RUN_IMPORT_GUARD = """
import builtins
import runpy
import sys
original_import = builtins.__import__
blocked = {'numpy', 'torch', 'gymnasium', 'stable_baselines3', 'pynput', 'mss', 'tkinter'}
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('dry-run imported ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProfileValidationTests(unittest.TestCase):
    def test_four_defaults_are_explicit_and_round_trip(self):
        self.assertEqual(set(PROFILE_PRESETS),
                         {'human_duel', 'bot_training', 'online_duel', 'online_teams'})
        for name in PROFILE_PRESETS:
            with self.subTest(name=name):
                options = default_options(name)
                self.assertEqual(SessionOptions.from_dict(options.to_dict()), options)
                self.assertEqual(options.profile['self_player'], 'Player1')
                self.assertEqual(options.steps, 0)
                self.assertEqual(options.learning, name != 'human_duel')
                self.assertEqual(options.profile['online'], name.startswith('online_'))
        teams = default_options('online_teams').profile
        self.assertEqual(teams['teammate_player'], 'Player2')
        self.assertEqual(teams['opponents'], ['Player3', 'Player4'])

    def test_roles_online_flags_and_friendly_fire_cannot_be_ambiguous(self):
        duel = default_options().profile
        invalid_duels = [
            {'opponents': ['Player1']}, {'opponents': ['Player2', 'Player3']},
            {'teammate_player': 'Player3'}, {'friendly_fire': True},
            {'friendly_fire': 0}, {'online': True}, {'self_player': 'self'},
        ]
        for mutation in invalid_duels:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_profile({**duel, **mutation})
        for overrides in ({'friendly_fire': True}, {'friendly_fire': 1},
                          {'teammate_player': 'Player3'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                default_options(**overrides)
        teams = default_options('online_teams').profile
        for mutation in ({'teammate_player': None}, {'teammate_player': 'Player3'},
                         {'opponents': ['Player3', 'Player3']}, {'friendly_fire': 'true'},
                         {'match_format': '1v1'}, {'online': False}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_profile({**teams, **mutation})

    def test_session_run_limits_and_devices_are_validated(self):
        for field, value in (
                ('steps', -1), ('steps', True), ('steps', 1.5),
                ('checkpoint_interval', 0), ('checkpoint_interval', True),
                ('learning', 1), ('max_seconds', -1), ('max_seconds', float('nan')),
                ('max_seconds', float('inf')), ('device', 'directml'), ('model_root', ''),
                ('initialize_from', '')):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                default_options(**{field: value})


class SessionRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model_root = self.directory / 'models'
        self.output = io.StringIO()

    def options(self, name='bot_training', **kwargs):
        return default_options(name, **{
            'steps': 16, 'checkpoint_interval': 8,
            'model_root': str(self.model_root), 'device': 'cpu', **kwargs,
        })

    def provider(self, options, *, alter=None, episode_steps=3):
        provider = SessionProviderFixture(options.profile, alter=alter, episode_steps=episode_steps)
        self.addCleanup(provider.close)
        return provider

    def config(self, provider):
        return {'step_seconds': 0.001, 'sessions': {'history_frames': 4},
                'bridge': {'host': '127.0.0.1', 'port': provider.port,
                           'timeout_seconds': 0.5, 'episode_timeout_seconds': 1,
                           'expected_build': '11.00.36'}}

    def run_options(self, options, *, provider=None, **kwargs):
        provider = self.provider(options) if provider is None else provider
        with contextlib.redirect_stdout(self.output):
            result = run_session(options, self.config(provider),
                                 model_kwargs=MODEL_KWARGS, **kwargs)
        self.assertTrue(provider.released.wait(1), provider.requests)
        return result, provider

    def checkpoint(self, branch):
        return self.model_root / branch / 'latest.zip'

    def metadata(self, branch):
        return json.loads(self.checkpoint(branch).with_suffix('.json').read_text(encoding='utf-8'))

    def test_duel_training_team_transfer_branch_resume_and_human_evaluation(self):
        from stable_baselines3 import PPO

        result, duel_provider = self.run_options(self.options())
        self.assertEqual(result, 0)
        duel_path = self.checkpoint('duel')
        original_duel_hash = file_hash(duel_path)
        original_duel_metadata = duel_path.with_suffix('.json').read_bytes()
        duel_model = PPO.load(str(duel_path), device='cpu')
        self.assertEqual(duel_model.num_timesteps, 16)
        self.assertEqual(duel_model.observation_space.shape, (210,))
        self.assertEqual(self.metadata('duel')['contract']['guidance_schema'], 'tactical-v1')
        self.assertGreaterEqual(duel_provider.episode, 2)
        self.assertEqual(self.metadata('duel')['last_session']['session_steps'], 16)
        self.assertEqual(self.metadata('duel')['last_session']['model_origin'], 'new')

        result, _ = self.run_options(self.options('online_teams'))
        self.assertEqual(result, 0)
        team_path = self.checkpoint('teams')
        first_team_hash = file_hash(team_path)
        team_model = PPO.load(str(team_path), device='cpu')
        self.assertEqual(team_model.num_timesteps, 16)
        lineage = self.metadata('teams')
        self.assertEqual(lineage['strategy'], 'warm_started')
        self.assertEqual(lineage['source_sha256'], original_duel_hash)
        self.assertEqual(Path(lineage['source_path']), duel_path.resolve())
        self.assertEqual(file_hash(duel_path), original_duel_hash)

        result, _ = self.run_options(self.options('online_teams'))
        self.assertEqual(result, 0)
        resumed = PPO.load(str(team_path), device='cpu')
        self.assertEqual(resumed.num_timesteps, 32)
        self.assertNotEqual(file_hash(team_path), first_team_hash)
        self.assertEqual(self.metadata('teams')['strategy'], 'resumed')
        self.assertEqual(self.metadata('teams')['last_session']['session_steps'], 16)
        self.assertEqual(self.metadata('teams')['initialization_strategy'], 'warm_started')
        self.assertEqual(file_hash(duel_path), original_duel_hash)

        before_teams_hash = file_hash(team_path)
        result, evaluation_provider = self.run_options(self.options('human_duel', steps=5))
        self.assertEqual(result, 0)
        self.assertEqual(sum(request['command'] == 'step' for request in evaluation_provider.requests), 5)
        self.assertEqual(file_hash(duel_path), original_duel_hash)
        self.assertEqual(duel_path.with_suffix('.json').read_bytes(), original_duel_metadata)
        self.assertEqual(file_hash(team_path), before_teams_hash)
        self.assertIn('model weights were not saved or changed', self.output.getvalue())

    def test_live_language_command_reaches_next_policy_decision_and_checkpoint(self):
        from guidance_io import write_guidance
        from language_commands import interpret_command
        from stable_baselines3.common.policies import ActorCriticPolicy
        from test_language_commands import OllamaFixture

        options = self.options(guidance={'opponent_spacing': 'close'})
        guidance_file = self.directory / 'guidance.json'
        write_guidance(guidance_file, options.guidance)
        language = OllamaFixture({'message': 'Keep distance and practise the input drill.',
                                  'needs_clarification': False,
                                  'patch': {'guidance': {'opponent_spacing': 'far',
                                                         'combo': ['down_light', 'jump_light']}}})
        self.addCleanup(language.close)
        sent = []
        observations = []
        original = ActorCriticPolicy.forward

        def command_during_match(request, response):
            if request['command'] == 'step' and not sent:
                result = interpret_command('Keep more distance and practise down light then jump light',
                                           options, language.config(timeout_seconds=2))
                write_guidance(guidance_file, result.options.guidance)
                sent.append(result)
            return response

        def record_observation(policy, observation, *args, **kwargs):
            observations.append(observation.detach().cpu().clone())
            return original(policy, observation, *args, **kwargs)

        provider = self.provider(options, alter=command_during_match)
        with patch.object(ActorCriticPolicy, 'forward', record_observation):
            result, _ = self.run_options(options, provider=provider, guidance_file=guidance_file)
        self.assertEqual(result, 0)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].changed)
        self.assertEqual(language.requests[0]['path'], '/api/chat')
        self.assertEqual(observations[0][0, 105], 1)  # Initial close-spacing goal.
        self.assertEqual(observations[1][0, 107], 1)  # New far goal after first action.
        self.assertEqual(self.metadata('duel')['last_session']['guidance'], sent[0].options.guidance)

    def test_unlimited_training_stops_by_file_releases_and_saves_progress(self):
        from stable_baselines3 import PPO

        observed_actions = threading.Event()
        stop_file = self.directory / 'stop.request'
        action_count = 0

        def observe(request, response):
            nonlocal action_count
            if request['command'] == 'step':
                action_count += 1
                if action_count >= 12:
                    observed_actions.set()
            return response

        def request_stop():
            if observed_actions.wait(5):
                stop_file.touch()

        options = self.options(steps=0, max_seconds=5)
        provider = self.provider(options, alter=observe)
        stopper = threading.Thread(target=request_stop, daemon=True)
        stopper.start()
        started = time.monotonic()
        result, _ = self.run_options(options, provider=provider, stop_file=stop_file)
        stopper.join(timeout=1)
        self.assertEqual(result, 0)
        self.assertTrue(stop_file.exists())
        self.assertFalse(stopper.is_alive())
        self.assertLess(time.monotonic() - started, 5)
        self.assertGreaterEqual(action_count, 12)
        self.assertLess(action_count, 128)
        model = PPO.load(str(self.checkpoint('duel')), device='cpu')
        self.assertGreaterEqual(model.num_timesteps, 8)
        self.assertEqual(self.metadata('duel')['last_session']['model_timesteps'], model.num_timesteps)
        self.assertIn('Session stopped: stop_requested', self.output.getvalue())

    def test_check_only_uses_no_gpu_session_actions_or_checkpoint(self):
        options = self.options(device='cuda')
        with patch('session_runner.choose_device', side_effect=AssertionError('GPU initialized')):
            result, provider = self.run_options(options, check_only=True)
        self.assertEqual(result, 0)
        self.assertEqual([request['command'] for request in provider.requests], ['hello', 'release'])
        self.assertFalse(self.model_root.exists())

    def test_check_rejects_online_and_controlled_slot_mismatches(self):
        for field, value in (('mode', 'offline_controlled'), ('controlled_player', 'Player2')):
            with self.subTest(field=field):
                options = self.options('online_duel')

                def alter(request, response):
                    if request['command'] == 'hello':
                        response[field] = value
                    return response

                provider = self.provider(options, alter=alter)
                with contextlib.redirect_stdout(self.output), self.assertRaises((ValueError, RuntimeError)):
                    run_session(options, self.config(provider), check_only=True)
                self.assertTrue(provider.released.wait(1), provider.requests)
                self.assertEqual([request['command'] for request in provider.requests], ['hello', 'release'])
                self.assertFalse(self.model_root.exists())

    def test_human_evaluation_requires_existing_duel_without_arranging_a_match(self):
        options = self.options('human_duel', steps=3)
        provider = self.provider(options)
        with contextlib.redirect_stdout(self.output), self.assertRaises(FileNotFoundError):
            run_session(options, self.config(provider), model_kwargs=MODEL_KWARGS)
        self.assertTrue(provider.released.wait(1))
        self.assertEqual([request['command'] for request in provider.requests], ['hello', 'release'])
        self.assertFalse(self.model_root.exists())


class SessionCliTests(unittest.TestCase):
    def test_dry_run_validates_all_profiles_without_numeric_or_desktop_dependencies(self):
        for name in PROFILE_PRESETS:
            with self.subTest(profile=name):
                result = subprocess.run(
                    [sys.executable, '-c', DRY_RUN_IMPORT_GUARD, str(ROOT / 'agent.py'),
                     'session', '--profile', name, '--dry-run'],
                    cwd=ROOT, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                settings = json.loads(result.stdout)
                self.assertEqual(settings['profile']['profile'], name)
                self.assertEqual(settings['steps'], 0)

    def test_saved_session_cannot_be_mixed_with_profile_or_option_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'options.json'
            saved.write_text(json.dumps(default_options().to_dict()), encoding='utf-8')
            for arguments in (['--profile', 'online_duel'], ['--steps', '8'], ['--learn'],
                              ['--self-player', 'Player3']):
                with self.subTest(arguments=arguments), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()) as errors:
                    result = main(['--config', str(ROOT / 'config.json'),
                                   '--session-file', str(saved), '--dry-run', *arguments])
                    self.assertEqual(result, 2)
                    self.assertIn('not both', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
