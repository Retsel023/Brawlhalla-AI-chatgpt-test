"""Two teammates must share one world step, reset and policy update stream."""

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
import torch

from controls import ACTIONS
from dual_team_env import DualTeamVecEnv
from session_bridge import SessionSnapshot
from session_profiles import make_profile
from state_bridge import BridgeError
from team_state import TeamState


class SharedWorldFixture:
    """A single synchronous world with explicit joint actions and snapshots."""

    game_build = '11.00.36'

    def __init__(self, profile=None, *, episode_steps=3, alter=None):
        self.profile = make_profile('online_teams', teammate_kind='ai') if profile is None else profile
        self.controlled_players = [self.profile['self_player'], self.profile['teammate_player']]
        self.episode_steps = episode_steps
        self.alter = alter
        self.calls = []
        self.close_calls = 0
        self.episode = 0
        self.tick = 0
        self.players = None

    def validate_control(self, profile):
        if set(self.controlled_players) != {profile['self_player'], profile['teammate_player']}:
            raise BridgeError('provider does not control both AI slots')

    def configure(self, profile):
        self.validate_control(profile)
        self.calls.append(('configure', copy.deepcopy(profile)))
        if profile != self.profile:
            raise BridgeError('wrong profile')
        return 'shared-session'

    def snapshot(self):
        snapshot = SessionSnapshot(f'match-{self.episode}', self.tick, copy.deepcopy(self.players),
                                   self.tick >= self.episode_steps, False,
                                   'shared-session', copy.deepcopy(self.profile))
        return snapshot if self.alter is None else self.alter(snapshot)

    def next_episode(self, seed=None):
        self.calls.append(('next_episode', seed))
        self.episode += 1
        self.tick = 0
        self.players = {f'Player{index}': {
            'x': index * 100, 'y': -index * 10, 'CurrDamage': 0, 'Eliminated': False,
            'BrawlhallaID': self.episode * 100 + index, 'PlayerName': f'P{index}',
        } for index in range(1, 5)}
        return self.snapshot()

    def step_joint(self, actions, hold_seconds):
        self.calls.append(('step_joint', copy.deepcopy(actions), hold_seconds))
        if set(actions) != set(self.controlled_players):
            raise BridgeError('both controlled actors must be present together')
        self.tick += 1
        self.players['Player1']['x'] += 10
        self.players['Player2']['x'] += 20
        for slot, increment in (('Player1', 2), ('Player2', 4), ('Player3', 10), ('Player4', 10)):
            self.players[slot]['CurrDamage'] += increment
        return self.snapshot()

    def close(self):
        self.close_calls += 1


class DualTeamVecEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def env(self, **kwargs):
        client = SharedWorldFixture(**kwargs)
        env = DualTeamVecEnv(client, client.profile)
        self.addCleanup(env.close)
        return env, client

    def test_one_joint_world_step_maps_both_actors_and_their_perspectives(self):
        env, client = self.env(episode_steps=99)
        initial = env.reset()
        self.assertEqual(initial.shape, (2, 26))
        self.assertEqual(initial.dtype, np.float32)
        self.assertEqual(env.num_envs, 2)
        self.assertEqual(initial[:, 0].tolist(), [np.float32(0.1), np.float32(0.2)])
        self.assertEqual(initial[:, 10].tolist(), [np.float32(0.2), np.float32(0.1)])
        left, right = ACTIONS.index(('left',)), ACTIONS.index(('right',))
        env.step_async(np.asarray([left, right]))
        self.assertFalse(any(call[0] == 'step_joint' for call in client.calls))
        observation, reward, done, infos = env.step_wait()
        joint = [call for call in client.calls if call[0] == 'step_joint']
        self.assertEqual(len(joint), 1)
        self.assertEqual(joint[0][1], {'Player1': ('left',), 'Player2': ('right',)})
        np.testing.assert_allclose(reward, [0.14, 0.14])
        np.testing.assert_array_equal(done, [False, False])
        for index, slot in enumerate(('Player1', 'Player2')):
            expected = TeamState.from_players(client.players, slot, ('Player3', 'Player4'),
                                              ('Player2', 'Player1')[index]).to_vector()
            np.testing.assert_array_equal(observation[index], expected)
            self.assertEqual(infos[index]['controlled_player'], slot)
            self.assertTrue(infos[index]['shared_world'])
            self.assertEqual(infos[index]['tick'], 1)
            self.assertEqual(infos[index]['projection']['damage_dealt'], 20)
            self.assertEqual(infos[index]['projection']['damage_received'], 6)

    def test_actor_frame_histories_stay_separate_and_terminal_reset_is_shared(self):
        base, client = self.env(episode_steps=2)
        env = VecFrameStack(base, n_stack=4)
        self.addCleanup(env.close)
        first = env.reset()
        np.testing.assert_array_equal(first[:, :78], np.zeros((2, 78)))
        second, _, _, _ = env.step(np.asarray([1, 2]))
        np.testing.assert_array_equal(second[:, -52:-26], first[:, -26:])
        np.testing.assert_allclose(second[:, -26], [0.11, 0.22])
        fresh, rewards, dones, infos = env.step(np.asarray([2, 1]))
        np.testing.assert_array_equal(dones, [True, True])
        np.testing.assert_allclose(rewards, [0.14, 0.14])
        self.assertEqual(sum(call[0] == 'configure' for call in client.calls), 1)
        self.assertEqual(sum(call[0] == 'next_episode' for call in client.calls), 2)
        self.assertEqual(sum(call[0] == 'step_joint' for call in client.calls), 2)
        np.testing.assert_array_equal(fresh[:, :78], np.zeros((2, 78)))
        for index, expected_x in enumerate((0.12, 0.24)):
            self.assertEqual(infos[index]['terminal_observation'].shape, (104,))
            np.testing.assert_allclose(infos[index]['terminal_observation'][-26], expected_x)
            self.assertFalse(infos[index]['TimeLimit.truncated'])
            self.assertEqual(infos[index]['episode_id'], 'match-1')
            self.assertEqual(base.reset_infos[index]['episode_id'], 'match-2')

    def test_truncation_uses_provider_flags_and_elimination_proxy_does_not_reset(self):
        def alter(snapshot):
            if snapshot.tick:
                snapshot.players['Player3']['Eliminated'] = True
                return replace(snapshot, truncated=snapshot.tick == 2)
            return snapshot

        env, client = self.env(episode_steps=99, alter=alter)
        env.reset()
        _, _, done, infos = env.step([0, 0])
        self.assertFalse(done.any())
        self.assertTrue(all(info['projection']['terminal_proxy'] for info in infos))
        self.assertEqual(client.episode, 1)
        _, _, done, infos = env.step([0, 0])
        self.assertTrue(done.all())
        self.assertTrue(all(info['TimeLimit.truncated'] for info in infos))
        self.assertEqual(client.episode, 2)

    def test_invalid_actions_close_without_advancing_shared_world(self):
        for actions in ([0], [0, 1, 2], [0.0, 1.0], [True, False], [True, 0], [0, False],
                        [-1, 0], [len(ACTIONS), 0], [[0], [1]]):
            with self.subTest(actions=actions):
                env, client = self.env()
                env.reset()
                with self.assertRaises(ValueError):
                    env.step_async(actions)
                self.assertEqual(client.close_calls, 1)
                self.assertFalse(any(call[0] == 'step_joint' for call in client.calls))

    def test_pending_actions_require_exactly_one_async_and_wait_pair(self):
        env, client = self.env()
        env.reset()
        with self.assertRaises(BridgeError):
            env.step_wait()
        self.assertEqual(client.close_calls, 1)
        env, client = self.env()
        env.reset()
        env.step_async([0, 1])
        with self.assertRaises(BridgeError):
            env.step_async([1, 0])
        self.assertFalse(any(call[0] == 'step_joint' for call in client.calls))
        self.assertEqual(client.close_calls, 1)

    def test_identity_roster_profile_and_tick_changes_fail_closed(self):
        def identity(snapshot):
            snapshot.players['Player2']['BrawlhallaID'] = 999
            return snapshot

        def roster(snapshot):
            snapshot.players['Player9'] = snapshot.players.pop('Player4')
            return snapshot

        def roles(snapshot):
            snapshot.profile['opponents'].reverse()
            return snapshot

        for mutation in (identity, roster, roles, lambda snapshot: replace(snapshot, tick=0),
                         lambda snapshot: replace(snapshot, terminated=1),
                         lambda snapshot: replace(snapshot, session_id='changed')):
            with self.subTest(mutation=mutation):
                env, client = self.env(alter=lambda snapshot: mutation(snapshot) if snapshot.tick else snapshot)
                env.reset()
                with self.assertRaises(BridgeError):
                    env.step([0, 0])
                self.assertEqual(client.close_calls, 1)

    def test_damage_decrease_rebaselines_both_team_perspectives(self):
        def respawn(snapshot):
            if snapshot.tick == 2:
                snapshot.players['Player2']['CurrDamage'] = 0
            return snapshot

        env, _ = self.env(episode_steps=99, alter=respawn)
        env.reset()
        env.step([0, 0])
        _, rewards, _, infos = env.step([0, 0])
        np.testing.assert_array_equal(rewards, [0, 0])
        self.assertTrue(all(info['projection']['discontinuity'] for info in infos))
        self.assertTrue(all('Player2_damage_decreased' in info['projection']['diagnostics'] for info in infos))

    def test_both_slots_require_control_acknowledgment_and_active_new_match(self):
        env, client = self.env()
        client.controlled_players = ['Player1']
        with self.assertRaises(BridgeError):
            env.reset()
        self.assertEqual(client.close_calls, 1)
        self.assertFalse(client.calls)

        def inactive(snapshot):
            snapshot.players['Player4']['Eliminated'] = True
            return snapshot

        env, client = self.env(alter=inactive)
        with self.assertRaises(BridgeError):
            env.reset()
        self.assertEqual(client.close_calls, 1)
        client = SharedWorldFixture()
        with self.assertRaises(ValueError):
            DualTeamVecEnv(client, make_profile('online_teams'))

    def test_world_seed_is_shared_and_helpers_cannot_reset_individual_actors(self):
        env, client = self.env()
        self.assertEqual(env.seed(23), [23, 23])
        env.reset()
        self.assertEqual(client.calls[1], ('next_episode', 23))
        self.assertEqual(env.get_attr('controlled_player'), ['Player1', 'Player2'])
        self.assertEqual(env.get_attr('teammate_player', indices=1), ['Player1'])
        self.assertEqual(env.env_is_wrapped(gym.Wrapper), [False, False])
        self.assertEqual(env.get_attr('render_mode'), [None, None])
        with self.assertRaises(AttributeError):
            env.env_method('reset', indices=0)
        with self.assertRaises(ValueError):
            env.set_attr('hold_seconds', 0.2, indices=1)
        env.set_attr('hold_seconds', 0.2)
        self.assertEqual(env.hold_seconds, 0.2)
        with self.assertRaises(AttributeError):
            env.set_attr('profile', {})
        with self.assertRaises(ValueError):
            env.set_options([{'different': True}, {}])
        env.close()
        env.close()
        self.assertEqual(client.close_calls, 1)

    def test_real_ppo_trains_two_perspectives_with_one_world_advance_per_pair(self):
        base, client = self.env(episode_steps=3)
        env = VecFrameStack(base, n_stack=4)
        self.addCleanup(env.close)
        model = PPO('MlpPolicy', env, n_steps=4, batch_size=8, n_epochs=1, seed=19,
                    policy_kwargs={'net_arch': {'pi': [16], 'vf': [16]}}, device='cpu', verbose=0)
        before = {name: value.clone() for name, value in model.policy.state_dict().items()}
        model.learn(16)
        self.assertEqual(model.num_timesteps, 16)
        self.assertEqual(sum(call[0] == 'step_joint' for call in client.calls), 8)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.policy.state_dict().items()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shared.zip'
            model.save(str(path))
            restored = PPO.load(str(path), env=env, device='cpu')
            self.assertEqual(restored.n_envs, 2)
            self.assertEqual(restored.observation_space.shape, (104,))
            restored.learn(8, reset_num_timesteps=False)
            self.assertEqual(restored.num_timesteps, 24)

    def test_real_protocol3_joint_control_maps_custom_slots_and_one_match_reset(self):
        from team_control_bridge import TeamControlClient
        from test_team_control_bridge import TeamControlProviderFixture

        profile = make_profile('online_teams', teammate_kind='ai', self_player='Player7',
                               teammate_player='Player12', opponents=['Player3', 'Player4'])
        provider = TeamControlProviderFixture(profile, episode_steps=1)
        self.addCleanup(provider.close)
        client = TeamControlClient(port=provider.port, expected_build='11.00.36')
        self.addCleanup(client.close)
        env = DualTeamVecEnv(client, profile)
        self.addCleanup(env.close)
        initial = env.reset()
        self.assertEqual(initial.shape, (2, 26))
        _, rewards, dones, infos = env.step(np.asarray([8, 11]))
        self.assertTrue(dones.all())
        np.testing.assert_allclose(rewards, [0.14, 0.14])
        self.assertEqual([info['controlled_player'] for info in infos], ['Player7', 'Player12'])
        commands = [request['command'] for request in provider.requests]
        self.assertEqual(commands, ['hello', 'configure', 'next_episode', 'step_joint', 'next_episode'])
        joint = provider.requests[3]
        self.assertEqual(joint['actions'], {'Player7': list(ACTIONS[8]), 'Player12': list(ACTIONS[11])})
        env.close()
        self.assertTrue(provider.released.wait(1))


if __name__ == '__main__':
    unittest.main()
