"""Check causal goal timing, bounded rewards and untouched policy actions."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from controls import ACTIONS
from dual_team_env import DualTeamVecEnv
from guidance_settings import ACTION_LABELS, GUIDANCE_FEATURE_NAMES
from tactical_guidance import TacticalGuidanceVecEnv
from team_state import TeamRewardTracker, TeamState
from test_dual_team_env import SharedWorldFixture


class ScriptedGame(gym.Env):
    """A coherent small state source for reward timing, not a live game."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32)
    action_space = gym.spaces.Discrete(len(ACTIONS))

    def __init__(self, *, distances=(500, 400, 300, 200), damages=(0, 0, 10, 20),
                 done_at=99, truncated=False, discontinuity_at=None, eliminated_at=None,
                 episode_change_at=None, teams=False):
        self.distances, self.damages = distances, damages
        self.done_at, self.truncated = done_at, truncated
        self.discontinuity_at, self.eliminated_at = discontinuity_at, eliminated_at
        self.episode_change_at, self.teams = episode_change_at, teams
        self.count, self.episode = 0, 0
        self.actions = []
        self.tracker = TeamRewardTracker()

    def state(self):
        distance = self.distances[min(self.count, len(self.distances) - 1)]
        damage = self.damages[min(self.count, len(self.damages) - 1)]
        players = {
            'Player1': {'x': 0, 'y': 0, 'CurrDamage': 0, 'Eliminated': False, 'BrawlhallaID': 1},
            'Player3': {'x': distance, 'y': 0, 'CurrDamage': damage,
                        'Eliminated': self.count == self.eliminated_at, 'BrawlhallaID': 3},
        }
        if self.teams:
            players['Player2'] = {'x': 200, 'y': 0, 'CurrDamage': 0, 'Eliminated': False, 'BrawlhallaID': 2}
            players['Player4'] = {'x': 800, 'y': 0, 'CurrDamage': damage, 'Eliminated': False, 'BrawlhallaID': 4}
        return TeamState.from_players(players, 'Player1', ('Player3', 'Player4') if self.teams else ('Player3',),
                                      'Player2' if self.teams else None)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        self.episode += 1
        self.tracker.reset()
        state = self.state()
        self.tracker.update(state)
        return state.to_vector(), {}

    def step(self, action):
        self.actions.append(int(action))
        self.count += 1
        state = self.state()
        projection = self.tracker.update(state)
        from dataclasses import asdict
        info = {'session_id': 'scripted', 'episode_id': f'match-{self.episode}',
                'tick': self.count, 'projection': asdict(projection)}
        if self.count == self.discontinuity_at:
            info['projection']['discontinuity'] = True
        if self.episode_change_at is not None and self.count >= self.episode_change_at:
            info['episode_id'] = 'unannounced-next-match'
        done = self.count >= self.done_at
        return state.to_vector(), projection.reward, done and not self.truncated, done and self.truncated, info


class TacticalGuidanceTests(unittest.TestCase):
    def env(self, guidance=None, *, learning=True, guidance_file=None, **game_options):
        game = ScriptedGame(**game_options)
        frames = VecFrameStack(DummyVecEnv([lambda: game]), n_stack=4)
        env = TacticalGuidanceVecEnv(frames, guidance, learning=learning, guidance_file=guidance_file)
        self.addCleanup(env.close)
        return env, game, frames

    def test_suffix_encodes_exact_goals_and_leaves_history_unchanged(self):
        env, _, frames = self.env({'opponent_spacing': 'mid', 'teammate_spacing': 'spread',
                                   'strength': 0.6, 'combo': ['jump', 'light']})
        observation = env.reset()
        self.assertEqual(observation.shape, (1, 210))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_array_equal(observation[:, :104], frames.stacked_obs.stacked_obs)
        suffix = observation[0, 104:]
        self.assertEqual(suffix[2], 1)
        self.assertEqual(suffix[6], 1)
        self.assertAlmostEqual(suffix[7], 0.6)
        self.assertEqual(suffix[8 + ACTION_LABELS.index('jump')], 1)
        self.assertEqual(suffix[8 + 16 + ACTION_LABELS.index('light')], 1)
        self.assertAlmostEqual(suffix[-2], 2 / 6)
        self.assertEqual(suffix[-1], 0)
        observation, _, _, _ = env.step(np.asarray([ACTION_LABELS.index('jump')]))
        np.testing.assert_array_equal(observation[:, :104], frames.stacked_obs.stacked_obs)
        self.assertAlmostEqual(observation[0, -1], 1 / 6)
        self.assertEqual(len(GUIDANCE_FEATURE_NAMES), 106)

    def test_positional_shaping_is_bounded_and_stationary_idle_has_no_positive_bonus(self):
        env, _, _ = self.env({'opponent_spacing': 'close', 'strength': 1},
                             distances=(500, 400, 400, 150), damages=(0, 0, 0, 0))
        env.reset()
        _, _, _, infos = env.step([1])
        first = infos[0]['guidance']['position_bonus']
        self.assertGreater(first, 0)
        self.assertLessEqual(abs(first), 0.01)
        _, reward, _, infos = env.step([0])
        self.assertLessEqual(infos[0]['guidance']['position_bonus'], 0)
        self.assertLessEqual(reward[0], 0)
        _, _, _, infos = env.step([1])
        self.assertLessEqual(abs(infos[0]['guidance']['position_bonus']), 0.01)

    def test_exact_drill_preserves_actions_and_needs_observed_damage(self):
        env, game, _ = self.env({'combo': ['jump', 'light'], 'strength': 1})
        env.reset()
        env.step([3])
        _, reward, _, infos = env.step([5])
        self.assertEqual(game.actions, [3, 5])
        self.assertTrue(infos[0]['guidance']['drill_completed'])
        self.assertAlmostEqual(infos[0]['guidance']['combo_bonus'], 0.02)
        self.assertAlmostEqual(reward[0], 0.12)
        self.assertEqual(infos[0]['guidance']['next_combo_progress'], 0)
        env, _, _ = self.env({'combo': ['jump', 'light'], 'strength': 1}, damages=(0, 0, 0))
        env.reset()
        env.step([3])
        _, reward, _, infos = env.step([5])
        self.assertTrue(infos[0]['guidance']['drill_completed'])
        self.assertEqual(infos[0]['guidance']['combo_bonus'], 0)
        self.assertEqual(reward[0], 0)

    def test_wrong_action_resets_drill_and_unrelated_damage_gets_no_bonus(self):
        env, game, _ = self.env({'combo': ['jump', 'light'], 'strength': 1})
        env.reset()
        env.step([3])
        _, _, _, infos = env.step([6])
        self.assertEqual(infos[0]['guidance']['next_combo_progress'], 0)
        _, _, _, infos = env.step([5])
        self.assertEqual(game.actions, [3, 6, 5])
        self.assertFalse(infos[0]['guidance']['drill_completed'])
        self.assertEqual(infos[0]['guidance']['combo_bonus'], 0)

    def test_repeated_prefix_drill_matches_the_actual_consecutive_action_suffix(self):
        env, game, _ = self.env({'combo': ['light', 'light', 'heavy'], 'strength': 1},
                               damages=(0, 0, 0, 0, 10))
        env.reset()
        for action in (5, 5, 5):
            _, _, _, infos = env.step([action])
            self.assertFalse(infos[0]['guidance']['drill_completed'])
        self.assertEqual(infos[0]['guidance']['next_combo_progress'], 2)
        _, _, _, infos = env.step([6])
        self.assertEqual(game.actions, [5, 5, 5, 6])
        self.assertTrue(infos[0]['guidance']['drill_completed'])
        self.assertAlmostEqual(infos[0]['guidance']['combo_bonus'], 0.02)

    def test_team_bonus_is_disabled_and_actor_progress_is_independent(self):
        client = SharedWorldFixture(episode_steps=99)
        base = DualTeamVecEnv(client, client.profile)
        frames = VecFrameStack(base, n_stack=4)
        env = TacticalGuidanceVecEnv(frames, {'combo': ['jump', 'light'], 'strength': 1})
        self.addCleanup(env.close)
        env.reset()
        observation, _, _, infos = env.step(np.asarray([3, 6]))
        np.testing.assert_allclose(observation[:, -1], [1 / 6, 0])
        observation, rewards, _, infos = env.step(np.asarray([5, 3]))
        np.testing.assert_allclose(observation[:, -1], [0, 1 / 6])
        self.assertTrue(infos[0]['guidance']['drill_completed'])
        self.assertFalse(infos[1]['guidance']['drill_completed'])
        self.assertTrue(all(info['guidance']['combo_bonus'] == 0 for info in infos))
        self.assertTrue(all(any('cannot be attributed' in message for message in info['guidance']['diagnostics']) for info in infos))
        np.testing.assert_allclose(rewards, [0.14, 0.14])

    def test_discontinuities_elimination_and_unannounced_episode_changes_reset_bonus(self):
        for options in ({'discontinuity_at': 2}, {'eliminated_at': 2}, {'episode_change_at': 2}):
            with self.subTest(options=options):
                env, _, _ = self.env({'opponent_spacing': 'close', 'combo': ['jump', 'light'], 'strength': 1}, **options)
                env.reset()
                env.step([3])
                _, _, _, infos = env.step([5])
                self.assertEqual(infos[0]['guidance']['total_bonus'], 0)
                self.assertEqual(infos[0]['guidance']['next_combo_progress'], 0)

    def test_live_goal_change_scores_prior_goal_and_encodes_new_goal_after_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'guidance.json'
            path.write_text(json.dumps({'opponent_spacing': 'close', 'strength': 1}), encoding='utf-8')
            env, _, _ = self.env(guidance_file=path, damages=(0, 0, 0))
            first = env.reset()
            self.assertEqual(first[0, 105], 1)  # close in the suffix
            path.write_text(json.dumps({'opponent_spacing': 'far', 'strength': 1}), encoding='utf-8')
            observation, _, _, infos = env.step([1])
            report = infos[0]['guidance']
            self.assertEqual(report['active']['opponent_spacing'], 'close')
            self.assertEqual(report['next']['opponent_spacing'], 'far')
            self.assertGreater(report['position_bonus'], 0)
            self.assertEqual(observation[0, 107], 1)  # far in the next suffix
            self.assertEqual(observation[0, 105], 0)
            _, _, _, infos = env.step([1])
            self.assertEqual(infos[0]['guidance']['active']['opponent_spacing'], 'far')
            self.assertLess(infos[0]['guidance']['position_bonus'], 0)

    def test_bad_live_file_keeps_goal_and_reports_warning_without_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'guidance.json'
            path.write_text(json.dumps({'combo': ['jump', 'light']}), encoding='utf-8')
            env, game, _ = self.env(guidance_file=path)
            env.reset()
            path.write_text('{"combo":', encoding='utf-8')
            with self.assertLogs('tactical_guidance', level='WARNING'):
                _, _, _, infos = env.step([3])
            self.assertEqual(env.guidance['combo'], ['jump', 'light'])
            self.assertTrue(any('update ignored' in message for message in infos[0]['guidance']['diagnostics']))
            _, _, _, infos = env.step([5])
            self.assertEqual(game.actions, [3, 5])
            self.assertTrue(infos[0]['guidance']['drill_completed'])

    def test_terminal_observation_keeps_old_goal_and_prefix_then_resets_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'guidance.json'
            path.write_text(json.dumps({'opponent_spacing': 'close', 'combo': ['jump', 'light']}), encoding='utf-8')
            env, _, _ = self.env(guidance_file=path, done_at=1, truncated=True)
            env.reset()
            path.write_text(json.dumps({'opponent_spacing': 'far'}), encoding='utf-8')
            observation, _, done, infos = env.step([3])
            self.assertTrue(done[0])
            terminal = infos[0]['terminal_observation']
            self.assertEqual(terminal.shape, (210,))
            np.testing.assert_allclose(terminal[78 + 8], 0.4)
            self.assertEqual(terminal[105], 1)
            self.assertAlmostEqual(terminal[-1], 1 / 6)
            self.assertEqual(observation[0, 107], 1)
            self.assertEqual(observation[0, -1], 0)
            expected_bonus = 0.003 * (0.99 * math.exp(-250 / 200) - math.exp(-350 / 200))
            self.assertAlmostEqual(infos[0]['guidance']['total_bonus'], expected_bonus)
            self.assertTrue(infos[0]['TimeLimit.truncated'])

    def test_true_terminal_uses_absorbing_potential_and_truncation_uses_terminal_frame(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                env, _, _ = self.env({'opponent_spacing': 'close', 'strength': 1},
                                     distances=(500, 150), damages=(0, 0), done_at=1, truncated=truncated)
                env.reset()
                _, reward, done, infos = env.step([1])
                self.assertTrue(done[0])
                initial = 0.01 * math.exp(-350 / 200)
                expected = 0.99 * 0.01 - initial if truncated else -initial
                self.assertAlmostEqual(infos[0]['guidance']['position_bonus'], expected)
                self.assertAlmostEqual(reward[0], expected)
                # The next match starts at distance 500; it must not inherit the
                # terminal state's distance 150 as its potential baseline.
                _, next_reward, _, next_infos = env.step([1])
                self.assertAlmostEqual(next_infos[0]['guidance']['position_bonus'], expected)
                self.assertAlmostEqual(next_reward[0], expected)

    def test_shaping_discount_can_match_a_loaded_policy(self):
        env, _, _ = self.env({'opponent_spacing': 'close', 'strength': 1}, damages=(0, 0))
        env.shaping_gamma = 0.5
        env.reset()
        _, reward, _, infos = env.step([1])
        expected = 0.01 * (0.5 * math.exp(-250 / 200) - math.exp(-350 / 200))
        self.assertAlmostEqual(infos[0]['guidance']['position_bonus'], expected)
        self.assertAlmostEqual(reward[0], expected)
        for invalid in (-1, 1.1, True, float('nan'), '0.99'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                env.shaping_gamma = invalid

    def test_evaluation_and_zero_strength_add_no_bonus(self):
        for guidance, learning in (({'opponent_spacing': 'close', 'combo': ['jump', 'light'], 'strength': 1}, False),
                                   ({'opponent_spacing': 'close', 'combo': ['jump', 'light'], 'strength': 0}, True)):
            with self.subTest(learning=learning, strength=guidance['strength']):
                env, _, _ = self.env(guidance, learning=learning)
                env.reset()
                env.step([3])
                observation, reward, _, infos = env.step([5])
                self.assertEqual(infos[0]['guidance']['total_bonus'], 0)
                self.assertAlmostEqual(reward[0], 0.1)
                self.assertEqual(observation[0, 105], 1)


if __name__ == '__main__':
    unittest.main()
