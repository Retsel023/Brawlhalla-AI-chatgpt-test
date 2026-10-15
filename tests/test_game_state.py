"""Verify state features and reward behavior across real feed ambiguities."""

from dataclasses import asdict
import unittest

import numpy as np

from game_state import FEATURE_NAMES, GameState, PlayerState, RewardTracker


def state(own_damage=0, other_damage=0, *, own_id=1, other_id=2,
          own_eliminated=False, other_eliminated=False):
    players = {
        "Player2": {"x": -100, "y": 50, "CurrDamage": own_damage,
                    "Eliminated": own_eliminated},
        "Player7": {"x": 300, "y": -150, "CurrDamage": other_damage,
                    "Eliminated": other_eliminated},
    }
    for slot, identity in (("Player2", own_id), ("Player7", other_id)):
        if identity is not None:
            players[slot]["BrawlhallaID"] = identity
    return GameState.from_players(players, "Player2", "Player7")


class GameStateTests(unittest.TestCase):
    def test_explicit_slot_selection_and_vector_order(self):
        current = state(20, 40, other_eliminated=True)
        observation = current.to_vector(position_scale=200, damage_scale=50)
        self.assertEqual(observation.shape, (10,))
        self.assertEqual(len(FEATURE_NAMES), 10)
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_allclose(observation,
                                   [-0.5, 0.25, 0.4, 0, 1.5, -0.75, 0.8, 1, 2, -1])
        self.assertEqual(current.self_player.slot, "Player2")
        self.assertEqual(current.opponent.brawlhalla_id, 2)

    def test_slots_must_exist_and_be_distinct_without_auto_selection(self):
        raw = {"Player1": {"x": 0, "y": 0, "CurrDamage": 0, "Eliminated": False}}
        for own, other in (("Player1", "Player1"), ("Player1", "Player2"),
                           (None, "Player1"), ("self", "Player1")):
            with self.subTest(own=own, other=other), self.assertRaises(ValueError):
                GameState.from_players(raw, own, other)
        with self.assertRaises(ValueError):
            GameState(state().self_player, state().self_player)

    def test_scale_and_finite_float32_validation(self):
        for scale in (0, -1, True, "10", float("nan"), float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    state().to_vector(position_scale=scale)
                with self.assertRaises(ValueError):
                    state().to_vector(damage_scale=scale)
                with self.assertRaises(ValueError):
                    RewardTracker(damage_scale=scale)
        huge = GameState(PlayerState("Player1", 1e308, 0, 0, False),
                         PlayerState("Player2", -1e308, 0, 0, False))
        with self.assertRaises(ValueError):
            huge.to_vector()
        np.testing.assert_allclose(huge.to_vector(position_scale=1e308),
                                   [1, 0, 0, 0, -1, 0, 0, 0, -2, 0])
        with self.assertRaises(ValueError):
            state().to_vector(position_scale=1e-308)
        for field, value in (("x", float("nan")), ("y", True), ("damage", -1),
                             ("eliminated", 1), ("brawlhalla_id", True),
                             ("player_name", 1)):
            values = dict(slot="Player1", x=0, y=0, damage=0, eliminated=False)
            values[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                PlayerState(**values)

    def test_damage_rewards_measure_only_new_damage(self):
        tracker = RewardTracker()
        baseline = tracker.update(state(10, 20))
        self.assertTrue(baseline.baseline_reset)
        self.assertFalse(baseline.discontinuity)
        self.assertEqual(baseline.reward, 0)
        result = tracker.update(state(15, 50))
        self.assertAlmostEqual(result.reward, 0.25)
        self.assertEqual((result.damage_dealt, result.damage_received), (30, 5))
        self.assertFalse(result.terminal_proxy)
        self.assertEqual(tracker.update(state(15, 50)).reward, 0)
        penalty = tracker.update(state(35, 50))
        self.assertAlmostEqual(penalty.reward, -0.2)
        self.assertIn("damage_received", asdict(penalty))

    def test_damage_drop_rebaselines_entire_ambiguous_transition(self):
        tracker = RewardTracker()
        tracker.update(state(150, 80))
        respawn = tracker.update(state(0, 90))
        self.assertEqual(respawn.reward, 0)
        self.assertEqual(respawn.damage_dealt, 0)
        self.assertTrue(respawn.baseline_reset)
        self.assertTrue(respawn.discontinuity)
        self.assertIn("Player2_damage_decreased", respawn.diagnostics)
        self.assertAlmostEqual(tracker.update(state(5, 110)).reward, 0.15)
        other_respawn = tracker.update(state(5, 0))
        self.assertEqual(other_respawn.reward, 0)
        self.assertIn("Player7_damage_decreased", other_respawn.diagnostics)
        self.assertAlmostEqual(tracker.update(state(5, 10)).reward, 0.1)

    def test_identity_changes_and_missing_identity_rebaseline(self):
        tracker = RewardTracker()
        tracker.update(state(0, 0))
        for current in (state(0, 100, other_id=3), state(0, 120, other_id=None),
                        state(0, 130, other_id=3)):
            changed = tracker.update(current)
            self.assertEqual(changed.reward, 0)
            self.assertTrue(changed.discontinuity)
            self.assertIn("Player7_identity_changed", changed.diagnostics)
        self.assertAlmostEqual(tracker.update(state(0, 150, other_id=3)).reward, 0.2)

    def test_names_and_slot_assignments_cannot_silently_switch_players(self):
        tracker = RewardTracker()
        tracker.update(state())
        own = PlayerState("Player2", -100, 50, 10, False, 1, "New name")
        renamed = tracker.update(GameState(own, state(other_damage=40).opponent))
        self.assertTrue(renamed.discontinuity)
        self.assertIn("Player2_identity_changed", renamed.diagnostics)
        current = state()
        switched = tracker.update(GameState(current.opponent, current.self_player))
        self.assertEqual(switched.reward, 0)
        self.assertTrue(switched.discontinuity)
        self.assertTrue(any("slot_changed" in item for item in switched.diagnostics))

    def test_elimination_edges_are_reported_once_without_stock_bonus(self):
        tracker = RewardTracker()
        tracker.update(state(0, 100))
        eliminated = tracker.update(state(0, 120, other_eliminated=True))
        self.assertAlmostEqual(eliminated.reward, 0.2)
        self.assertTrue(eliminated.terminal_proxy)
        self.assertEqual(eliminated.eliminated_slots, ("Player7",))
        repeated = tracker.update(state(0, 120, other_eliminated=True))
        self.assertEqual(repeated.reward, 0)
        self.assertFalse(repeated.terminal_proxy)
        self.assertEqual(repeated.eliminated_slots, ())
        tracker.reset()
        initial_eliminated = tracker.update(state(0, 120, other_eliminated=True))
        self.assertFalse(initial_eliminated.terminal_proxy)
        self.assertTrue(initial_eliminated.baseline_reset)

    def test_unknown_identity_is_allowed_but_not_invented(self):
        tracker = RewardTracker()
        current = state(own_id=None, other_id=None)
        self.assertEqual(current.self_player.identity, (None, None))
        tracker.update(current)
        result = tracker.update(state(other_damage=10, own_id=None, other_id=None))
        self.assertAlmostEqual(result.reward, 0.1)
        self.assertFalse(result.discontinuity)

    def test_overflowing_reward_is_rejected_without_consuming_baseline(self):
        tracker = RewardTracker(damage_scale=1e308)
        tracker.update(state())
        with self.assertRaises(ValueError):
            tracker.update(state(other_damage=100))
        self.assertEqual(tracker.update(state(other_damage=1)).reward, 1e308)


if __name__ == "__main__":
    unittest.main()
