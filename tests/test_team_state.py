"""Check explicit team roles, transfer-compatible features and team rewards."""

from dataclasses import asdict
import unittest

import numpy as np

from game_state import GameState, PlayerState, RewardTracker
from team_state import FEATURE_COUNT, FEATURE_NAMES, TeamRewardTracker, TeamState


def payload(damage=(10, 20, 30, 40)):
    return {
        f"Player{index}": {"x": index * 100, "y": index * -50,
                           "CurrDamage": value, "Eliminated": False,
                           "BrawlhallaID": index, "PlayerName": f"P{index}"}
        for index, value in enumerate(damage, 1)
    }


def team(players=None, **kwargs):
    return TeamState.from_players(payload() if players is None else players,
                                  "Player1", ("Player3", "Player4"), "Player2", **kwargs)


class TeamStateTests(unittest.TestCase):
    def test_duel_preserves_legacy_vector_and_masks_team_roles(self):
        players = payload()
        current = TeamState.from_players(players, "Player4", ("Player2",))
        vector = current.to_vector(position_scale=200, damage_scale=50)
        legacy = GameState.from_players(players, "Player4", "Player2").to_vector(200, 50)
        np.testing.assert_array_equal(vector[:10], legacy)
        np.testing.assert_array_equal(vector[10:], np.zeros(16, np.float32))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector.shape, (26,))
        self.assertEqual(FEATURE_COUNT, 26)
        self.assertEqual(len(FEATURE_NAMES), 26)
        self.assertEqual(len(set(FEATURE_NAMES)), 26)
        self.assertEqual(current.format, "1v1")
        self.assertIsNone(current.teammate)

    def test_team_vector_distinguishes_ally_and_ordered_enemies(self):
        current = team(friendly_fire=True)
        vector = current.to_vector(position_scale=100, damage_scale=10)
        np.testing.assert_array_equal(
            vector[:10], GameState.from_players(payload(), "Player1", "Player3").to_vector(100, 10))
        np.testing.assert_allclose(vector[10:17], [2, -1, 2, 0, 1, -0.5, 1])
        np.testing.assert_allclose(vector[17:24], [4, -2, 4, 0, 3, -1.5, 1])
        np.testing.assert_array_equal(vector[24:], [1, 1])
        self.assertEqual(current.format, "2v2")
        self.assertEqual(current.teammate.slot, "Player2")
        self.assertEqual(tuple(player.slot for player in current.opponents), ("Player3", "Player4"))
        reversed_order = TeamState.from_players(payload(), "Player1", ("Player4", "Player3"), "Player2")
        self.assertEqual(reversed_order.opponents[0].slot, "Player4")
        self.assertEqual(reversed_order.to_vector()[4], np.float32(0.4))

    def test_invalid_or_ambiguous_roles_are_rejected(self):
        assignments = [
            ("Player1", (), None),
            ("Player1", ("Player2", "Player3"), None),
            ("Player1", ("Player3",), "Player2"),
            ("Player1", ("Player1",), None),
            ("Player1", ("Player3", "Player3"), "Player2"),
            ("Player1", ("Player3", "Player4"), "Player3"),
            ("Player1", "Player2", None),
            (None, ("Player2",), None),
            ("Player1", ("Player9",), None),
        ]
        for own, opponents, ally in assignments:
            with self.subTest(own=own, opponents=opponents, ally=ally), self.assertRaises(ValueError):
                TeamState.from_players(payload(), own, opponents, ally)
        for value in (1, None, "false"):
            with self.subTest(friendly_fire=value), self.assertRaises(ValueError):
                team(friendly_fire=value)
        with self.assertRaises(ValueError):
            TeamState(PlayerState("Player1", 0, 0, 0, False), ())

    def test_extra_players_do_not_infer_or_change_assignments(self):
        players = payload()
        selected = TeamState.from_players(players, "Player2", ["Player1"])
        self.assertEqual(tuple(player.slot for player in selected.selected_players), ("Player2", "Player1"))
        self.assertIsInstance(selected.opponents, tuple)
        self.assertIsNone(selected.teammate)
        players["Player3"]["x"] = 900
        np.testing.assert_array_equal(
            selected.to_vector(), TeamState.from_players(players, "Player2", ("Player1",)).to_vector())

    def test_team_rewards_sum_both_sides_without_attack_attribution(self):
        tracker = TeamRewardTracker()
        initial = tracker.update(team(payload((0, 0, 0, 0))))
        self.assertEqual(initial.reward, 0)
        self.assertTrue(initial.baseline_reset)
        result = tracker.update(team(payload((10, 20, 40, 30))))
        self.assertEqual(result.damage_dealt, 70)
        self.assertEqual(result.damage_received, 30)
        self.assertAlmostEqual(result.reward, 0.4)
        self.assertFalse(result.discontinuity)
        self.assertFalse(result.terminal_proxy)
        self.assertIn("damage_dealt", asdict(result))
        self.assertEqual(tracker.update(team(payload((10, 20, 40, 30)))).reward, 0)
        penalty = tracker.update(team(payload((30, 50, 40, 30))))
        self.assertAlmostEqual(penalty.reward, -0.5)

    def test_duel_rewards_match_legacy_tracker(self):
        old_tracker = RewardTracker()
        new_tracker = TeamRewardTracker()
        for damages in ((0, 0, 0, 0), (10, 0, 20, 0), (25, 0, 20, 0), (0, 0, 30, 0)):
            players = payload(damages)
            old = old_tracker.update(GameState.from_players(players, "Player1", "Player3"))
            new = new_tracker.update(TeamState.from_players(players, "Player1", ("Player3",)))
            self.assertEqual(asdict(new), asdict(old))

    def test_any_team_member_respawn_rebaselines_entire_transition(self):
        for reset_index in range(4):
            tracker = TeamRewardTracker()
            tracker.update(team(payload((100, 100, 100, 100))))
            damages = [120, 120, 120, 120]
            damages[reset_index] = 0
            reset = tracker.update(team(payload(damages)))
            self.assertEqual((reset.reward, reset.damage_dealt, reset.damage_received), (0, 0, 0))
            self.assertTrue(reset.discontinuity)
            self.assertIn(f"Player{reset_index + 1}_damage_decreased", reset.diagnostics)
            later = [value + 10 for value in damages]
            self.assertEqual(tracker.update(team(payload(later))).reward, 0)

    def test_identity_and_role_changes_cannot_mix_rewards(self):
        tracker = TeamRewardTracker()
        tracker.update(team())
        players = payload((20, 30, 60, 90))
        players["Player2"]["BrawlhallaID"] = 99
        changed = tracker.update(team(players))
        self.assertTrue(changed.discontinuity)
        self.assertEqual(changed.reward, 0)
        self.assertIn("Player2_identity_changed", changed.diagnostics)
        reordered = tracker.update(TeamState.from_players(players, "Player1", ("Player4", "Player3"), "Player2"))
        self.assertEqual(reordered.reward, 0)
        self.assertTrue(any("_slot_changed_to_" in code for code in reordered.diagnostics))
        duel = tracker.update(TeamState.from_players(players, "Player1", ("Player3",)))
        self.assertEqual(duel.reward, 0)
        self.assertIn("format_changed", duel.diagnostics)

    def test_elimination_is_only_a_single_edge_proxy(self):
        tracker = TeamRewardTracker()
        tracker.update(team())
        players = payload()
        players["Player2"]["Eliminated"] = True
        players["Player4"]["Eliminated"] = True
        result = tracker.update(team(players))
        self.assertEqual(result.reward, 0)
        self.assertTrue(result.terminal_proxy)
        self.assertEqual(result.eliminated_slots, ("Player2", "Player4"))
        self.assertFalse(tracker.update(team(players)).terminal_proxy)
        tracker.reset()
        initial = tracker.update(team(players))
        self.assertTrue(initial.baseline_reset)
        self.assertFalse(initial.terminal_proxy)

    def test_context_changes_rebaseline_without_inventing_damage(self):
        tracker = TeamRewardTracker()
        tracker.update(team())
        changed = tracker.update(team(payload((10, 20, 80, 90)), friendly_fire=True))
        self.assertEqual(changed.reward, 0)
        self.assertIn("friendly_fire_changed", changed.diagnostics)
        duel = TeamState.from_players(payload(), "Player1", ("Player2",), friendly_fire=True)
        np.testing.assert_array_equal(duel.to_vector()[24:], [0, 1])

    def test_finite_features_and_reward_scales(self):
        for scale in (0, -1, True, "2", float("nan"), float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    team().to_vector(position_scale=scale)
                with self.assertRaises(ValueError):
                    team().to_vector(damage_scale=scale)
                with self.assertRaises(ValueError):
                    TeamRewardTracker(scale)
        players = payload()
        players["Player2"]["x"] = 1e308
        with self.assertRaises(ValueError):
            team(players).to_vector()
        self.assertTrue(np.isfinite(team(players).to_vector(position_scale=1e308)).all())
        tracker = TeamRewardTracker(1e308)
        tracker.update(team(payload((0, 0, 0, 0))))
        with self.assertRaises(ValueError):
            tracker.update(team(payload((0, 0, 100, 100))))
        self.assertEqual(tracker.update(team(payload((0, 0, 1, 0)))).reward, 1e308)
        tracker = TeamRewardTracker()
        tracker.update(team(payload((0, 0, 0, 0))))
        with self.assertRaises(ValueError):
            tracker.update(team(payload((0, 0, 1e308, 1e308))))


if __name__ == "__main__":
    unittest.main()
