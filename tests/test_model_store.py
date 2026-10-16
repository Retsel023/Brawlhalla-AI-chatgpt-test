"""Exercise checkpoint branching and migration with real, tiny PPO policies."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch

from controls import ACTIONS
from model_store import DEFAULT_FEATURE_NAMES, ModelStore


CONTRACT = {
    "feature_names": list(DEFAULT_FEATURE_NAMES),
    "position_scale": 1000.0,
}
MODEL_KWARGS = {
    "n_steps": 8,
    "batch_size": 8,
    "n_epochs": 1,
    "seed": 17,
    "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}},
}


class NumericFixture(gym.Env):
    """Small deterministic task; no desktop or game process is involved."""

    def __init__(self, features=26):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(features,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.count = 0

    def observation(self):
        values = np.zeros(self.observation_space.shape, dtype=np.float32)
        values[0] = self.count / 8
        values[1] = 1.0
        return values

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return self.observation(), {}

    def step(self, action):
        self.count += 1
        return (self.observation(), float(action == 1), False,
                self.count >= 8, {})


class VisualFixture(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
    action_space = gym.spaces.Discrete(len(ACTIONS))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        return (*self.reset()[:1], 0.0, False, True, {})


class ModelStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "sessions"
        self.store = ModelStore(self.root, model_kwargs=MODEL_KWARGS)

    def env(self, features=26):
        env = NumericFixture(features)
        self.addCleanup(env.close)
        return env

    def load(self, match_format="duel", *, features=26, **kwargs):
        return self.store.load_or_create(
            self.env(features), match_format, device="cpu", contract=CONTRACT,
            **kwargs)

    def save(self, model, lineage, match_format="duel"):
        return self.store.save(
            model, match_format, lineage, contract=CONTRACT)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_training_resume_preserves_weights_optimizer_and_steps(self):
        self.assertEqual(self.store.path_for("duel"),
                         self.root / "duel" / "latest.zip")
        self.assertEqual(self.store.path_for("teams"),
                         self.root / "teams" / "latest.zip")
        model, lineage = self.load()
        before = {name: value.clone()
                  for name, value in model.policy.state_dict().items()}
        model.learn(8)
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in model.policy.state_dict().items()))
        self.assertTrue(model.policy.optimizer.state)
        self.save(model, lineage)

        resumed, resumed_lineage = self.load()
        self.assertEqual(resumed.num_timesteps, 8)
        self.assert_tree_equal(model.policy.state_dict(),
                               resumed.policy.state_dict())
        self.assert_tree_equal(model.policy.optimizer.state_dict(),
                               resumed.policy.optimizer.state_dict())
        self.assertIsInstance(resumed_lineage, dict)
        resumed.learn(8, reset_num_timesteps=False)
        self.assertEqual(resumed.num_timesteps, 16)
        self.assertTrue(any(not torch.equal(value, resumed.policy.state_dict()[name])
                            for name, value in model.policy.state_dict().items()))

    def test_teams_inherits_duel_weights_then_resumes_its_own_branch(self):
        duel, duel_lineage = self.load()
        duel.learn(8)
        duel_path = self.save(duel, duel_lineage)
        teams, teams_lineage = self.load("teams")
        self.assert_tree_equal(duel.policy.state_dict(), teams.policy.state_dict())
        self.assertEqual(teams.num_timesteps, 0)
        self.assertFalse(teams.policy.optimizer.state)
        self.assertIsInstance(teams_lineage, dict)
        self.assertTrue(teams_lineage)
        teams.learn(8)
        teams_path = self.save(teams, teams_lineage, "teams")
        self.assertNotEqual(duel_path, teams_path)

        # A later duel checkpoint must not overwrite or seed an existing team run.
        duel.learn(8, reset_num_timesteps=False)
        self.save(duel, duel_lineage)
        resumed, _ = self.load("teams")
        self.assert_tree_equal(teams.policy.state_dict(), resumed.policy.state_dict())
        self.assert_tree_equal(teams.policy.optimizer.state_dict(),
                               resumed.policy.optimizer.state_dict())
        self.assertEqual(resumed.num_timesteps, 8)

    def test_legacy_numeric_migration_preserves_behavior_and_zeros_new_inputs(self):
        legacy = PPO("MlpPolicy", self.env(10), device="cpu", **MODEL_KWARGS)
        legacy.learn(8)
        source = Path(self.temporary.name) / "legacy-state.zip"
        legacy.save(source)
        self.assertFalse(source.with_suffix(".json").exists())
        migrated, lineage = self.load(initialize_from=source)
        self.assertEqual(migrated.observation_space.shape, (26,))
        self.assertEqual(migrated.num_timesteps, 0)
        self.assertFalse(migrated.policy.optimizer.state)
        self.assertIsInstance(lineage, dict)
        old_state, new_state = legacy.policy.state_dict(), migrated.policy.state_dict()
        expanded = []
        for name, old_value in old_state.items():
            new_value = new_state[name]
            if old_value.ndim == 2 and old_value.shape[1] == 10:
                expanded.append(name)
                self.assertTrue(torch.equal(new_value[:, :10], old_value))
                self.assertEqual(torch.count_nonzero(new_value[:, 10:]).item(), 0)
            else:
                self.assertTrue(torch.equal(new_value, old_value), name)
        self.assertEqual(len(expanded), 2, "both actor and value input layers expand")

        old_observations = torch.tensor(
            np.random.default_rng(4).normal(size=(5, 10)).astype(np.float32))
        new_observations = torch.cat(
            (old_observations, torch.zeros((5, 16))), dim=1)
        with torch.no_grad():
            old_distribution = legacy.policy.get_distribution(old_observations)
            new_distribution = migrated.policy.get_distribution(new_observations)
            torch.testing.assert_close(
                old_distribution.distribution.probs,
                new_distribution.distribution.probs, rtol=0, atol=1e-7)
            torch.testing.assert_close(
                legacy.policy.predict_values(old_observations),
                migrated.policy.predict_values(new_observations),
                rtol=0, atol=1e-7)
        migrated.learn(8)
        self.assertTrue(migrated.policy.optimizer.state)

    def test_cnn_checkpoint_cannot_initialize_numeric_session(self):
        image_env = VisualFixture()
        self.addCleanup(image_env.close)
        cnn = PPO("CnnPolicy", image_env, n_steps=8, batch_size=8,
                  n_epochs=1, device="cpu")
        source = Path(self.temporary.name) / "screen-policy.zip"
        cnn.save(source)
        with self.assertRaises(ValueError):
            self.load(initialize_from=source)
        self.assertFalse(self.store.path_for("duel").exists())

    def test_history_migrations_place_legacy_weights_only_in_latest_frame(self):
        for source_features in (10, 26):
            with self.subTest(source_features=source_features):
                legacy = PPO("MlpPolicy", self.env(source_features),
                             device="cpu", **MODEL_KWARGS)
                legacy.learn(8)
                source = Path(self.temporary.name) / f"legacy-{source_features}.zip"
                legacy.save(source)
                migrated, _ = self.load(features=104, initialize_from=source)
                self.assertEqual(migrated.observation_space.shape, (104,))
                self.assertFalse(migrated.policy.optimizer.state)
                self.assertEqual(migrated.num_timesteps, 0)
                old_state = legacy.policy.state_dict()
                new_state = migrated.policy.state_dict()
                start = 104 - 26
                expanded = []
                for name, old_value in old_state.items():
                    new_value = new_state[name]
                    if old_value.ndim == 2 and old_value.shape[1] == source_features:
                        expanded.append(name)
                        self.assertTrue(torch.equal(
                            new_value[:, start:start + source_features], old_value))
                        self.assertEqual(torch.count_nonzero(new_value[:, :start]).item(), 0)
                        self.assertEqual(torch.count_nonzero(
                            new_value[:, start + source_features:]).item(), 0)
                    else:
                        self.assertTrue(torch.equal(new_value, old_value), name)
                self.assertEqual(len(expanded), 2)

                old_observations = torch.tensor(np.random.default_rng(8).normal(
                    size=(5, source_features)).astype(np.float32))
                # Earlier history is arbitrary; its zero weights cannot affect output.
                new_observations = torch.tensor(np.random.default_rng(9).normal(
                    size=(5, 104)).astype(np.float32))
                new_observations[:, start:start + source_features] = old_observations
                with torch.no_grad():
                    torch.testing.assert_close(
                        legacy.policy.get_distribution(old_observations).distribution.probs,
                        migrated.policy.get_distribution(new_observations).distribution.probs,
                        rtol=0, atol=1e-7)
                    torch.testing.assert_close(
                        legacy.policy.predict_values(old_observations),
                        migrated.policy.predict_values(new_observations),
                        rtol=0, atol=1e-7)

    def test_stacked_history_branches_transfer_and_reject_history_mismatch(self):
        duel, lineage = self.load(features=104)
        duel.learn(8)
        source = self.save(duel, lineage)
        teams, _ = self.load("teams", features=104)
        self.assert_tree_equal(duel.policy.state_dict(), teams.policy.state_dict())
        self.assertFalse(teams.policy.optimizer.state)
        self.assertEqual(teams.num_timesteps, 0)
        with self.assertRaises(ValueError):
            self.store.load_or_create(
                self.env(104), "teams", initialize_from=source, device="cpu",
                contract=dict(CONTRACT, history_frames=2))
        with self.assertRaises(ValueError):
            self.store.load_or_create(
                self.env(26), "duel", device="cpu", contract=CONTRACT)

    def test_metadata_bearing_single_frame_source_can_explicitly_expand_history(self):
        duel, lineage = self.load()
        duel.learn(8)
        source = self.save(duel, lineage)
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            self.load("teams", features=104)
        teams, teams_lineage = self.load(
            "teams", features=104, initialize_from=source)
        self.assertEqual(teams_lineage["strategy"], "migrated_legacy")
        self.assertEqual(teams_lineage["contract"]["history_frames"], 4)
        self.assertEqual(teams.num_timesteps, 0)
        self.assertFalse(teams.policy.optimizer.state)
        for name, tensor in duel.policy.state_dict().items():
            transferred = teams.policy.state_dict()[name]
            if tensor.ndim == 2 and tensor.shape[1] == 26:
                self.assertTrue(torch.equal(transferred[:, 78:], tensor))
                self.assertEqual(torch.count_nonzero(transferred[:, :78]).item(), 0)
            else:
                self.assertTrue(torch.equal(transferred, tensor), name)
        destination = self.save(teams, teams_lineage, "teams")
        self.assertEqual(source.read_bytes(), original)
        metadata = json.loads(destination.with_suffix(".json").read_text())
        self.assertEqual(metadata["contract"]["history_frames"], 4)

    def test_incompatible_checkpoint_action_space_is_rejected(self):
        source_env = self.env(10)
        source_env.action_space = gym.spaces.Discrete(len(ACTIONS) - 1)
        incompatible = PPO("MlpPolicy", source_env, device="cpu", **MODEL_KWARGS)
        source = Path(self.temporary.name) / "wrong-actions.zip"
        incompatible.save(source)
        with self.assertRaises(ValueError):
            self.load(initialize_from=source)
        self.assertFalse(self.store.path_for("duel").exists())

    def test_missing_branch_metadata_rejects_resume_without_replacing_weights(self):
        model, lineage = self.load()
        destination = self.save(model, lineage)
        original = destination.read_bytes()
        destination.with_suffix(".json").unlink()
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(destination.read_bytes(), original)

    def test_observation_contract_change_rejects_resume_and_transfer(self):
        model, lineage = self.load()
        source = self.save(model, lineage)
        for changed in (
                dict(CONTRACT, position_scale=500.0),
                dict(CONTRACT, feature_names=list(reversed(CONTRACT["feature_names"]))),
        ):
            with self.subTest(contract=changed):
                with self.assertRaises(ValueError):
                    self.store.load_or_create(
                        self.env(), "duel", device="cpu", contract=changed)
                with self.assertRaises(ValueError):
                    self.store.load_or_create(
                        self.env(), "teams", device="cpu", contract=changed,
                        initialize_from=source)

    def test_explicit_initializer_cannot_replace_existing_or_target_itself(self):
        model, lineage = self.load()
        source = self.save(model, lineage)
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            self.load(initialize_from=source)
        self.assertEqual(source.read_bytes(), original)
        with self.assertRaises(ValueError):
            self.load("teams", initialize_from=self.store.path_for("teams"))

    def test_metadata_describes_saved_weights_and_detects_checkpoint_tampering(self):
        model, lineage = self.load()
        destination = self.save(model, lineage)
        metadata_path = destination.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertIsInstance(metadata, dict)
        for name, value in CONTRACT.items():
            self.assertEqual(metadata["contract"][name], value)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        self.assertEqual(metadata["checkpoint_sha256"], digest)
        self.assertEqual(metadata["lineage"], lineage)
        for name, bad_value in (("action_names_sha256", "0" * 64),
                                ("observation_schema", "unknown-observation-v99")):
            with self.subTest(metadata_field=name):
                changed = dict(metadata, **{name: bad_value})
                metadata_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        destination.write_bytes(destination.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            self.load()

    def test_failed_save_preserves_previous_checkpoint_and_metadata(self):
        model, lineage = self.load()
        destination = self.save(model, lineage)
        metadata = destination.with_suffix(".json")
        old_checkpoint, old_metadata = destination.read_bytes(), metadata.read_bytes()

        def interrupted_save(path, *args, **kwargs):
            Path(path).write_bytes(b"incomplete checkpoint")
            raise OSError("simulated disk write failure")

        with patch.object(model, "save", side_effect=interrupted_save):
            with self.assertRaisesRegex(OSError, "simulated disk write failure"):
                self.save(model, lineage)
        self.assertEqual(destination.read_bytes(), old_checkpoint)
        self.assertEqual(metadata.read_bytes(), old_metadata)
        resumed, _ = self.load()
        self.assert_tree_equal(model.policy.state_dict(), resumed.policy.state_dict())


if __name__ == "__main__":
    unittest.main()
