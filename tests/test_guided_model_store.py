"""Check goal suffix migration using actual PPO weights, logits, and Adam state."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch

from controls import ACTIONS
from guidance_settings import GUIDANCE_FEATURE_NAMES, GUIDANCE_SCHEMA
from model_store import ModelStore


BASE_SIZE = 26 * 4
GUIDED_SIZE = BASE_SIZE + len(GUIDANCE_FEATURE_NAMES)
GUIDED_CONTRACT = {
    "history_frames": 4,
    "guidance_schema": GUIDANCE_SCHEMA,
    "guidance_features": list(GUIDANCE_FEATURE_NAMES),
}
SMALL_PPO = {
    "n_steps": 8, "batch_size": 8, "n_epochs": 1, "seed": 34,
    "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}},
}
FIRST_LAYERS = ("mlp_extractor.policy_net.0.weight",
                "mlp_extractor.value_net.0.weight")


class StructuredFixture(gym.Env):
    def __init__(self, size):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.steps = 0

    def observation(self):
        observation = np.full(self.observation_space.shape, 0.25, np.float32)
        observation[0] = self.steps / 8
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        return self.observation(), float(action == 1), False, self.steps >= 8, {}


class GuidedModelStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ModelStore(self.root / "sessions", model_kwargs=SMALL_PPO)

    def env(self, size):
        env = StructuredFixture(size)
        self.addCleanup(env.close)
        return env

    def load_guided(self, match_format="duel", **kwargs):
        return self.store.load_or_create(
            self.env(GUIDED_SIZE), match_format, contract=GUIDED_CONTRACT, **kwargs)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def assert_equivalent(self, old, new, old_observations, new_observations):
        with torch.no_grad():
            torch.testing.assert_close(
                old.policy.get_distribution(old_observations).distribution.logits,
                new.policy.get_distribution(new_observations).distribution.logits,
                rtol=0, atol=1e-6)
            torch.testing.assert_close(old.policy.predict_values(old_observations),
                                       new.policy.predict_values(new_observations),
                                       rtol=0, atol=1e-6)

    def test_guided_training_save_and_resume_preserves_optimizer(self):
        model, lineage = self.load_guided()
        self.assertEqual(model.observation_space.shape, (GUIDED_SIZE,))
        self.assertEqual(lineage["contract"]["guidance_features"], list(GUIDANCE_FEATURE_NAMES))
        before = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
        model.learn(8)
        self.assertTrue(any(not torch.equal(value, model.policy.state_dict()[name])
                            for name, value in before.items()))
        self.assertTrue(model.policy.optimizer.state)
        saved = self.store.save(model, "duel", lineage)
        metadata = json.loads(saved.with_suffix(".json").read_text())
        self.assertEqual(metadata["contract"]["history_frames"], 4)
        self.assertEqual(metadata["contract"]["guidance_schema"], GUIDANCE_SCHEMA)
        resumed, resumed_lineage = self.load_guided()
        self.assertEqual(resumed_lineage["strategy"], "resumed")
        self.assertEqual(resumed.num_timesteps, 8)
        self.assert_tree_equal(model.policy.state_dict(), resumed.policy.state_dict())
        self.assert_tree_equal(model.policy.optimizer.state_dict(), resumed.policy.optimizer.state_dict())
        resumed.learn(8, reset_num_timesteps=False)
        self.assertEqual(resumed.num_timesteps, 16)

    def test_existing_history_branch_migrates_without_changing_saved_files(self):
        old, old_lineage = self.store.load_or_create(self.env(BASE_SIZE), "duel")
        old.learn(8)
        saved = self.store.save(old, "duel", old_lineage)
        old_checkpoint, old_metadata = saved.read_bytes(), saved.with_suffix(".json").read_bytes()

        migrated, lineage = self.load_guided()
        self.assertEqual(lineage["strategy"], "migrated_guidance")
        self.assertEqual(migrated.num_timesteps, old.num_timesteps)
        self.assertFalse(migrated.policy.optimizer.state)
        self.assertEqual(saved.read_bytes(), old_checkpoint)
        self.assertEqual(saved.with_suffix(".json").read_bytes(), old_metadata)
        for name in FIRST_LAYERS:
            new_weights = migrated.policy.state_dict()[name]
            self.assertTrue(torch.equal(new_weights[:, :BASE_SIZE], old.policy.state_dict()[name]))
            self.assertEqual(torch.count_nonzero(new_weights[:, BASE_SIZE:]).item(), 0)
        old_observations = torch.tensor(np.random.default_rng(3).normal(size=(7, BASE_SIZE)),
                                        dtype=torch.float32)
        new_observations = torch.cat((old_observations,
                                      torch.zeros((7, len(GUIDANCE_FEATURE_NAMES)))), dim=1)
        self.assert_equivalent(old, migrated, old_observations, new_observations)

        migrated.learn(8, reset_num_timesteps=False)
        self.assertEqual(migrated.num_timesteps, 16)
        self.assertTrue(migrated.policy.optimizer.state)
        self.store.save(migrated, "duel", lineage)
        resumed, resumed_lineage = self.load_guided()
        self.assertEqual(resumed_lineage["strategy"], "resumed")
        self.assertEqual(resumed_lineage["initialization_strategy"], "migrated_guidance")
        self.assertEqual(resumed.num_timesteps, 16)
        self.assert_tree_equal(migrated.policy.state_dict(), resumed.policy.state_dict())
        self.assert_tree_equal(migrated.policy.optimizer.state_dict(), resumed.policy.optimizer.state_dict())

    def test_new_team_branch_can_inherit_guided_or_nonguided_duel(self):
        for guided in (False, True):
            with self.subTest(guided=guided):
                store = ModelStore(self.root / str(guided), model_kwargs=SMALL_PPO)
                size = GUIDED_SIZE if guided else BASE_SIZE
                contract = GUIDED_CONTRACT if guided else None
                duel, origin = store.load_or_create(self.env(size), "duel", contract=contract)
                duel.learn(8)
                source = store.save(duel, "duel", origin)
                source_bytes = source.read_bytes()
                teams, lineage = store.load_or_create(self.env(GUIDED_SIZE), "teams",
                                                      contract=GUIDED_CONTRACT)
                self.assertEqual(lineage["strategy"], "warm_started" if guided else "migrated_guidance")
                self.assertEqual(teams.num_timesteps, 0)
                self.assertFalse(teams.policy.optimizer.state)
                observations = torch.tensor(np.random.default_rng(6).normal(size=(5, size)),
                                            dtype=torch.float32)
                team_observations = observations if guided else torch.cat(
                    (observations, torch.zeros((5, len(GUIDANCE_FEATURE_NAMES)))), dim=1)
                self.assert_equivalent(duel, teams, observations, team_observations)
                store.save(teams, "teams", lineage)
                self.assertEqual(source.read_bytes(), source_bytes)

    def test_explicit_legacy_weights_target_latest_state_frame_before_goals(self):
        for size in (10, 26):
            with self.subTest(legacy_size=size):
                old = PPO("MlpPolicy", self.env(size), device="cpu", **SMALL_PPO)
                old.learn(8)
                source = self.root / f"legacy-{size}.zip"
                old.save(source)
                migrated, lineage = self.load_guided(initialize_from=source)
                self.assertEqual(lineage["strategy"], "migrated_legacy")
                self.assertEqual(migrated.num_timesteps, 0)
                self.assertFalse(migrated.policy.optimizer.state)
                start = BASE_SIZE - 26
                for name in FIRST_LAYERS:
                    weights = migrated.policy.state_dict()[name]
                    self.assertTrue(torch.equal(weights[:, start:start + size], old.policy.state_dict()[name]))
                    self.assertEqual(torch.count_nonzero(weights[:, :start]).item(), 0)
                    self.assertEqual(torch.count_nonzero(weights[:, start + size:]).item(), 0)
                old_observations = torch.tensor(np.random.default_rng(8).normal(size=(5, size)),
                                                dtype=torch.float32)
                # Arbitrary older frames and goals are initially ignored, not just zero inputs.
                new_observations = torch.tensor(np.random.default_rng(9).normal(size=(5, GUIDED_SIZE)),
                                                dtype=torch.float32)
                new_observations[:, start:start + size] = old_observations
                self.assert_equivalent(old, migrated, old_observations, new_observations)

    def test_metadata_bearing_single_frame_can_explicitly_expand_into_guided_history(self):
        old, origin = self.store.load_or_create(self.env(26), "duel")
        source = self.store.save(old, "duel", origin)
        with self.assertRaises(ValueError):
            self.load_guided("teams")
        migrated, lineage = self.load_guided("teams", initialize_from=source)
        self.assertEqual(lineage["strategy"], "migrated_legacy")
        for name in FIRST_LAYERS:
            weights = migrated.policy.state_dict()[name]
            self.assertTrue(torch.equal(weights[:, 78:104], old.policy.state_dict()[name]))
            self.assertEqual(torch.count_nonzero(weights[:, :78]).item(), 0)
            self.assertEqual(torch.count_nonzero(weights[:, 104:]).item(), 0)

    def test_guidance_contract_and_shape_must_be_explicit_and_exact(self):
        malformed = [
            {},
            dict(GUIDED_CONTRACT, guidance_schema="unknown-v9"),
            dict(GUIDED_CONTRACT, guidance_features=list(reversed(GUIDANCE_FEATURE_NAMES))),
            dict(GUIDED_CONTRACT, guidance_features=list(GUIDANCE_FEATURE_NAMES[:-1])),
            dict(GUIDED_CONTRACT, history_frames=3),
            {"guidance_schema": GUIDANCE_SCHEMA},
            {"guidance_features": list(GUIDANCE_FEATURE_NAMES)},
        ]
        for contract in malformed:
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                self.store.load_or_create(self.env(GUIDED_SIZE), "duel", contract=contract)
        with self.assertRaises(ValueError):
            self.store.load_or_create(self.env(BASE_SIZE), "duel", contract=GUIDED_CONTRACT)

    def test_guidance_cannot_be_removed_or_change_other_observation_semantics(self):
        model, origin = self.load_guided()
        source = self.store.save(model, "duel", origin)
        checkpoint, metadata = source.read_bytes(), source.with_suffix(".json").read_bytes()
        with self.assertRaises(ValueError):
            self.store.load_or_create(self.env(BASE_SIZE), "duel")
        changed = [dict(GUIDED_CONTRACT, position_scale=500),
                   dict(GUIDED_CONTRACT, damage_scale=10),
                   dict(GUIDED_CONTRACT, history_order="newest_to_oldest")]
        names = deepcopy(origin["contract"]["feature_names"])
        changed.append(dict(GUIDED_CONTRACT, feature_names=list(reversed(names))))
        actions = deepcopy(origin["contract"]["action_names"])
        changed.append(dict(GUIDED_CONTRACT, action_names=list(reversed(actions))))
        for contract in changed:
            with self.subTest(contract=contract):
                with self.assertRaises(ValueError):
                    self.store.load_or_create(self.env(GUIDED_SIZE), "duel", contract=contract)
                with self.assertRaises(ValueError):
                    self.store.load_or_create(self.env(GUIDED_SIZE), "teams", contract=contract)
        self.assertEqual(source.read_bytes(), checkpoint)
        self.assertEqual(source.with_suffix(".json").read_bytes(), metadata)

    def test_guided_checkpoint_without_contract_metadata_cannot_claim_guidance_order(self):
        raw = PPO("MlpPolicy", self.env(GUIDED_SIZE), device="cpu", **SMALL_PPO)
        source = self.root / "unlabeled-guided.zip"
        raw.save(source)
        with self.assertRaises(ValueError):
            self.load_guided(initialize_from=source)


if __name__ == "__main__":
    unittest.main()
