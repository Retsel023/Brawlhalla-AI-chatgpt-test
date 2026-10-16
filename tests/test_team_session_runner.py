"""Exercise team session orchestration with synthetic protocol-3 providers.

These checks use real PPO and sockets; they do not establish Windows game
integration, separate physical input devices, or real matchmaking support.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from model_store import ModelStore
from session_profiles import default_options
from session_runner import run_session
from test_session_bridge import SessionProviderFixture
from test_team_control_bridge import TeamControlProviderFixture


ROOT = Path(__file__).resolve().parents[1]
MODEL_KWARGS = {
    "n_steps": 8, "batch_size": 8, "n_epochs": 1, "verbose": 0, "seed": 47,
    "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}},
}
DRY_RUN_GUARD = """
import builtins
import runpy
import sys
original = builtins.__import__
blocked = {'numpy', 'torch', 'gymnasium', 'stable_baselines3', 'pynput', 'mss', 'tkinter'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('dry-run imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TeamSessionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model_root = self.directory / "models"
        self.output = io.StringIO()

    def options(self, *, name="online_teams", **overrides):
        return default_options(name, **{
            "steps": 16, "checkpoint_interval": 16,
            "model_root": str(self.model_root), "device": "cpu", **overrides,
        })

    def provider(self, options, *, alter=None, episode_steps=2):
        fixture = TeamControlProviderFixture if options.profile["match_format"] == "2v2" else SessionProviderFixture
        provider = fixture(options.profile, alter=alter, episode_steps=episode_steps)
        self.addCleanup(provider.close)
        return provider

    def config(self, provider):
        return {"step_seconds": 0.001, "sessions": {"history_frames": 4},
                "bridge": {"host": "127.0.0.1", "port": provider.port,
                           "timeout_seconds": 0.5, "episode_timeout_seconds": 1,
                           "expected_build": "11.00.36"}}

    def run_options(self, options, *, provider=None, **kwargs):
        provider = self.provider(options) if provider is None else provider
        with contextlib.redirect_stdout(self.output):
            result = run_session(options, self.config(provider),
                                 model_kwargs=MODEL_KWARGS, **kwargs)
        self.assertTrue(provider.released.wait(1), provider.requests)
        return result, provider

    def checkpoint(self, branch="teams"):
        return self.model_root / branch / "latest.zip"

    def metadata(self, branch="teams"):
        return json.loads(self.checkpoint(branch).with_suffix(".json").read_text(encoding="utf-8"))

    def commands(self, provider, command):
        return [item for item in provider.requests if item["command"] == command]

    def test_random_then_ai_teammate_resumes_one_shared_branch_and_counts_world_matches_once(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy

        result, random_provider = self.run_options(self.options(teammate_kind="random"))
        self.assertEqual(result, 0)
        self.assertEqual(len(self.commands(random_provider, "step")), 16)
        self.assertFalse(self.commands(random_provider, "step_joint"))
        before = PPO.load(str(self.checkpoint()), device="cpu")
        self.assertEqual(before.num_timesteps, 16)
        self.assertEqual(before.observation_space.shape, (210,))

        forward_batches = []
        policy_ids = set()
        original_forward = ActorCriticPolicy.forward

        def record_forward(policy, observations, *args, **kwargs):
            if observations.shape == (2, 210):
                policy_ids.add(id(policy))
                forward_batches.append(observations[:, 78:104].detach().cpu().clone())
            return original_forward(policy, observations, *args, **kwargs)

        with patch.object(ActorCriticPolicy, "forward", record_forward):
            result, ai_provider = self.run_options(self.options(teammate_kind="ai"))
        self.assertEqual(result, 0)
        self.assertEqual(len(self.commands(ai_provider, "step_joint")), 8)
        self.assertFalse(self.commands(ai_provider, "step"))
        self.assertEqual(len(forward_batches), 8)
        self.assertEqual(len(policy_ids), 1, "both perspectives use the same policy parameters")
        self.assertFalse(torch.equal(forward_batches[0][0], forward_batches[0][1]))
        for request in self.commands(ai_provider, "step_joint"):
            self.assertEqual(set(request["actions"]), {"Player1", "Player2"})
        resumed = PPO.load(str(self.checkpoint()), device="cpu")
        self.assertEqual(resumed.num_timesteps, 32)
        self.assertEqual(resumed.observation_space.shape, (210,))
        self.assertEqual(resumed.n_envs, 2)
        self.assertTrue(any(not torch.equal(value, resumed.policy.state_dict()[name])
                            for name, value in before.policy.state_dict().items()))
        progress = self.metadata()["last_session"]
        self.assertEqual(progress["session_steps"], 16)
        self.assertEqual(progress["episodes"], 4, "a completed shared match counts once")
        self.assertEqual(self.metadata()["strategy"], "resumed")

    def test_duel_weights_initialize_dual_team_policy_without_changing_duel_files(self):
        from stable_baselines3 import PPO

        result, _ = self.run_options(self.options(name="bot_training", steps=8))
        self.assertEqual(result, 0)
        duel_path = self.checkpoint("duel")
        duel = PPO.load(str(duel_path), device="cpu")
        duel_hash = file_hash(duel_path)
        duel_metadata = duel_path.with_suffix(".json").read_bytes()
        captured = []
        original_load = ModelStore.load_or_create

        def record_initialization(store, *args, **kwargs):
            model, lineage = original_load(store, *args, **kwargs)
            captured.append((lineage, {name: value.detach().clone()
                                      for name, value in model.policy.state_dict().items()},
                             bool(model.policy.optimizer.state)))
            return model, lineage

        with patch.object(ModelStore, "load_or_create", record_initialization):
            result, provider = self.run_options(self.options(teammate_kind="ai"))
        self.assertEqual(result, 0)
        self.assertEqual(len(captured), 1)
        lineage, initial_weights, optimizer_has_state = captured[0]
        self.assertEqual(lineage["strategy"], "warm_started")
        self.assertEqual(lineage["source_sha256"], duel_hash)
        self.assertFalse(optimizer_has_state)
        for name, value in duel.policy.state_dict().items():
            self.assertTrue(torch.equal(value, initial_weights[name]), name)
        self.assertEqual(file_hash(duel_path), duel_hash)
        self.assertEqual(duel_path.with_suffix(".json").read_bytes(), duel_metadata)
        self.assertEqual(len(self.commands(provider, "step_joint")), 8)

    def test_ai_evaluation_counts_actor_decisions_and_never_saves(self):
        from stable_baselines3 import PPO

        self.run_options(self.options(teammate_kind="ai"))
        checkpoint = self.checkpoint()
        initial_hash = file_hash(checkpoint)
        initial_metadata = checkpoint.with_suffix(".json").read_bytes()
        initial_steps = PPO.load(str(checkpoint), device="cpu").num_timesteps
        with patch.object(ModelStore, "save", side_effect=AssertionError("evaluation saved weights")):
            result, provider = self.run_options(
                self.options(teammate_kind="ai", learning=False, steps=6))
        self.assertEqual(result, 0)
        self.assertEqual(len(self.commands(provider, "step_joint")), 3)
        self.assertFalse(self.commands(provider, "step"))
        self.assertEqual(file_hash(checkpoint), initial_hash)
        self.assertEqual(checkpoint.with_suffix(".json").read_bytes(), initial_metadata)
        self.assertEqual(PPO.load(str(checkpoint), device="cpu").num_timesteps, initial_steps)

    def test_check_only_refuses_missing_joint_capability_or_routes_before_actions(self):
        for missing in ("joint_actions", "independent_inputs", "input_routes"):
            with self.subTest(missing=missing):
                options = self.options(teammate_kind="ai", device="cuda")

                def alter(request, response):
                    if request["command"] == "hello":
                        if missing == "input_routes":
                            response.pop("input_routes")
                        else:
                            response["capabilities"].remove(missing)
                    return response

                provider = self.provider(options, alter=alter)
                with contextlib.redirect_stdout(self.output), \
                        patch("session_runner.choose_device", side_effect=AssertionError("GPU initialized")), \
                        self.assertRaises((ValueError, RuntimeError)):
                    run_session(options, self.config(provider), check_only=True)
                self.assertTrue(provider.released.wait(1), provider.requests)
                self.assertEqual([item["command"] for item in provider.requests], ["hello", "release"])
        self.assertFalse(self.model_root.exists())

    def test_unlimited_dual_session_stops_after_joint_commands_and_saves_releases(self):
        from stable_baselines3 import PPO

        stop_file = self.directory / "stop.request"
        actions = 0

        def stop_after_eight(request, response):
            nonlocal actions
            if request["command"] == "step_joint":
                actions += 1
                if actions >= 8:
                    stop_file.touch()
            return response

        options = self.options(teammate_kind="ai", steps=0, max_seconds=5)
        provider = self.provider(options, alter=stop_after_eight)
        started = time.monotonic()
        result, _ = self.run_options(options, provider=provider, stop_file=stop_file)
        self.assertEqual(result, 0)
        self.assertTrue(stop_file.exists())
        self.assertEqual(actions, 8)
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(self.checkpoint().is_file())
        model = PPO.load(str(self.checkpoint()), device="cpu")
        self.assertGreater(model.num_timesteps, 0)
        self.assertEqual(self.metadata()["last_session"]["model_timesteps"], model.num_timesteps)
        self.assertIn("Session stopped: stop_requested", self.output.getvalue())


class TeamSessionCliTests(unittest.TestCase):
    def test_dry_run_exposes_teammate_arrangements_and_account_identity_without_dependencies(self):
        cases = [
            (["--teammate-kind", "random"], "random", None),
            (["--teammate-kind", "ai"], "ai", None),
            (["--teammate-kind", "regular", "--teammate-account-id", "777"], "regular", 777),
        ]
        for arguments, kind, account_id in cases:
            with self.subTest(kind=kind):
                result = subprocess.run(
                    [sys.executable, "-c", DRY_RUN_GUARD, str(ROOT / "agent.py"),
                     "session", "--profile", "online_teams", "--dry-run", *arguments],
                    cwd=ROOT, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                settings = json.loads(result.stdout)
                self.assertEqual(settings["profile"]["teammate_kind"], kind)
                self.assertEqual(settings["profile"]["teammate_account_id"], account_id)
                self.assertEqual(settings["profile"]["match_format"], "2v2")

    def test_saved_teammate_settings_cannot_be_mixed_with_cli_identity_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "options.json"
            saved.write_text(json.dumps(default_options(
                "online_teams", teammate_kind="regular", teammate_account_id=777).to_dict()),
                encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-c", DRY_RUN_GUARD, str(ROOT / "agent.py"),
                 "session", "--session-file", str(saved), "--dry-run",
                 "--teammate-kind", "ai"],
                cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("not both", result.stderr)


if __name__ == "__main__":
    unittest.main()
