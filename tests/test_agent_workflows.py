"""Check user-facing workflows without a display or a running game."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from agent import checkpoint_path
from controls import ACTIONS, Feedback
from desktop import GameEnv, KeyboardActions


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent.py"
BINDINGS = {
    "left": "a", "right": "d", "jump": "space", "down": "s",
    "light": "j", "heavy": "k", "dodge": "l",
}

# Running the real CLI with an import guard verifies that the read-only source
# never constructs controls or initializes the policy/GPU stack.
READ_ONLY_RUNNER = """
import builtins
import runpy
import sys
original_import = builtins.__import__
blocked = {'desktop', 'mss', 'pynput', 'torch', 'stable_baselines3', 'gymnasium'}
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('read-only workflow imported ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


class FakeController:
    def __init__(self):
        self.events = []
        self.held = set()

    def press(self, key):
        self.events.append(("press", key))
        self.held.add(key)

    def release(self, key):
        self.events.append(("release", key))
        self.held.discard(key)


class FakeCamera:
    def __init__(self):
        self.closed = False
        self.grabs = 0
        self.error = None

    def grab(self, region):
        self.grabs += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(size=(4, 4), rgb=bytes((12, 24, 36)) * 16)

    def close(self):
        self.closed = True


class DesktopWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.controls = KeyboardActions(
            BINDINGS, controller=self.controller,
            key_type=SimpleNamespace(space="SPACE"))
        self.camera = FakeCamera()
        self.feedback = Feedback()
        self.config = {
            "keys": BINDINGS,
            "capture": {"left": 0, "top": 0, "width": 4, "height": 4},
            "step_seconds": 0.001,
            "episode_seconds": 60,
        }

    def make_env(self):
        env = GameEnv(self.config, self.feedback,
                      camera=self.camera, controls=self.controls)
        self.addCleanup(env.close)
        return env

    def test_movement_is_held_while_attack_buttons_pulse(self):
        attack_left = ACTIONS.index(("left", "light"))
        self.controls.apply(attack_left)
        self.assertEqual(self.controller.held, {"a", "j"})
        self.controls.release_buttons()
        self.assertEqual(self.controller.held, {"a"})
        self.controls.apply(attack_left)
        self.controls.release_buttons()
        self.assertEqual(self.controller.events.count(("press", "a")), 1)
        self.assertEqual(self.controller.events.count(("press", "j")), 2)

        self.controls.apply(ACTIONS.index(("right", "heavy")))
        self.assertEqual(self.controller.held, {"d", "k"})
        self.assertIn(("release", "a"), self.controller.events)
        self.controls.release()
        self.assertFalse(self.controls.held)
        self.assertFalse(self.controller.held)

    def test_step_produces_visual_observation_and_consumes_feedback(self):
        env = self.make_env()
        initial, _ = env.reset()
        self.assertEqual(initial.shape, (84, 84, 1))
        self.assertEqual(initial.dtype, np.uint8)
        self.feedback.add(0.75)
        observation, reward, terminated, truncated, _ = env.step(
            ACTIONS.index(("left", "light")))
        self.assertEqual(observation.shape, initial.shape)
        self.assertEqual(reward, 0.75)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(self.controller.held, {"a"})
        self.assertEqual(self.feedback.consume(), (0.0, False))

    def test_capture_failure_releases_movement_and_attack_keys(self):
        env = self.make_env()
        env.reset()
        self.camera.error = RuntimeError("capture failed")
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            env.step(ACTIONS.index(("left", "light")))
        self.assertFalse(self.controls.held)
        self.assertFalse(self.controller.held)
        self.assertIn(("release", "a"), self.controller.events)
        self.assertIn(("release", "j"), self.controller.events)

    def test_stop_before_step_releases_previously_held_keys(self):
        env = self.make_env()
        self.controls.apply(ACTIONS.index(("left", "light")))
        self.feedback.stop = True
        with self.assertRaises(KeyboardInterrupt):
            env.step(0)
        self.assertFalse(self.controls.held)
        self.assertFalse(self.controller.held)
        self.assertEqual(self.camera.grabs, 0)

    def test_stop_during_action_wait_releases_keys_without_capturing(self):
        self.config["step_seconds"] = 1.0
        env = self.make_env()

        def request_stop(_seconds):
            self.feedback.stop = True

        with patch("desktop.time.sleep", side_effect=request_stop), \
                self.assertRaises(KeyboardInterrupt):
            env.step(ACTIONS.index(("right", "heavy")))
        self.assertFalse(self.controls.held)
        self.assertFalse(self.controller.held)
        self.assertEqual(self.camera.grabs, 0)

    def test_controls_constructor_failure_closes_camera(self):
        with patch("desktop.KeyboardActions", side_effect=RuntimeError("no keyboard")), \
                self.assertRaisesRegex(RuntimeError, "no keyboard"):
            GameEnv(self.config, self.feedback, camera=self.camera)
        self.assertTrue(self.camera.closed)


class AgentCliTests(unittest.TestCase):
    def run_read_only_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-c", READ_ONLY_RUNNER, str(AGENT), *arguments],
            cwd=ROOT, text=True, capture_output=True, timeout=15)

    def test_spectator_training_and_play_are_rejected_before_dependencies(self):
        system_python = Path("/usr/bin/python3")
        python = str(system_python) if system_python.exists() else sys.executable
        for mode in ("train", "play"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [python, "-S", str(AGENT), mode, "--source", "telemetry",
                     "--config", "/a/config/that/does/not/exist.json"],
                    cwd=ROOT, text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("spectator-only", result.stderr)
                self.assertIn("cannot acknowledge actions or resets", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_observe_cli_reads_actual_export_and_emits_ten_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"state": {
                "position_scale": 100, "damage_scale": 100,
            }}), encoding="utf-8")
            payload = {
                "Player1": {"x": -50, "y": 20, "CurrDamage": 30,
                            "Eliminated": False},
                "Player2": {"x": 150, "y": -40, "CurrDamage": 60,
                            "Eliminated": False},
            }
            (root / "PlayerStatuses.json").write_text(
                json.dumps(payload), encoding="utf-8")
            result = self.run_read_only_cli(
                "observe", "--config", str(config), "--telemetry-dir", directory,
                "--duration", "0.01", "--timeout", "5")

        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(records), 1, result.stdout)
        record = records[0]
        self.assertEqual(record["source"], "spectator")
        self.assertEqual(record["self_player"], "Player1")
        self.assertEqual(record["opponent_player"], "Player2")
        self.assertEqual(len(record["observation"]), 10)
        np.testing.assert_allclose(record["observation"],
                                   [-0.5, 0.2, 0.3, 0, 1.5, -0.4, 0.6, 0, 2, -0.6])
        self.assertTrue(record["reward_projection"]["baseline_reset"])
        self.assertEqual(record["reward_projection"]["reward"], 0)
        self.assertIn("no controls or learning", result.stderr)
        self.assertIn("Observed 1 fresh updates", result.stderr)

    def test_observe_without_feed_returns_nonzero_without_desktop_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text("{}", encoding="utf-8")
            result = self.run_read_only_cli(
                "observe", "--config", str(config), "--telemetry-dir", directory,
                "--duration", "0.01", "--timeout", "0.1")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("No usable player observations", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_checkpoint_defaults_are_separate_and_custom_suffix_is_preserved(self):
        visual = checkpoint_path(None, "visual")
        state = checkpoint_path(None, "bridge")
        self.assertNotEqual(visual, state)
        self.assertEqual(visual, Path("models/ppo_brawlhalla.zip"))
        self.assertEqual(state, Path("models/ppo_state.zip"))
        self.assertEqual(checkpoint_path("models/policy.v2", "bridge"),
                         Path("models/policy.v2.zip"))
        self.assertEqual(checkpoint_path("models/policy.v2.zip", "bridge"),
                         Path("models/policy.v2.zip"))


if __name__ == "__main__":
    unittest.main()
