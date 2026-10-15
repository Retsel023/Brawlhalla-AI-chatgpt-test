"""Exercise the project bridge over real loopback sockets, without the game."""

import copy
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

import numpy as np

from state_bridge import (
    BridgeError, ControlledStateEnv, MAX_RESPONSE_BYTES, OBSERVATION_SCHEMA,
    StateBridgeClient,
)


def players():
    return {
        "Player1": {"x": -100, "y": 20, "CurrDamage": 0, "Eliminated": False,
                    "BrawlhallaID": 10, "PlayerName": "Agent"},
        "Player2": {"x": 100, "y": 20, "CurrDamage": 0, "Eliminated": False,
                    "BrawlhallaID": 20, "PlayerName": "Opponent"},
    }


class FakeProvider:
    """Synthetic protocol fixture; it neither launches nor instruments Brawlhalla."""

    def __init__(self, alter=None, episode_steps=None):
        self.alter = alter
        self.episode_steps = episode_steps
        self.requests = []
        self.errors = []
        self.episode = 0
        self.tick = 0
        self.players = players()
        self.released = threading.Event()
        self.finished = threading.Event()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(2)
        self.port = self.listener.getsockname()[1]
        self.connection = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _response(self, request):
        command = request["command"]
        if command == "hello":
            return {"request_id": request["request_id"], "protocol": 1,
                    "mode": "offline_controlled", "game_build": "11.00.36",
                    "observation_schema": OBSERVATION_SCHEMA,
                    "controlled_player": "Player1",
                    "capabilities": ["state", "actions", "reset"]}
        if command == "reset":
            self.episode += 1
            self.tick = 0
            self.players = players()
        elif command == "step":
            self.tick += 1
            self.players["Player2"]["CurrDamage"] += 10
            self.players["Player1"]["CurrDamage"] += 2
        else:
            raise AssertionError(f"unexpected request {command}")
        return {"request_id": request["request_id"], "episode_id": f"episode-{self.episode}",
                "tick": self.tick, "players": copy.deepcopy(self.players),
                "terminated": False,
                "truncated": bool(self.episode_steps and self.tick >= self.episode_steps)}

    def _serve(self):
        try:
            connection, _ = self.listener.accept()
            self.connection = connection
            connection.settimeout(2)
            with connection, connection.makefile("rb") as stream:
                while raw := stream.readline():
                    request = json.loads(raw)
                    self.requests.append(request)
                    if request["command"] == "release":
                        self.released.set()
                        break
                    response = self._response(request)
                    if self.alter is not None:
                        response = self.alter(request, response)
                    if response is None:
                        continue
                    if not isinstance(response, bytes):
                        response = (json.dumps(response) + "\n").encode("utf-8")
                    connection.sendall(response)
        except (OSError, ValueError) as error:
            # Disconnect while returning a deliberately invalid fixture is normal.
            if not isinstance(error, (BrokenPipeError, ConnectionResetError)):
                self.errors.append(error)
        finally:
            self.finished.set()

    def close(self):
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.listener.close()
        self.thread.join(timeout=2)


class StateBridgeTests(unittest.TestCase):
    def setup_bridge(self, alter=None, *, episode_steps=None, **kwargs):
        provider = FakeProvider(alter, episode_steps=episode_steps)
        self.addCleanup(provider.close)
        client = StateBridgeClient(port=provider.port, expected_build="11.00.36", **kwargs)
        self.addCleanup(client.close)
        return client, provider

    def assert_released(self, client, provider):
        self.assertTrue(provider.released.wait(1), provider.requests)
        self.assertIsNone(client._socket)
        with self.assertRaisesRegex(BridgeError, "closed"):
            client.connect()

    def test_handshake_reset_actions_reward_and_close(self):
        client, provider = self.setup_bridge()
        env = ControlledStateEnv(client)
        observation, info = env.reset(seed=42)
        self.assertEqual(observation.shape, (10,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["game_build"], "11.00.36")
        self.assertTrue(info["projection"]["baseline_reset"])
        observation, reward, terminated, truncated, info = env.step(10)
        self.assertAlmostEqual(reward, 0.08)
        self.assertEqual((terminated, truncated), (False, False))
        self.assertEqual(info["tick"], 1)
        self.assertEqual(info["projection"]["damage_dealt"], 10)
        self.assertEqual(provider.requests[0]["observation_schema"], OBSERVATION_SCHEMA)
        self.assertEqual(provider.requests[1]["seed"], 42)
        self.assertEqual(provider.requests[2]["action"], ["left", "light"])
        self.assertEqual(provider.requests[2]["hold_seconds"], 0.1)
        env.close()
        env.close()
        self.assert_released(client, provider)
        self.assertEqual([r["request_id"] for r in provider.requests], [1, 2, 3, 4])

    def test_handshake_rejects_spectator_build_and_missing_capabilities(self):
        cases = [("mode", "spectator"), ("protocol", True),
                 ("observation_schema", "another-schema"), ("game_build", "11.00.35"),
                 ("game_build", ""), ("capabilities", ["state", "actions"]),
                 ("controlled_player", None), ("controlled_player", "PlayerX")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                def alter(request, response):
                    response[field] = value
                    return response
                client, provider = self.setup_bridge(alter)
                with self.assertRaises(BridgeError):
                    client.connect()
                self.assert_released(client, provider)

    def test_controlled_player_must_match_self_slot_before_reset_or_actions(self):
        client, provider = self.setup_bridge()
        env = ControlledStateEnv(client, self_slot="Player2", opponent_slot="Player1")
        with self.assertRaisesRegex(BridgeError, "controlled_player"):
            env.reset()
        self.assert_released(client, provider)
        self.assertEqual([request["command"] for request in provider.requests], ["hello", "release"])

        def alter(request, response):
            if request["command"] == "hello":
                response["controlled_player"] = "Player2"
            return response
        client, provider = self.setup_bridge(alter)
        env = ControlledStateEnv(client, self_slot="Player2", opponent_slot="Player1")
        env.reset()
        _, reward, _, _, _ = env.step(1)
        self.assertEqual(client.controlled_player, "Player2")
        self.assertAlmostEqual(reward, -0.08)

    def test_invalid_response_envelopes_fail_closed(self):
        cases = [("tick", True), ("tick", -1), ("terminated", 0),
                 ("truncated", None), ("episode_id", ""), ("players", {}),
                 ("request_id", True), ("request_id", 900)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                def alter(request, response):
                    if request["command"] == "reset":
                        response[field] = value
                    return response
                client, provider = self.setup_bridge(alter)
                with self.assertRaises(BridgeError):
                    client.reset()
                self.assert_released(client, provider)

    def test_malformed_json_nonfinite_duplicate_and_oversize_fail_closed(self):
        invalid = [b"not-json\n", b"[]\n", b'{"request_id":1,"value":NaN}\n',
                   b'{"request_id":1,"value":1e999}\n',
                   b'{"request_id":1,"request_id":1}\n', b"\xff\n",
                   b'{}\n{}\n', b"x" * MAX_RESPONSE_BYTES + b"\n"]
        for raw in invalid:
            with self.subTest(raw=raw[:50]):
                client, provider = self.setup_bridge(lambda request, response: raw)
                with self.assertRaises(BridgeError):
                    client.connect()
                self.assert_released(client, provider)

    def test_step_requires_increasing_tick_matching_episode_and_response_id(self):
        cases = [("tick", 0), ("episode_id", "another-episode"), ("request_id", 2)]
        for field, value in cases:
            with self.subTest(field=field):
                def alter(request, response):
                    if request["command"] == "step":
                        response[field] = value
                    return response
                client, provider = self.setup_bridge(alter)
                client.reset()
                with self.assertRaises(BridgeError):
                    client.step(["left"], 0.1)
                self.assert_released(client, provider)
                self.assertEqual(sum(r["command"] == "step" for r in provider.requests), 1)

    def test_reset_requires_new_episode_even_when_previous_episode_not_finished(self):
        def alter(request, response):
            if request["command"] == "reset":
                response["episode_id"] = "same-episode"
            return response
        client, provider = self.setup_bridge(alter)
        client.reset()
        with self.assertRaisesRegex(BridgeError, "new episode_id"):
            client.reset()
        self.assert_released(client, provider)

    def test_reset_requires_selected_active_players_and_nonterminal_state(self):
        def missing(request, response):
            response["players"].pop("Player2")

        def eliminated(request, response):
            response["players"]["Player1"]["Eliminated"] = True

        def terminal(request, response):
            response["terminated"] = True

        def missing_field(request, response):
            response["players"]["Player1"].pop("x")

        for mutation in (missing, eliminated, terminal, missing_field):
            with self.subTest(mutation=mutation.__name__):
                def alter(request, response):
                    if request["command"] == "reset":
                        mutation(request, response)
                    return response
                client, provider = self.setup_bridge(alter)
                env = ControlledStateEnv(client)
                with self.assertRaises(BridgeError):
                    env.reset()
                self.assert_released(client, provider)

    def test_identity_change_stops_environment_before_returning_transition(self):
        def alter(request, response):
            if request["command"] == "step":
                response["players"]["Player1"]["BrawlhallaID"] = 100
            return response
        client, provider = self.setup_bridge(alter)
        env = ControlledStateEnv(client)
        env.reset()
        with self.assertRaisesRegex(BridgeError, "identity changed"):
            env.step(1)
        self.assert_released(client, provider)

    def test_damage_decrease_rebaselines_without_inventing_termination(self):
        def alter(request, response):
            if request["command"] == "step" and response["tick"] == 2:
                response["players"]["Player2"]["CurrDamage"] = 0
            return response
        client, _ = self.setup_bridge(alter)
        env = ControlledStateEnv(client)
        env.reset()
        env.step(1)
        _, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated or truncated)
        self.assertTrue(info["projection"]["baseline_reset"])
        self.assertIn("Player2_damage_decreased", info["projection"]["diagnostics"])

    def test_explicit_terminal_flags_override_elimination_proxy_and_reset_baseline(self):
        def alter(request, response):
            if request["command"] == "step":
                for player in response["players"].values():
                    player["Eliminated"] = True
                response["terminated"] = response["tick"] >= 2
            return response
        client, _ = self.setup_bridge(alter)
        env = ControlledStateEnv(client)
        env.reset()
        _, _, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated or truncated)
        self.assertTrue(info["projection"]["terminal_proxy"])
        self.assertEqual(info["projection"]["eliminated_slots"], ("Player1", "Player2"))
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["projection"]["terminal_proxy"])
        with self.assertRaisesRegex(BridgeError, "reset"):
            env.step(1)
        _, reset_info = env.reset()
        self.assertNotEqual(reset_info["episode_id"], info["episode_id"])
        self.assertTrue(reset_info["projection"]["baseline_reset"])

    def test_timeout_releases_and_never_retries_action(self):
        def alter(request, response):
            return None if request["command"] == "step" else response
        client, provider = self.setup_bridge(alter, timeout=0.04)
        env = ControlledStateEnv(client)
        env.reset()
        started = time.monotonic()
        with self.assertRaisesRegex(BridgeError, "timed out"):
            env.step(1)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assert_released(client, provider)
        self.assertEqual(sum(r["command"] == "step" for r in provider.requests), 1)

    def test_disconnect_before_response_closes_client_without_retry(self):
        holder = {}
        def alter(request, response):
            if request["command"] == "step":
                holder["provider"].connection.shutdown(socket.SHUT_WR)
                return None
            return response
        client, provider = self.setup_bridge(alter)
        holder["provider"] = provider
        client.reset()
        with self.assertRaisesRegex(BridgeError, "disconnected"):
            client.step(["left"], 0.1)
        self.assert_released(client, provider)
        self.assertEqual(sum(r["command"] == "step" for r in provider.requests), 1)

    def test_user_stop_interrupts_wait_promptly_and_releases(self):
        stop = threading.Event()
        def alter(request, response):
            if request["command"] == "step":
                stop.set()
                return None
            return response
        client, provider = self.setup_bridge(alter, stop_requested=stop.is_set)
        env = ControlledStateEnv(client)
        env.reset()
        started = time.monotonic()
        with self.assertRaises(KeyboardInterrupt):
            env.step(1)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assert_released(client, provider)

    def test_rejects_nonloopback_or_invalid_configuration_without_connecting(self):
        configurations = [dict(host="example.com"), dict(host="192.168.1.2"),
                          dict(host="0.0.0.0"), dict(port=True), dict(port=0),
                          dict(timeout=float("nan")), dict(timeout=float("inf")),
                          dict(timeout=-1), dict(timeout=True), dict(expected_build="")]
        for configuration in configurations:
            with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                StateBridgeClient(**configuration)

    def test_actual_mlp_learning_updates_and_checkpoint_resumes(self):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        torch.set_num_threads(2)
        client, _ = self.setup_bridge(episode_steps=4)
        env = DummyVecEnv([lambda: ControlledStateEnv(client)])
        self.addCleanup(env.close)
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, device="cpu")
        before = [parameter.detach().clone() for parameter in model.policy.parameters()]
        model.learn(16)
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, model.policy.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "structured"
            observation = env.reset()
            expected, _ = model.predict(observation, deterministic=True)
            model.save(checkpoint)
            resumed = PPO.load(checkpoint, env=env, device="cpu")
            actual, _ = resumed.predict(observation, deterministic=True)
            np.testing.assert_array_equal(expected, actual)
            resumed.learn(8, reset_num_timesteps=False)
            self.assertEqual(resumed.num_timesteps, 24)


if __name__ == "__main__":
    unittest.main()
