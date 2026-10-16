"""Synthetic loopback sessions validate orchestration, not real-game readiness."""

import copy
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

import numpy as np

from session_bridge import OBSERVATION_SCHEMA, SessionBridgeClient, SessionEnv
from session_profiles import make_profile
from state_bridge import BridgeError


class SessionProviderFixture:
    """Project protocol fixture with explicit roles and real socket transport."""

    def __init__(self, profile, alter=None, episode_steps=3):
        self.profile = copy.deepcopy(profile)
        self.alter = alter
        self.episode_steps = episode_steps
        self.requests = []
        self.episode = 0
        self.tick = 0
        self.session_id = "session-fixture"
        self.players = None
        self.released = threading.Event()
        self.finished = threading.Event()
        self.connection = None
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(2)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _response(self, request):
        command = request["command"]
        if command == "hello":
            return {"request_id": request["request_id"], "protocol": 2,
                    "observation_schema": OBSERVATION_SCHEMA,
                    "mode": "online_controlled" if self.profile["online"] else "offline_controlled",
                    "controlled_player": self.profile["self_player"],
                    "game_build": "11.00.36",
                    "capabilities": ["state", "actions", "session", "next_episode"]}
        if command == "configure":
            return {"request_id": request["request_id"], "session_id": self.session_id,
                    **copy.deepcopy(self.profile)}
        if command == "next_episode":
            self.episode += 1
            self.tick = 0
            slots = [self.profile["self_player"], *self.profile["opponents"]]
            if self.profile["teammate_player"] is not None:
                slots.append(self.profile["teammate_player"])
            self.players = {slot: {"x": index * 100, "y": 10, "CurrDamage": 0,
                                    "Eliminated": False, "BrawlhallaID": self.episode * 100 + index,
                                    "PlayerName": f"participant-{self.episode}-{index}"}
                            for index, slot in enumerate(slots)}
        elif command == "step":
            self.tick += 1
            self.players[self.profile["self_player"]]["CurrDamage"] += 2
            for slot in self.profile["opponents"]:
                self.players[slot]["CurrDamage"] += 10
            if self.profile["teammate_player"] is not None:
                self.players[self.profile["teammate_player"]]["CurrDamage"] += 4
        else:
            raise AssertionError(f"unexpected session request {command}")
        return {"request_id": request["request_id"], "session_id": self.session_id,
                "episode_id": f"match-{self.episode}", "tick": self.tick,
                "terminated": self.tick >= self.episode_steps, "truncated": False,
                "players": copy.deepcopy(self.players), **copy.deepcopy(self.profile)}

    def _serve(self):
        try:
            connection, _ = self.listener.accept()
            self.connection = connection
            connection.settimeout(3)
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
                    if response is not None:
                        connection.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except (OSError, ValueError):
            pass
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


class SessionBridgeTests(unittest.TestCase):
    def setup_session(self, name="bot_training", *, profile=None, alter=None,
                      episode_steps=3, **client_settings):
        profile = make_profile(name) if profile is None else profile
        provider = SessionProviderFixture(profile, alter, episode_steps)
        self.addCleanup(provider.close)
        client = SessionBridgeClient(port=provider.port, expected_build="11.00.36", **client_settings)
        self.addCleanup(client.close)
        env = SessionEnv(client, profile)
        self.addCleanup(env.close)
        return env, client, provider

    def assert_released(self, client, provider):
        self.assertTrue(provider.released.wait(1), provider.requests)
        self.assertIsNone(client._socket)

    def test_online_teams_roles_features_reward_and_confirmed_session(self):
        profile = make_profile("online_teams", friendly_fire=True)
        env, client, provider = self.setup_session(profile=profile)
        observation, info = env.reset(seed=12)
        self.assertEqual(observation.shape, (26,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["profile"], profile)
        self.assertEqual(info["session_id"], "session-fixture")
        self.assertEqual(info["game_build"], "11.00.36")
        self.assertEqual(observation[-2:].tolist(), [1, 1])
        observation, reward, terminated, truncated, info = env.step(10)
        self.assertAlmostEqual(reward, 0.14)
        self.assertEqual(info["projection"]["damage_dealt"], 20)
        self.assertEqual(info["projection"]["damage_received"], 6)
        self.assertFalse(terminated or truncated)
        self.assertEqual([request["command"] for request in provider.requests],
                         ["hello", "configure", "next_episode", "step"])
        configure, next_episode = provider.requests[1:3]
        for field, value in profile.items():
            self.assertEqual(configure[field], value)
            self.assertEqual(next_episode[field], value)
        self.assertIsNone(next_episode["after_episode_id"])
        self.assertEqual(next_episode["seed"], 12)
        self.assertEqual(provider.requests[-1]["action"], ["left", "light"])
        self.assertEqual(client.expected_profile, profile)
        env.close()
        self.assert_released(client, provider)

    def test_continuous_bot_episodes_use_next_match_instead_of_reset(self):
        env, _, provider = self.setup_session(episode_steps=2)
        _, info = env.reset()
        first_episode = info["episode_id"]
        _, _, terminated, _, _ = env.step(1)
        self.assertFalse(terminated)
        _, _, terminated, _, _ = env.step(1)
        self.assertTrue(terminated)
        _, second = env.reset()
        self.assertNotEqual(second["episode_id"], first_episode)
        self.assertTrue(second["projection"]["baseline_reset"])
        self.assertEqual([request["command"] for request in provider.requests].count("configure"), 1)
        next_requests = [request for request in provider.requests if request["command"] == "next_episode"]
        self.assertEqual(next_requests[1]["after_episode_id"], first_episode)
        self.assertFalse(any(request["command"] == "reset" for request in provider.requests))

    def test_all_supported_profiles_and_two_bot_formats_share_observation_shape(self):
        profiles = [make_profile(name) for name in ("human_duel", "bot_training", "online_duel", "online_teams")]
        profiles.append(make_profile("bot_training", match_format="2v2"))
        for profile in profiles:
            with self.subTest(profile=profile):
                env, _, _ = self.setup_session(profile=profile)
                observation, _ = env.reset()
                self.assertEqual(observation.shape, (26,))
                if profile["match_format"] == "1v1":
                    np.testing.assert_array_equal(observation[10:], np.zeros(16))
                env.close()

    def test_v1_and_incomplete_capability_providers_are_refused(self):
        for field, value in (("protocol", 1), ("observation_schema", "brawlhalla-players-v1"),
                             ("capabilities", ["state", "actions", "reset"]),
                             ("capabilities", ["state", "actions", "session"]),
                             ("mode", "spectator")):
            with self.subTest(field=field, value=value):
                def alter(request, response):
                    if request["command"] == "hello":
                        response[field] = value
                    return response
                env, client, provider = self.setup_session(alter=alter)
                with self.assertRaises(BridgeError):
                    env.reset()
                self.assert_released(client, provider)
                self.assertFalse(any(request["command"] == "configure" for request in provider.requests))

    def test_online_mode_or_controlled_player_mismatch_refused_before_configure(self):
        for field, value in (("mode", "offline_controlled"), ("controlled_player", "Player2")):
            with self.subTest(field=field):
                def alter(request, response):
                    if request["command"] == "hello":
                        response[field] = value
                    return response
                env, client, provider = self.setup_session("online_duel", alter=alter)
                with self.assertRaises(BridgeError):
                    env.reset()
                self.assert_released(client, provider)
                self.assertEqual([request["command"] for request in provider.requests], ["hello", "release"])

    def test_configure_requires_exact_profile_fields_session_and_request_ack(self):
        changes = [("match_format", "2v2"), ("opponents", ["Player3"]),
                   ("online", True), ("session_id", ""), ("request_id", 99),
                   ("request_id", True), ("friendly_fire", 0)]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                def alter(request, response):
                    if request["command"] == "configure":
                        response[field] = value
                    return response
                env, client, provider = self.setup_session(alter=alter)
                with self.assertRaises(BridgeError):
                    env.reset()
                self.assert_released(client, provider)
                self.assertFalse(any(request["command"] == "next_episode" for request in provider.requests))

    def test_provider_arrangement_failure_is_reported_without_retry(self):
        def alter(request, response):
            if request["command"] == "configure":
                return {"request_id": request["request_id"], "error": "cannot verify requested game lobby"}
            return response
        env, client, provider = self.setup_session(alter=alter)
        with self.assertRaisesRegex(BridgeError, "cannot verify requested game lobby"):
            env.reset()
        self.assert_released(client, provider)
        self.assertEqual(sum(request["command"] == "configure" for request in provider.requests), 1)

    def test_step_rejects_roster_team_format_session_and_stale_tick_changes(self):
        def wrong_roster(response):
            response["players"]["Player9"] = response["players"].pop("Player4")
        def extra_player(response):
            response["players"]["Player9"] = copy.deepcopy(response["players"]["Player4"])
        def reversed_opponents(response):
            response["opponents"].reverse()
        changes = [wrong_roster, extra_player, reversed_opponents,
                   lambda response: response.update(match_format="1v1"),
                   lambda response: response.update(teammate_player="Player3", opponents=["Player2", "Player4"]),
                   lambda response: response.update(session_id="another-session"),
                   lambda response: response.update(episode_id="another-match"),
                   lambda response: response.update(tick=0),
                   lambda response: response.update(online=False)]
        for mutation in changes:
            with self.subTest(mutation=mutation):
                def alter(request, response):
                    if request["command"] == "step":
                        mutation(response)
                    return response
                env, client, provider = self.setup_session("online_teams", alter=alter)
                env.reset()
                with self.assertRaises(BridgeError):
                    env.step(1)
                self.assert_released(client, provider)
                self.assertEqual(sum(request["command"] == "step" for request in provider.requests), 1)

    def test_new_episode_ids_cannot_reuse_any_earlier_id(self):
        def alter(request, response):
            if request["command"] == "next_episode" and response["episode_id"] == "match-3":
                response["episode_id"] = "match-1"
            return response
        env, client, provider = self.setup_session(alter=alter)
        env.reset()
        env.reset()
        with self.assertRaisesRegex(BridgeError, "globally new episode_id"):
            env.reset()
        self.assert_released(client, provider)

    def test_inactive_or_terminal_match_is_not_accepted_as_new_episode(self):
        changes = [lambda response: response.update(terminated=True),
                   lambda response: response.update(truncated=True),
                   lambda response: response["players"]["Player2"].update(Eliminated=True)]
        for mutation in changes:
            with self.subTest(mutation=mutation):
                def alter(request, response):
                    if request["command"] == "next_episode":
                        mutation(response)
                    return response
                env, client, provider = self.setup_session(alter=alter)
                with self.assertRaises(BridgeError):
                    env.reset()
                self.assert_released(client, provider)

    def test_identity_changes_within_match_stop_but_new_match_players_are_allowed(self):
        def alter(request, response):
            if request["command"] == "step":
                response["players"]["Player2"]["BrawlhallaID"] = 99999
            return response
        env, client, provider = self.setup_session("online_teams", alter=alter)
        env.reset()
        env.reset()  # The fixture deliberately supplies new identities in each match.
        with self.assertRaisesRegex(BridgeError, "identity"):
            env.step(1)
        self.assert_released(client, provider)

    def test_elimination_proxy_does_not_end_match_and_authoritative_truncation_does(self):
        def alter(request, response):
            if request["command"] == "step":
                response["players"]["Player2"]["Eliminated"] = True
                response["truncated"] = response["tick"] >= 2
            return response
        env, _, _ = self.setup_session(alter=alter, episode_steps=99)
        env.reset()
        _, _, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated or truncated)
        self.assertTrue(info["projection"]["terminal_proxy"])
        _, _, terminated, truncated, _ = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_waiting_for_human_match_has_separate_bounded_timeout_and_stop_release(self):
        def delayed(request, response):
            if request["command"] == "next_episode":
                time.sleep(0.08)
            return response
        env, client, _ = self.setup_session("human_duel", alter=delayed,
                                             timeout=0.03, episode_timeout=0.3)
        env.reset()
        self.assertEqual(client.timeout, 0.03)
        env.close()

        stop = threading.Event()
        def waiting(request, response):
            if request["command"] == "next_episode":
                stop.set()
                return None
            return response
        env, client, provider = self.setup_session("online_duel", alter=waiting,
                                                   episode_timeout=30, stop_requested=stop.is_set)
        started = time.monotonic()
        with self.assertRaises(KeyboardInterrupt):
            env.reset()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assert_released(client, provider)

    def test_next_episode_timeout_closes_without_replaying_request(self):
        def waiting(request, response):
            return None if request["command"] == "next_episode" else response
        env, client, provider = self.setup_session(alter=waiting, episode_timeout=0.04)
        with self.assertRaisesRegex(BridgeError, "timed out"):
            env.reset()
        self.assert_released(client, provider)
        self.assertEqual(sum(request["command"] == "next_episode" for request in provider.requests), 1)

    def test_legacy_reset_is_unavailable_and_profile_cannot_change_in_place(self):
        env, client, provider = self.setup_session()
        env.reset()
        with self.assertRaisesRegex(BridgeError, "legacy reset"):
            client.reset()
        with self.assertRaisesRegex(BridgeError, "cannot switch"):
            client.configure(make_profile("human_duel"))
        self.assert_released(client, provider)

    def test_shared_26_feature_policy_learns_duels_and_resumes_for_teams(self):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        torch.set_num_threads(2)
        duel_env, _, _ = self.setup_session()
        duel = DummyVecEnv([lambda: duel_env])
        self.addCleanup(duel.close)
        model = PPO("MlpPolicy", duel, n_steps=8, batch_size=8, n_epochs=1, device="cpu")
        before = [parameter.detach().clone() for parameter in model.policy.parameters()]
        model.learn(16)
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, model.policy.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "shared-policy"
            model.save(checkpoint)
            team_env, _, _ = self.setup_session("online_teams")
            teams = DummyVecEnv([lambda: team_env])
            self.addCleanup(teams.close)
            resumed = PPO.load(checkpoint, env=teams, device="cpu")
            observation = teams.reset()
            expected, _ = model.predict(observation, deterministic=True)
            actual, _ = resumed.predict(observation, deterministic=True)
            np.testing.assert_array_equal(expected, actual)
            resumed.learn(8, reset_num_timesteps=False)
            self.assertEqual(resumed.num_timesteps, 24)


if __name__ == "__main__":
    unittest.main()
