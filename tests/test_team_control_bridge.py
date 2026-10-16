"""Synthetic protocol-3 peers test atomic control, not a Windows game provider."""

import copy
import json
import socket
import threading
import time
import unittest

from controls import ACTIONS
from session_profiles import make_profile
from state_bridge import BridgeError
from team_control_bridge import TeamControlClient


class TeamControlProviderFixture:
    """One real loopback connection with deliberately explicit control ownership."""

    def __init__(self, profile, alter=None, episode_steps=20):
        self.profile = copy.deepcopy(profile)
        self.alter = alter
        self.episode_steps = episode_steps
        self.requests = []
        self.episode = 0
        self.tick = 0
        self.players = None
        self.session_id = "controlled-team-fixture"
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
            controlled = [self.profile["self_player"]]
            if self.profile["teammate_kind"] == "ai":
                controlled.append(self.profile["teammate_player"])
            return {"request_id": request["request_id"], "protocol": 3,
                    "observation_schema": "brawlhalla-team-v2",
                    "mode": "online_controlled" if self.profile["online"] else "offline_controlled",
                    "controlled_players": controlled, "game_build": "11.00.36",
                    "input_routes": {slot: f"keyboard:{index}" for index, slot in enumerate(controlled)},
                    "capabilities": ["state", "actions", "session", "next_episode",
                                     "team_setup", "joint_actions", "independent_inputs"]}
        if command == "configure":
            return {"request_id": request["request_id"], "session_id": self.session_id,
                    **copy.deepcopy(self.profile)}
        if command == "next_episode":
            self.episode += 1
            self.tick = 0
            slots = [self.profile["self_player"], *self.profile["opponents"]]
            if self.profile["teammate_player"] is not None:
                slots.append(self.profile["teammate_player"])
            self.players = {
                slot: {"x": index * 100, "y": 10, "CurrDamage": 0,
                       "Eliminated": False, "BrawlhallaID": self.episode * 100 + index,
                       "PlayerName": f"participant-{self.episode}-{index}"}
                for index, slot in enumerate(slots)
            }
            if self.profile["teammate_kind"] == "regular":
                self.players[self.profile["teammate_player"]]["BrawlhallaID"] = (
                    self.profile["teammate_account_id"]
                    if self.profile["teammate_account_id"] is not None else 700)
        elif command in ("step", "step_joint"):
            self.tick += 1
            self.players[self.profile["self_player"]]["CurrDamage"] += 2
            for slot in self.profile["opponents"]:
                self.players[slot]["CurrDamage"] += 10
            if self.profile["teammate_player"] is not None:
                self.players[self.profile["teammate_player"]]["CurrDamage"] += 4
        else:
            raise AssertionError(f"unexpected team-control request {command}")
        return {"request_id": request["request_id"], "session_id": self.session_id,
                "episode_id": f"controlled-match-{self.episode}", "tick": self.tick,
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


class TeamControlBridgeTests(unittest.TestCase):
    def setup_control(self, *, profile=None, alter=None, **client_settings):
        if profile is None:
            profile = make_profile("online_teams", teammate_kind="ai")
        provider = TeamControlProviderFixture(profile, alter)
        self.addCleanup(provider.close)
        client = TeamControlClient(port=provider.port, expected_build="11.00.36", **client_settings)
        self.addCleanup(client.close)
        return client, provider, profile

    def start_episode(self, client, profile, seed=None):
        self.assertEqual(client.configure(profile), "controlled-team-fixture")
        return client.next_episode(seed=seed)

    def assert_released(self, client, provider):
        self.assertTrue(provider.released.wait(1), provider.requests)
        self.assertIsNone(client._socket)

    def commands(self, provider):
        return [request["command"] for request in provider.requests]

    def joint_actions(self, profile):
        return {profile["self_player"]: ACTIONS[8], profile["teammate_player"]: ACTIONS[11]}

    def test_both_controls_share_one_atomic_request_and_acknowledged_tick(self):
        profile = make_profile("online_teams", self_player="Player7", teammate_player="Player12",
                               opponents=["Player3", "Player4"], teammate_kind="ai")
        client, provider, _ = self.setup_control(profile=profile)
        initial = self.start_episode(client, profile, seed=12)
        self.assertEqual(initial.tick, 0)
        state = client.step_joint(self.joint_actions(profile), 0.125)
        self.assertEqual(state.tick, 1)
        self.assertEqual(state.episode_id, initial.episode_id)
        self.assertEqual(state.players["Player7"]["CurrDamage"], 2)
        self.assertEqual(state.players["Player12"]["CurrDamage"], 4)
        self.assertEqual(self.commands(provider), ["hello", "configure", "next_episode", "step_joint"])
        request = provider.requests[-1]
        self.assertEqual(request["actions"], {"Player7": ["left", "jump"],
                                              "Player12": ["right", "light"]})
        self.assertEqual(request["hold_seconds"], 0.125)
        self.assertNotIn("action", request)
        self.assertEqual(client.step_joint(self.joint_actions(profile), 0.1).tick, 2)
        self.assertEqual(self.commands(provider).count("step_joint"), 2)
        for command in ("configure", "next_episode"):
            request = next(item for item in provider.requests if item["command"] == command)
            for field, value in profile.items():
                self.assertEqual(request[field], value)
        self.assertEqual(provider.requests[2]["seed"], 12)
        self.assertEqual(client.expected_profile, profile)
        self.assertEqual(client.input_routes, {"Player7": "keyboard:0", "Player12": "keyboard:1"})

    def test_control_validation_is_read_only_hello_and_reuses_the_connection(self):
        client, provider, profile = self.setup_control()
        self.assertIsNone(client.validate_control(profile))
        self.assertEqual(self.commands(provider), ["hello"])
        self.assertIsNone(client.session_id)
        self.assertIsNone(client.expected_profile)
        self.start_episode(client, profile)
        self.assertEqual(self.commands(provider), ["hello", "configure", "next_episode"])

    def test_controlled_players_need_not_include_a_legacy_scalar_or_be_sorted(self):
        def alter(request, response):
            if request["command"] == "hello":
                response["controlled_players"].reverse()
                self.assertNotIn("controlled_player", response)
            return response
        client, provider, profile = self.setup_control(alter=alter)
        self.start_episode(client, profile)
        client.step_joint(self.joint_actions(profile), 0.1)
        self.assertEqual(self.commands(provider)[-1], "step_joint")

    def test_ai_control_requirements_fail_before_configure_or_actions(self):
        mutations = [
            lambda response: response["capabilities"].remove("joint_actions"),
            lambda response: response["capabilities"].remove("independent_inputs"),
            lambda response: response.update(controlled_players=["Player1"]),
            lambda response: response.update(controlled_players=["Player1", "Player2", "Player3"]),
            lambda response: response.update(controlled_players=["Player1", "Player1"]),
            lambda response: response.update(controlled_players=["Player1", "ally"]),
            lambda response: response.update(controlled_player="Player3"),
            lambda response: response.pop("input_routes"),
            lambda response: response.update(input_routes={"Player1": "keyboard:0"}),
            lambda response: response.update(input_routes={"Player1": "keyboard:0", "Player2": "keyboard:0"}),
            lambda response: response.update(input_routes={"Player1": "keyboard:0", "Player2": ""}),
            lambda response: response.update(input_routes={"Player1": "keyboard:0", "Player2": "keyboard:1",
                                                          "Player3": "keyboard:2"}),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                def alter(request, response):
                    if request["command"] == "hello":
                        mutation(response)
                    return response
                client, provider, profile = self.setup_control(alter=alter)
                with self.assertRaises(BridgeError):
                    client.configure(profile)
                self.assert_released(client, provider)
                self.assertEqual(self.commands(provider), ["hello", "release"])

    def test_v3_hello_rejects_legacy_and_incomplete_provider_contracts(self):
        changes = [("protocol", 2), ("protocol", True),
                   ("observation_schema", "brawlhalla-players-v1"),
                   ("mode", "spectator"), ("mode", "offline_controlled"),
                   ("game_build", "wrong-build"), ("controlled_players", "Player1"),
                   ("controlled_players", [])]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                def alter(request, response):
                    if request["command"] == "hello":
                        response[field] = value
                    return response
                client, provider, profile = self.setup_control(alter=alter)
                with self.assertRaises(BridgeError):
                    client.validate_control(profile)
                self.assert_released(client, provider)
                self.assertEqual(self.commands(provider), ["hello", "release"])
        def no_team_setup(request, response):
            if request["command"] == "hello":
                response["capabilities"].remove("team_setup")
            return response
        client, provider, profile = self.setup_control(alter=no_team_setup)
        with self.assertRaises(BridgeError):
            client.configure(profile)
        self.assert_released(client, provider)
        self.assertNotIn("configure", self.commands(provider))

    def test_ai_profile_cannot_send_legacy_single_player_step(self):
        client, provider, profile = self.setup_control()
        self.start_episode(client, profile)
        with self.assertRaises(BridgeError):
            client.step(ACTIONS[1], 0.1)
        self.assertNotIn("step", self.commands(provider))
        self.assertNotIn("step_joint", self.commands(provider))

    def test_joint_actions_must_exactly_cover_controlled_slots_with_semantic_keys(self):
        client, provider, profile = self.setup_control()
        self.start_episode(client, profile)
        invalid = [
            {"Player1": ACTIONS[1]},
            {"Player1": ACTIONS[1], "Player2": ACTIONS[2], "Player3": ()},
            {"Player1": ACTIONS[1], "Player3": ACTIONS[2]},
            {"Player1": 1, "Player2": ACTIONS[2]},
            {"Player1": "left", "Player2": ACTIONS[2]},
            {"Player1": ("left", "right"), "Player2": ACTIONS[2]},
            [ACTIONS[1], ACTIONS[2]],
        ]
        for actions in invalid:
            with self.subTest(actions=actions):
                with self.assertRaises((ValueError, BridgeError)):
                    client.step_joint(actions, 0.1)
        self.assertNotIn("step_joint", self.commands(provider))

    def test_invalid_hold_is_rejected_without_sending_actions(self):
        client, provider, profile = self.setup_control()
        self.start_episode(client, profile)
        for hold in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(hold=hold):
                with self.assertRaises(ValueError):
                    client.step_joint(self.joint_actions(profile), hold)
        self.assertNotIn("step_joint", self.commands(provider))

    def test_random_and_regular_humans_require_only_the_primary_control(self):
        def alter(request, response):
            if request["command"] == "hello":
                response["capabilities"].remove("joint_actions")
            return response
        for kind in ("random", "regular"):
            with self.subTest(kind=kind):
                profile = make_profile("online_teams", teammate_kind=kind)
                client, provider, _ = self.setup_control(profile=profile, alter=alter)
                self.start_episode(client, profile)
                state = client.step(ACTIONS[1], 0.1)
                self.assertEqual(state.tick, 1)
                self.assertEqual(provider.requests[-1]["action"], ["left"])
                self.assertNotIn("step_joint", self.commands(provider))

    def test_single_control_routes_are_optional_but_explicit_routes_must_match(self):
        profile = make_profile("online_teams", teammate_kind="random")
        def no_routes(request, response):
            if request["command"] == "hello":
                response.pop("input_routes")
            return response
        client, provider, _ = self.setup_control(profile=profile, alter=no_routes)
        self.start_episode(client, profile)
        self.assertEqual(client.step(ACTIONS[1], 0.1).tick, 1)
        self.assertEqual(client.input_routes, {})

        for routes in ({}, {"Player2": "keyboard:0"}, {"Player1": ""},
                       {"Player1": "keyboard:0", "Player2": "keyboard:1"}):
            with self.subTest(routes=routes):
                def alter(request, response):
                    if request["command"] == "hello":
                        response["input_routes"] = routes
                    return response
                client, provider, _ = self.setup_control(profile=profile, alter=alter)
                with self.assertRaises(BridgeError):
                    client.configure(profile)
                self.assert_released(client, provider)
                self.assertEqual(self.commands(provider), ["hello", "release"])

    def test_regular_teammate_requires_live_account_identity_before_actions(self):
        for identity in (None, "missing"):
            with self.subTest(identity=identity):
                def alter(request, response):
                    if request["command"] == "next_episode":
                        ally = response["players"]["Player2"]
                        if identity == "missing":
                            ally.pop("BrawlhallaID")
                        else:
                            ally["BrawlhallaID"] = identity
                    return response
                profile = make_profile("online_teams", teammate_kind="regular")
                client, provider, _ = self.setup_control(profile=profile, alter=alter)
                client.configure(profile)
                with self.assertRaises(BridgeError):
                    client.next_episode()
                self.assert_released(client, provider)
                self.assertNotIn("step", self.commands(provider))

    def test_explicit_regular_identity_must_match_first_match(self):
        def alter(request, response):
            if request["command"] == "next_episode":
                response["players"]["Player2"]["BrawlhallaID"] = 888
            return response
        profile = make_profile("online_teams", teammate_kind="regular", teammate_account_id=777)
        client, provider, _ = self.setup_control(profile=profile, alter=alter)
        client.configure(profile)
        with self.assertRaises(BridgeError):
            client.next_episode()
        self.assert_released(client, provider)
        self.assertNotIn("step", self.commands(provider))

    def test_regular_identity_stays_pinned_across_matches_and_swap_stops_before_actions(self):
        def alter(request, response):
            if request["command"] == "next_episode" and provider.episode >= 3:
                response["players"]["Player2"]["BrawlhallaID"] = 701
            return response
        profile = make_profile("online_teams", teammate_kind="regular")
        client, provider, _ = self.setup_control(profile=profile, alter=alter)
        first = self.start_episode(client, profile)
        client.step(ACTIONS[1], 0.1)
        second = client.next_episode()
        self.assertNotEqual(first.episode_id, second.episode_id)
        self.assertEqual(first.players["Player2"]["BrawlhallaID"],
                         second.players["Player2"]["BrawlhallaID"])
        with self.assertRaises(BridgeError):
            client.next_episode()
        self.assert_released(client, provider)
        self.assertEqual(self.commands(provider).count("step"), 1)
        with self.assertRaises(BridgeError):
            client.step(ACTIONS[1], 0.1)
        self.assertEqual(self.commands(provider).count("step"), 1)

    def test_regular_identity_change_within_match_releases_without_retry(self):
        def alter(request, response):
            if request["command"] == "step":
                response["players"]["Player2"]["BrawlhallaID"] = 701
            return response
        profile = make_profile("online_teams", teammate_kind="regular")
        client, provider, _ = self.setup_control(profile=profile, alter=alter)
        self.start_episode(client, profile)
        with self.assertRaises(BridgeError):
            client.step(ACTIONS[1], 0.1)
        self.assert_released(client, provider)
        self.assertEqual(self.commands(provider).count("step"), 1)

    def test_random_teammate_identity_can_change_between_matches(self):
        profile = make_profile("online_teams", teammate_kind="random")
        client, provider, _ = self.setup_control(profile=profile)
        first = self.start_episode(client, profile)
        second = client.next_episode()
        self.assertNotEqual(first.players["Player2"]["BrawlhallaID"],
                            second.players["Player2"]["BrawlhallaID"])
        self.assertEqual(client.step(ACTIONS[2], 0.1).tick, 1)
        self.assertEqual(self.commands(provider).count("configure"), 1)

    def test_every_ack_requires_both_teammate_fields_unchanged(self):
        changes = [("teammate_kind", "random"), ("teammate_account_id", 999),
                   ("teammate_kind", "missing"), ("teammate_account_id", "missing")]
        for command in ("configure", "next_episode", "step_joint"):
            for field, value in changes:
                with self.subTest(command=command, field=field, value=value):
                    def alter(request, response):
                        if request["command"] == command:
                            if value == "missing":
                                response.pop(field)
                            else:
                                response[field] = value
                        return response
                    client, provider, profile = self.setup_control(alter=alter)
                    with self.assertRaises(BridgeError):
                        self.start_episode(client, profile)
                        client.step_joint(self.joint_actions(profile), 0.1)
                    self.assert_released(client, provider)
                    self.assertEqual(self.commands(provider).count(command), 1)
                    if command != "step_joint":
                        self.assertNotIn("step_joint", self.commands(provider))

    def test_joint_step_refuses_stale_or_misattributed_transition_without_retry(self):
        changes = [("tick", 0), ("session_id", "different-session"),
                   ("episode_id", "different-match"), ("request_id", 999),
                   ("request_id", True), ("terminated", 0)]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                def alter(request, response):
                    if request["command"] == "step_joint":
                        response[field] = value
                    return response
                client, provider, profile = self.setup_control(alter=alter)
                self.start_episode(client, profile)
                with self.assertRaises(BridgeError):
                    client.step_joint(self.joint_actions(profile), 0.1)
                self.assert_released(client, provider)
                self.assertEqual(self.commands(provider).count("step_joint"), 1)

    def test_provider_error_releases_without_replaying_joint_action(self):
        def alter(request, response):
            if request["command"] == "step_joint":
                return {"request_id": request["request_id"], "error": "second input device disconnected"}
            return response
        client, provider, profile = self.setup_control(alter=alter)
        self.start_episode(client, profile)
        with self.assertRaisesRegex(BridgeError, "second input device disconnected"):
            client.step_joint(self.joint_actions(profile), 0.1)
        self.assert_released(client, provider)
        self.assertEqual(self.commands(provider).count("step_joint"), 1)

    def test_joint_timeout_releases_both_controls_without_replaying_request(self):
        def alter(request, response):
            return None if request["command"] == "step_joint" else response
        client, provider, profile = self.setup_control(alter=alter, timeout=0.04)
        self.start_episode(client, profile)
        with self.assertRaisesRegex(BridgeError, "timed out"):
            client.step_joint(self.joint_actions(profile), 0.1)
        self.assert_released(client, provider)
        self.assertEqual(self.commands(provider).count("step_joint"), 1)

    def test_new_match_timeout_or_stop_releases_without_replaying_transition(self):
        def waiting(request, response):
            return None if request["command"] == "next_episode" else response
        client, provider, profile = self.setup_control(alter=waiting, episode_timeout=0.04)
        client.configure(profile)
        with self.assertRaisesRegex(BridgeError, "timed out"):
            client.next_episode()
        self.assert_released(client, provider)
        self.assertEqual(self.commands(provider).count("next_episode"), 1)

        stop = threading.Event()
        def stopped(request, response):
            if request["command"] == "next_episode":
                stop.set()
                return None
            return response
        client, provider, profile = self.setup_control(alter=stopped, episode_timeout=30,
                                                     stop_requested=stop.is_set)
        client.configure(profile)
        started = time.monotonic()
        with self.assertRaises(KeyboardInterrupt):
            client.next_episode()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assert_released(client, provider)
        self.assertEqual(self.commands(provider).count("next_episode"), 1)


if __name__ == "__main__":
    unittest.main()
