"""Protocol-3 client for verified teammate setup and atomic dual-player actions.

The game-side provider remains a separate Windows component. It must establish
the requested team, route each controlled slot independently and release inputs
after a step, on release/disconnect and on its own watchdog timeout. Two AI
players advance one shared world through one joint request, never two steps.
"""

from collections.abc import Mapping
import socket

from controls import ACTIONS
from session_bridge import OBSERVATION_SCHEMA, SessionBridgeClient
from session_profiles import controlled_slots, validate_profile
from state_bridge import BridgeError, _positive
from telemetry import PLAYER_KEY


PROTOCOL_VERSION = 3
REQUIRED_CAPABILITIES = {"state", "actions", "session", "next_episode", "team_setup"}


class TeamControlClient(SessionBridgeClient):
    """Session client requiring explicit teammate and controlled-player roles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlled_players = []
        self.capabilities = set()
        self.input_routes = {}
        self._declared_primary = None
        self._regular_teammate_account_id = None

    @property
    def regular_teammate_account_id(self):
        """The configured or first-observed regular teammate ID, if established."""
        return self._regular_teammate_account_id

    def connect(self):
        if self._closed:
            raise BridgeError("bridge is closed; create a new client to reconnect")
        if self._socket is not None:
            return self
        try:
            self._check_stop()
            self._socket = socket.create_connection((self.host, self.port), self.timeout)
            hello = self._request("hello", protocol=PROTOCOL_VERSION,
                                  observation_schema=OBSERVATION_SCHEMA)
            if type(hello.get("protocol")) is not int or hello["protocol"] != PROTOCOL_VERSION:
                raise BridgeError("hello: teammate selection requires bridge protocol 3")
            if hello.get("observation_schema") != OBSERVATION_SCHEMA:
                raise BridgeError("hello: teammate selection requires brawlhalla-team-v2 observations")
            mode = hello.get("mode")
            if mode not in ("offline_controlled", "online_controlled"):
                raise BridgeError("hello: expected offline_controlled or online_controlled provider mode")
            build = hello.get("game_build")
            if not isinstance(build, str) or not build.strip():
                raise BridgeError("hello: missing nonempty game_build")
            if self.expected_build is not None and build != self.expected_build:
                raise BridgeError("hello: provider game_build does not match expected_build")
            capabilities = hello.get("capabilities")
            if (not isinstance(capabilities, list)
                    or not all(isinstance(item, str) for item in capabilities)
                    or not REQUIRED_CAPABILITIES.issubset(capabilities)):
                raise BridgeError("hello: provider requires state, actions, session, next_episode and team_setup capabilities")
            slots = hello.get("controlled_players")
            if (not isinstance(slots, list) or not slots
                    or not all(isinstance(slot, str) and PLAYER_KEY.fullmatch(slot) for slot in slots)
                    or len(set(slots)) != len(slots)):
                raise BridgeError("hello: controlled_players must list unique explicit Player<number> slots")
            primary = hello.get("controlled_player")
            if "controlled_player" in hello and (
                    not isinstance(primary, str) or primary not in slots):
                raise BridgeError("hello: optional controlled_player must identify a declared controlled slot")
            routes = hello.get("input_routes", {})
            if "input_routes" in hello and (
                    not isinstance(routes, dict) or set(routes) != set(slots)
                    or not all(isinstance(route, str) and route.strip() for route in routes.values())
                    or len({route.strip() for route in routes.values()}) != len(slots)):
                raise BridgeError("hello: input_routes must map every controlled slot to a distinct nonempty input route")
            self.provider_mode = mode
            self.game_build = build
            self.capabilities = set(capabilities)
            self.controlled_players = list(slots)
            self.input_routes = dict(routes)
            self._declared_primary = primary
            self.controlled_player = primary if primary is not None else slots[0]
            return self
        except OSError as error:
            self.close()
            raise BridgeError(f"cannot connect to team-control provider at {self.host}:{self.port}: {error}") from error
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def validate_control(self, profile):
        """Validate hello-only support without configuring a match or acting."""
        requested = validate_profile(profile)
        self.connect()
        try:
            expected_mode = "online_controlled" if requested["online"] else "offline_controlled"
            if self.provider_mode != expected_mode:
                raise BridgeError("hello: provider mode does not support the requested online/offline session")
            expected_slots = controlled_slots(requested)
            if set(self.controlled_players) != set(expected_slots):
                raise BridgeError("hello: controlled_players must exactly match the requested AI player roles")
            if self._declared_primary is not None and self._declared_primary != requested["self_player"]:
                raise BridgeError("hello: controlled_player does not match the requested primary AI player")
            if requested["teammate_kind"] == "ai" and "joint_actions" not in self.capabilities:
                raise BridgeError("hello: controlling both teammates requires the joint_actions capability")
            if requested["teammate_kind"] == "ai" and (
                    "independent_inputs" not in self.capabilities
                    or set(self.input_routes) != set(expected_slots)):
                raise BridgeError("hello: both AI teammates require independent_inputs and distinct input_routes")
            self.controlled_player = requested["self_player"]
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def configure(self, profile):
        requested = validate_profile(profile)
        self.validate_control(requested)
        was_configured = self.session_id is not None
        session_id = super().configure(requested)
        if not was_configured and requested["teammate_kind"] == "regular":
            self._regular_teammate_account_id = requested["teammate_account_id"]
        return session_id

    def _acknowledged_profile(self, response):
        # Protocol 3 requires explicit echoes of the new teammate fields. The
        # legacy validator's defaults cannot stand in for a provider's consent.
        try:
            actual = validate_profile({field: response[field] for field in self._profile})
        except (KeyError, ValueError) as error:
            raise BridgeError(f"provider returned an invalid or incomplete team-control profile: {error}") from error
        if actual != self._profile:
            raise BridgeError("provider teammate setup, team assignments or session profile does not match the request")
        return actual

    def _snapshot(self, response):
        snapshot = super()._snapshot(response)
        if self._profile["teammate_kind"] == "regular":
            teammate = snapshot.players[self._profile["teammate_player"]]
            account_id = teammate.get("BrawlhallaID")
            if account_id is None:
                raise BridgeError("state: a regular teammate requires a nonmissing BrawlhallaID")
            if self._regular_teammate_account_id is None:
                self._regular_teammate_account_id = account_id
            elif account_id != self._regular_teammate_account_id:
                raise BridgeError("state: regular teammate BrawlhallaID changed or differs from the requested account")
        return snapshot

    def step(self, action, hold_seconds):
        if self._profile is None or self.session_id is None:
            raise BridgeError("step requires an acknowledged team-control configuration")
        if len(controlled_slots(self._profile)) != 1:
            raise BridgeError("both AI teammates must use one atomic step_joint request")
        return super().step(action, hold_seconds)

    def step_joint(self, actions, hold_seconds):
        """Apply every controlled actor's action in one acknowledged world step."""
        if self._profile is None or self.session_id is None:
            raise BridgeError("step_joint requires an acknowledged team-control configuration")
        if self._profile["teammate_kind"] != "ai":
            raise BridgeError("step_joint requires an AI-controlled teammate profile")
        expected = controlled_slots(self._profile)
        if not isinstance(actions, Mapping) or set(actions) != set(expected):
            raise ValueError("joint actions must contain exactly the configured controlled player slots")
        payload = {}
        for slot in expected:
            action = actions[slot]
            if not isinstance(action, (list, tuple)) or tuple(action) not in ACTIONS:
                raise ValueError(f"joint action for {slot} must be a configured ACTIONS key combination")
            payload[slot] = list(action)
        hold_seconds = _positive(hold_seconds, "hold_seconds")
        if self.last_state is None:
            raise BridgeError("step_joint requires a successful next_episode first")
        if self.last_state.terminated or self.last_state.truncated:
            raise BridgeError("match completed; request next_episode before stepping again")
        try:
            snapshot = self._snapshot(self._request("step_joint", actions=payload,
                                                     hold_seconds=hold_seconds))
            if snapshot.episode_id != self.last_state.episode_id:
                raise BridgeError("step_joint: episode_id changed without a next_episode transition")
            if snapshot.tick <= self.last_state.tick:
                raise BridgeError("step_joint: tick must strictly increase after each atomic world action")
            self.last_state = snapshot
            return snapshot
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise
