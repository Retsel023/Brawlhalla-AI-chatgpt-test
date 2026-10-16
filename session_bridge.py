"""Session-aware client contract for a separately implemented game provider.

Protocol 2 can describe offline and online matches and 1v1 or 2v2 roles. It does
not implement Windows game instrumentation, matchmaking, or keyboard control.
``next_episode`` asks the provider to wait for a new playable match; it never
means that a human opponent's match can be restarted by this client.
"""

import copy
from dataclasses import asdict, dataclass
import socket

import gymnasium as gym
import numpy as np

from controls import ACTIONS
from session_profiles import validate_profile
from state_bridge import BridgeError, BridgeSnapshot, StateBridgeClient, _positive
from team_state import TeamState, TeamRewardTracker
from telemetry import PLAYER_KEY


PROTOCOL_VERSION = 2
OBSERVATION_SCHEMA = "brawlhalla-team-v2"
PROFILE_FIELDS = ("profile", "match_format", "opponent_kind", "online", "self_player",
                  "teammate_player", "opponents", "friendly_fire")
REQUIRED_CAPABILITIES = {"state", "actions", "session", "next_episode"}


@dataclass(frozen=True)
class SessionSnapshot(BridgeSnapshot):
    session_id: str
    profile: dict

    @property
    def match_format(self):
        return self.profile["match_format"]


class SessionBridgeClient(StateBridgeClient):
    """Synchronous protocol-2 client reusing bounded protocol-1 transport.

    Only the wire transport is shared with the legacy client. Session setup and
    episode transitions require explicit acknowledgments and are never retried.
    This class is intended for one calling thread. ``episode_timeout`` bounds a
    provider's wait for a new match; stop requests remain polled every 0.1 second.
    """

    def __init__(self, host="127.0.0.1", port=8765, timeout=3.0,
                 expected_build=None, stop_requested=None, episode_timeout=30.0):
        super().__init__(host, port, timeout, expected_build, stop_requested)
        self.episode_timeout = _positive(episode_timeout, "episode_timeout")
        self.provider_mode = None
        self.session_id = None
        self._profile = None
        self._seen_episodes = set()

    @property
    def expected_profile(self):
        return copy.deepcopy(self._profile)

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
                raise BridgeError("hello: session profiles require bridge protocol 2")
            mode = hello.get("mode")
            if mode not in ("offline_controlled", "online_controlled"):
                raise BridgeError("hello: session provider must declare offline_controlled or online_controlled")
            if hello.get("observation_schema") != OBSERVATION_SCHEMA:
                raise BridgeError("hello: session profiles require brawlhalla-team-v2 observations")
            build = hello.get("game_build")
            if not isinstance(build, str) or not build.strip():
                raise BridgeError("hello: missing nonempty game_build")
            if self.expected_build is not None and build != self.expected_build:
                raise BridgeError("hello: provider game_build does not match expected_build")
            capabilities = hello.get("capabilities")
            if (not isinstance(capabilities, list)
                    or not all(isinstance(item, str) for item in capabilities)
                    or not REQUIRED_CAPABILITIES.issubset(capabilities)):
                raise BridgeError("hello: provider requires state, actions, session and next_episode capabilities")
            controlled = hello.get("controlled_player")
            if not isinstance(controlled, str) or not PLAYER_KEY.fullmatch(controlled):
                raise BridgeError("hello: controlled_player must be an explicit Player<number> slot")
            self.provider_mode = mode
            self.game_build = build
            self.controlled_player = controlled
            return self
        except OSError as error:
            self.close()
            raise BridgeError(f"cannot connect to session provider at {self.host}:{self.port}: {error}") from error
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def _acknowledged_profile(self, response):
        try:
            actual = validate_profile({field: response[field] for field in PROFILE_FIELDS})
        except (KeyError, ValueError) as error:
            raise BridgeError(f"provider returned an invalid or incomplete session profile: {error}") from error
        if actual != self._profile:
            raise BridgeError("provider session profile or player/team assignments do not match the request")
        return actual

    def configure(self, profile):
        requested = validate_profile(profile)
        if self._closed:
            raise BridgeError("bridge is closed; create a new client to reconnect")
        if self._profile is not None:
            if requested == self._profile:
                return self.session_id
            self.close()
            raise BridgeError("a configured client cannot switch session profiles; create a new client")
        self.connect()
        try:
            expected_mode = "online_controlled" if requested["online"] else "offline_controlled"
            if self.provider_mode != expected_mode:
                raise BridgeError("hello: provider mode does not support the requested online/offline session")
            if self.controlled_player != requested["self_player"]:
                raise BridgeError("hello: controlled_player does not match the selected self_player")
            self._profile = copy.deepcopy(requested)
            response = self._request("configure", **requested)
            self._acknowledged_profile(response)
            session_id = response.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                raise BridgeError("configure: provider must acknowledge a nonempty session_id")
            self.session_id = session_id
            return session_id
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def _snapshot(self, response):
        if self._profile is None or self.session_id is None:
            raise BridgeError("state requires an acknowledged session configuration")
        state = super()._snapshot(response)
        if response.get("session_id") != self.session_id:
            raise BridgeError("state: session_id does not match the configured session")
        profile = self._acknowledged_profile(response)
        selected = {profile["self_player"], *profile["opponents"]}
        if profile["teammate_player"] is not None:
            selected.add(profile["teammate_player"])
        if set(state.players) != selected:
            raise BridgeError("state: player roster must exactly match the configured player/team assignments")
        return SessionSnapshot(state.episode_id, state.tick, state.players,
                               state.terminated, state.truncated, self.session_id, profile)

    def next_episode(self, seed=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        if self._profile is None or self.session_id is None:
            raise BridgeError("next_episode requires a successful configure first")
        after = self.last_state.episode_id if self.last_state is not None else None
        request_timeout = self.timeout
        try:
            self.timeout = self.episode_timeout
            response = self._request("next_episode", session_id=self.session_id,
                                     after_episode_id=after, seed=seed, **self._profile)
            snapshot = self._snapshot(response)
            if snapshot.episode_id in self._seen_episodes:
                raise BridgeError("next_episode: provider must return a globally new episode_id within this session")
            if snapshot.terminated or snapshot.truncated:
                raise BridgeError("next_episode: provider returned an already completed match")
            if any(player["Eliminated"] for player in snapshot.players.values()):
                raise BridgeError("next_episode: every selected player must be active in the new playable match")
            self._seen_episodes.add(snapshot.episode_id)
            self.last_state = snapshot
            return snapshot
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise
        finally:
            self.timeout = request_timeout

    def reset(self, seed=None):
        raise BridgeError("session clients use next_episode; legacy reset cannot restart these matches")


class SessionEnv(gym.Env):
    """Fixed 26-feature environment for acknowledged 1v1 and 2v2 sessions."""

    metadata = {"render_modes": []}

    def __init__(self, client, profile, hold_seconds=0.1,
                 position_scale=1000.0, damage_scale=100.0, damage_reward_scale=0.01):
        super().__init__()
        self.client = client
        self.profile = validate_profile(profile)
        self.hold_seconds = _positive(hold_seconds, "hold_seconds")
        self.position_scale = _positive(position_scale, "position_scale")
        self.damage_scale = _positive(damage_scale, "damage_scale")
        self.tracker = TeamRewardTracker(damage_reward_scale=damage_reward_scale)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32)
        self._ready = False

    def _observe(self, snapshot):
        try:
            state = TeamState.from_players(
                snapshot.players, self.profile["self_player"], self.profile["opponents"],
                teammate_slot=self.profile["teammate_player"],
                friendly_fire=self.profile["friendly_fire"])
            observation = state.to_vector(self.position_scale, self.damage_scale)
            projection = self.tracker.update(state)
            if any("_identity_changed" in diagnostic or "_slot_changed_to_" in diagnostic
                   or diagnostic in ("format_changed", "friendly_fire_changed")
                   for diagnostic in projection.diagnostics):
                raise ValueError("player identity or team assignment changed within the current match")
        except ValueError as error:
            raise BridgeError(f"session state cannot be used for training: {error}") from error
        info = {"session_id": snapshot.session_id, "episode_id": snapshot.episode_id,
                "tick": snapshot.tick, "game_build": self.client.game_build,
                "profile": copy.deepcopy(snapshot.profile), "projection": asdict(projection)}
        return observation, projection.reward, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ready = False
        try:
            self.client.configure(self.profile)
            snapshot = self.client.next_episode(seed=seed)
            self.tracker.reset()
            observation, _, info = self._observe(snapshot)
            self._ready = True
            return observation, info
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def step(self, action):
        if not self._ready:
            raise BridgeError("session environment requires a successful next-episode reset")
        if not self.action_space.contains(action):
            raise ValueError("action is outside the discrete action space")
        try:
            snapshot = self.client.step(ACTIONS[int(action)], self.hold_seconds)
            observation, reward, info = self._observe(snapshot)
            self._ready = not (snapshot.terminated or snapshot.truncated)
            return observation, reward, snapshot.terminated, snapshot.truncated, info
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def close(self):
        self._ready = False
        self.client.close()
