"""Client for a project-defined offline-control bridge, not a built-in game API.

A separately implemented provider must synchronize actions, simulation ticks and
resets. It must release all inputs after each step and on release, disconnect or
watchdog timeout. This module never reads process memory or sends keyboard input.
"""

from dataclasses import asdict, dataclass
import json
import math
import socket
import time

import gymnasium as gym
import numpy as np

from controls import ACTIONS
from game_state import GameState, RewardTracker
from telemetry import PLAYER_KEY, validate_players


PROTOCOL_VERSION = 1
OBSERVATION_SCHEMA = "brawlhalla-players-v1"
MAX_RESPONSE_BYTES = 1024 * 1024


class BridgeError(RuntimeError):
    """The controlled bridge cannot provide a trustworthy transition."""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a positive finite number") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value {value}")


def _check_finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            _check_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _check_finite_json(child)


@dataclass(frozen=True)
class BridgeSnapshot:
    episode_id: str
    tick: int
    players: dict
    terminated: bool
    truncated: bool


class StateBridgeClient:
    """Bounded, synchronous JSONL TCP client restricted to loopback addresses.

    Requests are never replayed. A protocol or transport failure permanently
    closes this client, attempts a bounded input-release message and raises.
    A provider-side input watchdog is still required if the connection fails.
    """

    def __init__(self, host="127.0.0.1", port=8765, timeout=3.0,
                 expected_build=None, stop_requested=None):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("bridge host must be 127.0.0.1, localhost or ::1")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("bridge port must be an integer from 1 to 65535")
        self.timeout = _positive(timeout, "timeout")
        if expected_build is not None and (
                not isinstance(expected_build, str) or not expected_build.strip()):
            raise ValueError("expected_build must be a nonempty string or None")
        if stop_requested is not None and not callable(stop_requested):
            raise ValueError("stop_requested must be callable or None")
        # Avoid resolving an arbitrary hostname, including alternate localhost DNS.
        self.host = "127.0.0.1" if host == "localhost" else host
        self.port = port
        self.expected_build = expected_build
        self.stop_requested = stop_requested
        self.game_build = None
        self.controlled_player = None
        self.last_state = None
        self._socket = None
        self._request_id = 0
        self._closed = False

    def _check_stop(self):
        if self.stop_requested is not None and self.stop_requested():
            raise KeyboardInterrupt("controlled bridge stopped")

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
                raise BridgeError("hello: incompatible bridge protocol")
            if hello.get("mode") != "offline_controlled":
                raise BridgeError("hello: provider must declare mode offline_controlled")
            if hello.get("observation_schema") != OBSERVATION_SCHEMA:
                raise BridgeError("hello: incompatible observation_schema")
            build = hello.get("game_build")
            if not isinstance(build, str) or not build.strip():
                raise BridgeError("hello: missing nonempty game_build")
            if self.expected_build is not None and build != self.expected_build:
                raise BridgeError(f"hello: game build {build!r} does not match {self.expected_build!r}")
            capabilities = hello.get("capabilities")
            if (not isinstance(capabilities, list)
                    or not all(isinstance(item, str) for item in capabilities)
                    or not {"state", "actions", "reset"}.issubset(capabilities)):
                raise BridgeError("hello: provider requires state, actions and reset capabilities")
            controlled = hello.get("controlled_player")
            if not isinstance(controlled, str) or not PLAYER_KEY.fullmatch(controlled):
                raise BridgeError("hello: controlled_player must be an explicit Player<number> slot")
            self.game_build = build
            self.controlled_player = controlled
            return self
        except OSError as error:
            self.close()
            raise BridgeError(f"cannot connect to offline-control provider at {self.host}:{self.port}: {error}") from error
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def _send(self, data, deadline):
        offset = 0
        while offset < len(data):
            self._check_stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeError("bridge request timed out while sending")
            self._socket.settimeout(min(0.1, remaining))
            try:
                sent = self._socket.send(data[offset:])
            except socket.timeout:
                continue
            if sent == 0:
                raise BridgeError("bridge disconnected while sending")
            offset += sent

    def _read_line(self, deadline):
        data = bytearray()
        while True:
            self._check_stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeError("bridge response timed out")
            self._socket.settimeout(min(0.1, remaining))
            try:
                chunk = self._socket.recv(min(65536, MAX_RESPONSE_BYTES + 1 - len(data)))
            except socket.timeout:
                continue
            if not chunk:
                raise BridgeError("bridge disconnected before a complete response")
            data.extend(chunk)
            newline = data.find(b"\n")
            if len(data) > MAX_RESPONSE_BYTES:
                raise BridgeError("bridge response exceeds 1 MiB limit")
            if newline != -1:
                if newline != len(data) - 1:
                    raise BridgeError("bridge sent unsolicited data after its response")
                return bytes(data[:newline])

    def _request(self, command, **fields):
        if self._socket is None or self._closed:
            raise BridgeError("bridge is not connected")
        self._request_id += 1
        request_id = self._request_id
        payload = {"command": command, "request_id": request_id, **fields}
        try:
            deadline = time.monotonic() + self.timeout
            self._send((json.dumps(payload, allow_nan=False) + "\n").encode("utf-8"), deadline)
            raw = self._read_line(deadline)
            response = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object,
                                  parse_constant=_reject_constant)
            _check_finite_json(response)
            if not isinstance(response, dict):
                raise BridgeError(f"{command}: expected a JSON response object")
            if type(response.get("request_id")) is not int or response["request_id"] != request_id:
                raise BridgeError(f"{command}: response request_id does not match request")
            if "error" in response:
                detail = repr(response["error"])[:200]
                raise BridgeError(f"{command}: provider reported an error: {detail}")
            return response
        except (OSError, UnicodeError, ValueError, RecursionError) as error:
            self.close()
            raise BridgeError(f"{command}: invalid bridge response or transport failure: {error}") from error
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def _snapshot(self, response):
        episode = response.get("episode_id")
        if not isinstance(episode, str) or not episode.strip():
            raise BridgeError("state: missing nonempty episode_id")
        tick = response.get("tick")
        if type(tick) is not int or tick < 0:
            raise BridgeError("state: tick must be a nonnegative integer")
        for field in ("terminated", "truncated"):
            if type(response.get(field)) is not bool:
                raise BridgeError(f"state: {field} must be an explicit boolean")
        try:
            players = validate_players(response.get("players"))
        except ValueError as error:
            raise BridgeError(f"state: invalid players: {error}") from error
        return BridgeSnapshot(episode, tick, players, response["terminated"], response["truncated"])

    def reset(self, seed=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        self.connect()
        try:
            snapshot = self._snapshot(self._request("reset", seed=seed))
            if self.last_state is not None and snapshot.episode_id == self.last_state.episode_id:
                raise BridgeError("reset: provider must acknowledge a new episode_id")
            if snapshot.terminated or snapshot.truncated:
                raise BridgeError("reset: provider returned an already completed episode")
            self.last_state = snapshot
            return snapshot
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def step(self, action, hold_seconds):
        if not isinstance(action, (list, tuple)) or tuple(action) not in ACTIONS:
            raise ValueError("action must be one of the configured ACTIONS key combinations")
        hold_seconds = _positive(hold_seconds, "hold_seconds")
        if self.last_state is None:
            raise BridgeError("step requires a successful reset first")
        if self.last_state.terminated or self.last_state.truncated:
            raise BridgeError("episode completed; reset before stepping again")
        try:
            snapshot = self._snapshot(self._request("step", action=list(action), hold_seconds=hold_seconds))
            if snapshot.episode_id != self.last_state.episode_id:
                raise BridgeError("step: episode_id changed without a reset")
            if snapshot.tick <= self.last_state.tick:
                raise BridgeError("step: tick must strictly increase after each action")
            self.last_state = snapshot
            return snapshot
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def close(self):
        """Attempt a release without waiting for an acknowledgment, then close."""
        connection, self._socket = self._socket, None
        self._closed = True
        if connection is None:
            return
        try:
            self._request_id += 1
            connection.settimeout(min(self.timeout, 0.1))
            connection.sendall((json.dumps({"command": "release", "request_id": self._request_id})
                                + "\n").encode("utf-8"))
        except OSError:
            pass
        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass


class ControlledStateEnv(gym.Env):
    """Ten-feature PPO environment using only acknowledged bridge transitions."""

    metadata = {"render_modes": []}

    def __init__(self, client, self_slot="Player1", opponent_slot="Player2",
                 hold_seconds=0.1, position_scale=1000.0, damage_scale=100.0,
                 damage_reward_scale=0.01):
        super().__init__()
        for slot in (self_slot, opponent_slot):
            if not isinstance(slot, str) or not PLAYER_KEY.fullmatch(slot):
                raise ValueError("slots must be explicit Player<number> keys")
        if self_slot == opponent_slot:
            raise ValueError("self and opponent slots must be different")
        self.client = client
        self.self_slot, self.opponent_slot = self_slot, opponent_slot
        self.hold_seconds = _positive(hold_seconds, "hold_seconds")
        self.position_scale = _positive(position_scale, "position_scale")
        self.damage_scale = _positive(damage_scale, "damage_scale")
        self.tracker = RewardTracker(damage_scale=damage_reward_scale)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self._ready = False

    def _observe(self, snapshot, *, reset=False):
        try:
            state = GameState.from_players(snapshot.players, self.self_slot, self.opponent_slot)
            if reset and (state.self_player.eliminated or state.opponent.eliminated):
                raise ValueError("reset requires both selected players to be active")
            observation = state.to_vector(self.position_scale, self.damage_scale)
            projection = self.tracker.update(state)
            if any("_identity_changed" in item or "_slot_changed_to_" in item
                   for item in projection.diagnostics):
                raise ValueError("selected player identity changed within an episode")
        except ValueError as error:
            raise BridgeError(f"state cannot be used for training: {error}") from error
        info = {"episode_id": snapshot.episode_id, "tick": snapshot.tick,
                "game_build": self.client.game_build, "projection": asdict(projection)}
        return observation, projection.reward, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ready = False
        try:
            self.client.connect()
            if self.client.controlled_player != self.self_slot:
                raise BridgeError("hello: controlled_player does not match the selected self slot")
            snapshot = self.client.reset(seed=seed)
            self.tracker.reset()
            observation, _, info = self._observe(snapshot, reset=True)
            self._ready = True
            return observation, info
        except (BridgeError, KeyboardInterrupt):
            self.close()
            raise

    def step(self, action):
        if not self._ready:
            raise BridgeError("environment requires a successful reset before step")
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
