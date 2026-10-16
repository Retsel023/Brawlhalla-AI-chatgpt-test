"""Two policy perspectives of one synchronized, provider-controlled 2v2 world.

This VecEnv exposes two samples per world transition, not two independent games.
The provider must apply both actors' actions together and advance the world once.
Each sample receives the same team damage objective; episode counts must count
the shared match once. History stacking belongs in an outer VecFrameStack.
"""

from contextlib import suppress
import copy
from dataclasses import asdict
import math
from numbers import Integral

from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from controls import ACTIONS
from session_profiles import validate_profile
from state_bridge import BridgeError, _positive
from team_state import FEATURE_COUNT, TeamRewardTracker, TeamState


class DualTeamVecEnv(VecEnv):
    """Run one joint action for two AI teammates using a shared PPO policy."""

    def __init__(self, client, profile, hold_seconds=0.1,
                 position_scale=1000.0, damage_scale=100.0, damage_reward_scale=0.01):
        self.client = client
        self.profile = validate_profile(profile)
        if self.profile['match_format'] != '2v2' or self.profile['teammate_kind'] != 'ai':
            raise ValueError('DualTeamVecEnv requires a 2v2 profile with teammate_kind=ai')
        self.controlled_players = (self.profile['self_player'], self.profile['teammate_player'])
        self.hold_seconds = _positive(hold_seconds, 'hold_seconds')
        self.position_scale = _positive(position_scale, 'position_scale')
        self.damage_scale = _positive(damage_scale, 'damage_scale')
        self.trackers = tuple(TeamRewardTracker(damage_reward_scale) for _ in range(2))
        self.render_mode = None
        self._closed = False
        self._ready = False
        self._configured = False
        self._pending_actions = None
        self._session_id = None
        self._episode_id = None
        self._tick = None
        self._seen_episodes = set()
        super().__init__(2, spaces.Box(-np.inf, np.inf, shape=(FEATURE_COUNT,), dtype=np.float32),
                         spaces.Discrete(len(ACTIONS)))

    def _ensure_open(self):
        if self._closed:
            raise BridgeError('dual team environment is closed')

    def _validate_snapshot(self, snapshot, *, reset):
        try:
            if validate_profile(snapshot.profile) != self.profile:
                raise ValueError('provider profile or player/team assignments changed')
            selected = {*self.controlled_players, *self.profile['opponents']}
            if not isinstance(snapshot.players, dict) or set(snapshot.players) != selected:
                raise ValueError('player roster must match all four selected slots exactly')
            for name in ('session_id', 'episode_id'):
                value = getattr(snapshot, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'{name} must be a nonempty string')
            if self._session_id is not None and snapshot.session_id != self._session_id:
                raise ValueError('session_id changed within the configured shared world')
            if type(snapshot.tick) is not int or snapshot.tick < 0:
                raise ValueError('tick must be a nonnegative integer')
            if type(snapshot.terminated) is not bool or type(snapshot.truncated) is not bool:
                raise ValueError('episode termination flags must be explicit booleans')
            if reset:
                if snapshot.episode_id in self._seen_episodes:
                    raise ValueError('next_episode must acknowledge a new episode_id')
                if snapshot.terminated or snapshot.truncated:
                    raise ValueError('next_episode returned a completed match')
            elif snapshot.episode_id != self._episode_id or snapshot.tick <= self._tick:
                raise ValueError('joint step must keep the episode and strictly advance its tick')
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise BridgeError(f'invalid shared world snapshot: {error}') from error

    def _observe(self, snapshot, *, reset=False):
        self._validate_snapshot(snapshot, reset=reset)
        observations, rewards, infos = [], [], []
        try:
            for index, own in enumerate(self.controlled_players):
                ally = self.controlled_players[1 - index]
                state = TeamState.from_players(
                    snapshot.players, own, self.profile['opponents'],
                    teammate_slot=ally, friendly_fire=self.profile['friendly_fire'])
                if reset and any(player.eliminated for player in state.selected_players):
                    raise ValueError('every selected player must be active in a new match')
                observation = state.to_vector(self.position_scale, self.damage_scale)
                projection = self.trackers[index].update(state)
                if any('_identity_changed' in item or '_slot_changed_to_' in item
                       or item in ('format_changed', 'friendly_fire_changed')
                       for item in projection.diagnostics):
                    raise ValueError('player identity or team assignment changed within the match')
                if not math.isfinite(projection.reward) or abs(projection.reward) > float(np.finfo(np.float32).max):
                    raise ValueError('team reward cannot be represented as finite float32')
                observations.append(observation)
                rewards.append(projection.reward)
                infos.append({
                    'session_id': snapshot.session_id, 'episode_id': snapshot.episode_id,
                    'tick': snapshot.tick, 'game_build': self.client.game_build,
                    'profile': copy.deepcopy(self.profile), 'controlled_player': own,
                    'teammate_player': ally, 'shared_world': True,
                    'projection': asdict(projection),
                })
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise BridgeError(f'shared state cannot be used for training: {error}') from error
        self._session_id, self._episode_id, self._tick = snapshot.session_id, snapshot.episode_id, snapshot.tick
        return np.stack(observations), np.asarray(rewards, dtype=np.float32), infos

    def _next_episode(self):
        snapshot = self.client.next_episode(seed=self._seeds[0])
        for tracker in self.trackers:
            tracker.reset()
        observations, _, infos = self._observe(snapshot, reset=True)
        self._seen_episodes.add(snapshot.episode_id)
        self.reset_infos = infos
        self._reset_seeds()
        self._reset_options()
        self._ready = True
        return observations

    def reset(self):
        """Reset both perspectives by requesting one playable shared match."""
        try:
            self._ensure_open()
            self._ready = False
            self._pending_actions = None
            if not self._configured:
                for name in ('validate_control', 'configure', 'next_episode', 'step_joint'):
                    if not callable(getattr(self.client, name, None)):
                        raise BridgeError('dual AI teammates require a joint-control provider client')
                session_id = self.client.configure(self.profile)
                if not isinstance(session_id, str) or not session_id.strip():
                    raise BridgeError('provider must acknowledge a nonempty shared session_id')
                declared = self.client.controlled_players
                if (not isinstance(declared, (list, tuple)) or len(declared) != 2
                        or set(declared) != set(self.controlled_players)):
                    raise BridgeError('provider must acknowledge control of both AI teammate slots exactly')
                self._session_id = session_id
                self._configured = True
            return self._next_episode()
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def step_async(self, actions):
        """Cache both decisions; no world action occurs until step_wait."""
        try:
            self._ensure_open()
            if not self._ready:
                raise BridgeError('both actors require a successful shared reset before stepping')
            if self._pending_actions is not None:
                raise BridgeError('a joint action is already pending; call step_wait first')
            values = np.asarray(actions)
            if values.shape != (2,) or values.dtype.kind not in ('i', 'u'):
                raise ValueError('joint step requires exactly two integer action indices')
            if isinstance(actions, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in actions):
                raise ValueError('boolean values are not integer action indices')
            if any(not self.action_space.contains(action) for action in values):
                raise ValueError('joint action is outside the discrete action space')
            self._pending_actions = tuple(int(action) for action in values)
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def step_wait(self):
        try:
            self._ensure_open()
            if self._pending_actions is None:
                raise BridgeError('step_wait requires one pending pair from step_async')
            actions, self._pending_actions = self._pending_actions, None
            snapshot = self.client.step_joint(
                {slot: ACTIONS[action] for slot, action in zip(self.controlled_players, actions)},
                self.hold_seconds)
            observations, rewards, infos = self._observe(snapshot)
            done = snapshot.terminated or snapshot.truncated
            dones = np.full(2, done, dtype=bool)
            for info in infos:
                info['TimeLimit.truncated'] = snapshot.truncated and not snapshot.terminated
            if done:
                self._ready = False
                for index, info in enumerate(infos):
                    info['terminal_observation'] = observations[index].copy()
                observations = self._next_episode()
            return observations, rewards, dones, infos
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._ready = False
        self._pending_actions = None
        self.client.close()

    def seed(self, seed=None):
        """Both perspectives share one world's seed rather than two seeds."""
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError('seed must be a nonnegative integer or None')
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._seeds = [seed, seed]
        return list(self._seeds)

    def set_options(self, options=None):
        if options is None or options == {} or options == [{}, {}]:
            self._options = [{}, {}]
            return
        raise ValueError('shared-world reset options are configured by the session profile')

    def _indices(self, indices):
        if indices is None:
            return [0, 1]
        selected = [indices] if isinstance(indices, Integral) else list(indices)
        if any(isinstance(index, bool) or not isinstance(index, Integral)
               or not 0 <= index < 2 for index in selected):
            raise IndexError('actor indices must be 0 or 1')
        return [int(index) for index in selected]

    def get_attr(self, attr_name, indices=None):
        selected = self._indices(indices)
        if attr_name in ('controlled_player', 'self_player'):
            return [self.controlled_players[index] for index in selected]
        if attr_name == 'teammate_player':
            return [self.controlled_players[1 - index] for index in selected]
        if attr_name.startswith('_') or attr_name in ('client', 'trackers'):
            raise AttributeError(attr_name)
        value = getattr(self, attr_name)
        if callable(value):
            raise AttributeError(attr_name)
        return [copy.deepcopy(value) for _ in selected]

    def set_attr(self, attr_name, value, indices=None):
        if sorted(self._indices(indices)) != [0, 1]:
            raise ValueError('attributes cannot change only one perspective of a shared world')
        if self._pending_actions is not None:
            raise ValueError('attributes cannot change while a joint action is pending')
        if attr_name == 'hold_seconds':
            self.hold_seconds = _positive(value, 'hold_seconds')
        elif attr_name == 'render_mode' and value is None:
            return
        else:
            raise AttributeError('shared roles and observation settings are immutable; create a new environment')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        selected = self._indices(indices)
        if method_name == 'render' and not method_args and not method_kwargs:
            return [None for _ in selected]
        if method_name == 'close' and sorted(selected) == [0, 1] and not method_args and not method_kwargs:
            self.close()
            return [None, None]
        raise AttributeError('use the whole VecEnv API; per-actor world operations are unsupported')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._indices(indices)]

    def get_images(self):
        return [None, None]
