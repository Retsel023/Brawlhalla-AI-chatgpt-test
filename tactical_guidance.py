"""Goal observations and bounded shaping, without replacing policy actions.

Spacing targets use the export's coordinate units and are approximate training
preferences. Input drills are exact consecutive actions, not verified Brawlhalla
combos. Observed damage does not prove that the drill caused a hit. Attribution
is ambiguous in 2v2, so team observations never receive a drill bonus.
"""

import copy
import json
import logging
import math
from pathlib import Path

from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from controls import ACTIONS
from guidance_settings import (
    ACTION_LABELS, GUIDANCE_FEATURE_NAMES, GUIDANCE_SCHEMA, OPPONENT_SPACING,
    TEAMMATE_SPACING, validate_guidance,
)
from state_bridge import _positive
from team_state import FEATURE_COUNT


SPACING_TARGETS = {'close': 150.0, 'mid': 350.0, 'far': 650.0,
                   'near': 200.0, 'spread': 600.0}
POTENTIAL_SCALE = 0.01
SHAPING_GAMMA = 0.99
COMBO_BONUS_LIMIT = 0.02
LOGGER = logging.getLogger(__name__)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate guidance key: {key}')
        result[key] = value
    return result


class TacticalGuidanceVecEnv(VecEnvWrapper):
    """Append 106 goal features after an unchanged stacked team-state prefix.

    Live files are read at reset and after a completed transition. A transition
    always uses the goal encoded in its preceding observation. A newly accepted
    goal resets drill progress and is encoded in the next observation. Learned
    policies may respond to goals; evaluation adds no training reward bonuses.
    """

    def __init__(self, venv, guidance=None, *, guidance_file=None,
                 position_scale=1000.0, learning=True, shaping_gamma=SHAPING_GAMMA):
        base = venv.observation_space
        if (not isinstance(base, spaces.Box) or len(base.shape) != 1
                or base.dtype != np.float32 or base.shape[0] < FEATURE_COUNT
                or base.shape[0] % FEATURE_COUNT):
            raise ValueError('guidance must wrap flat float32 team observations after frame stacking')
        if (not isinstance(venv.action_space, spaces.Discrete)
                or venv.action_space.n != len(ACTIONS) or venv.action_space.start != 0
                or len(ACTION_LABELS) != len(ACTIONS)):
            raise ValueError('guidance requires the existing Brawlhalla discrete action mapping')
        if type(learning) is not bool:
            raise ValueError('learning must be a boolean')
        self.position_scale = _positive(position_scale, 'position_scale')
        self.learning = learning
        self.shaping_gamma = shaping_gamma
        self.guidance_file = Path(guidance_file) if guidance_file is not None else None
        self._guidance = validate_guidance({} if guidance is None else guidance)
        self._version = 0
        self._file_warning = None
        self._progress = [0] * venv.num_envs
        self._previous_potential = [None] * venv.num_envs
        self._previous_episode = [None] * venv.num_envs
        self._actions = None
        self._ready = False
        self.base_feature_count = base.shape[0]
        low = np.concatenate((base.low, np.zeros(len(GUIDANCE_FEATURE_NAMES), np.float32)))
        high = np.concatenate((base.high, np.ones(len(GUIDANCE_FEATURE_NAMES), np.float32)))
        super().__init__(venv, observation_space=spaces.Box(low, high, dtype=np.float32))

    @property
    def guidance(self):
        return copy.deepcopy(self._guidance)

    @property
    def shaping_gamma(self):
        return self._shaping_gamma

    @shaping_gamma.setter
    def shaping_gamma(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('shaping_gamma must be a finite number between 0 and 1')
        try:
            result = float(value)
        except (ValueError, OverflowError):
            raise ValueError('shaping_gamma must be a finite number between 0 and 1') from None
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError('shaping_gamma must be a finite number between 0 and 1')
        self._shaping_gamma = result

    def _poll_guidance(self):
        if self.guidance_file is None:
            return False
        try:
            if self.guidance_file.stat().st_size > 16384:
                raise ValueError('guidance file exceeds 16 KiB')
            raw = self.guidance_file.read_bytes()
            if len(raw) > 16384:
                raise ValueError('guidance file exceeds 16 KiB')
            proposed = validate_guidance(json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object))
        except (OSError, UnicodeError, ValueError, RecursionError) as error:
            warning = f'Guidance update ignored: {error}'
            if warning != self._file_warning:
                LOGGER.warning('%s', warning)
            self._file_warning = warning
            return False
        self._file_warning = None
        if proposed == self._guidance:
            return False
        self._guidance = proposed
        self._version += 1
        self._progress = [0] * self.num_envs
        self._previous_potential = [None] * self.num_envs
        return True

    def _goal_vector(self, guidance, progress):
        result = np.zeros(len(GUIDANCE_FEATURE_NAMES), np.float32)
        result[OPPONENT_SPACING.index(guidance['opponent_spacing'])] = 1
        result[4 + TEAMMATE_SPACING.index(guidance['teammate_spacing'])] = 1
        result[7] = guidance['strength']
        for index, label in enumerate(guidance['combo']):
            result[8 + index * len(ACTION_LABELS) + ACTION_LABELS.index(label)] = 1
        result[-2] = len(guidance['combo']) / 6
        result[-1] = progress / 6
        return result

    def _augment(self, observations):
        base = np.asarray(observations)
        if base.shape != (self.num_envs, self.base_feature_count) or base.dtype != np.float32 or not np.isfinite(base).all():
            raise ValueError('provider returned malformed stacked team observations')
        goals = np.stack([self._goal_vector(self._guidance, progress) for progress in self._progress])
        return np.concatenate((base, goals), axis=1)

    def _latest(self, observation):
        return np.asarray(observation)[-FEATURE_COUNT:]

    @staticmethod
    def _eliminated(frame):
        return bool(frame[3] or frame[7] or (frame[16] and frame[13]) or (frame[23] and frame[20]))

    def _potential(self, frame, guidance):
        if self._eliminated(frame):
            return None
        scores = []
        opponent = guidance['opponent_spacing']
        if opponent != 'free':
            distances = [math.hypot(float(frame[8]), float(frame[9]))]
            if frame[23]:
                distances.append(math.hypot(float(frame[21]), float(frame[22])))
            scores.append(self._spacing_score(min(distances), opponent))
        teammate = guidance['teammate_spacing']
        if teammate != 'free' and frame[16]:
            scores.append(self._spacing_score(math.hypot(float(frame[14]), float(frame[15])), teammate))
        if not scores:
            return None
        return POTENTIAL_SCALE * guidance['strength'] * math.fsum(scores) / len(scores)

    def _spacing_score(self, normalized_distance, target):
        target_distance = SPACING_TARGETS[target] / self.position_scale
        tolerance = 200.0 / self.position_scale
        return math.exp(-abs(normalized_distance - target_distance) / tolerance)

    def _advance_combo(self, index, action, guidance):
        combo = guidance['combo']
        if not combo:
            self._progress[index] = 0
            return False
        label = ACTION_LABELS[action]
        previous = self._progress[index]
        matched = combo[:previous] + [label]
        completed = len(matched) >= len(combo) and matched[-len(combo):] == combo
        # Preserve a repeated prefix: light/light/light/heavy includes the
        # light/light/heavy drill in its final three consecutive actions.
        longest = min(len(combo) - 1, len(matched))
        self._progress[index] = next((size for size in range(longest, 0, -1)
                                      if matched[-size:] == combo[:size]), 0)
        return completed

    def reset(self):
        self._poll_guidance()
        observations = self.venv.reset()
        self._actions = None
        self._progress = [0] * self.num_envs
        self._previous_episode = [None] * self.num_envs
        self._previous_potential = [self._potential(self._latest(row), self._guidance) for row in observations]
        self._ready = True
        return self._augment(observations)

    def step_async(self, actions):
        if not self._ready or self._actions is not None:
            raise RuntimeError('guidance requires reset and exactly one step_async per step_wait')
        values = np.asarray(actions)
        if values.shape != (self.num_envs,) or values.dtype.kind not in ('i', 'u') or any(
                not self.action_space.contains(action) for action in values):
            raise ValueError('guidance requires one valid integer action per actor')
        self._actions = values.copy()
        # Pass the sampled actions through unchanged so PPO trains on executed actions.
        self.venv.step_async(actions)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError('guidance step_wait requires pending policy actions')
        actions, self._actions = self._actions, None
        observations, rewards, dones, infos = self.venv.step_wait()
        result_rewards = np.asarray(rewards, dtype=np.float32).copy()
        infos = copy.deepcopy(infos)
        active, active_version = self.guidance, self._version
        progress_after = []
        for index, (row, done, info) in enumerate(zip(observations, dones, infos)):
            reset_frame = self._latest(row)
            terminal = None
            if done and 'terminal_observation' in info:
                terminal = np.asarray(info['terminal_observation'])
                if (terminal.shape != (self.base_feature_count,) or terminal.dtype != np.float32
                        or not np.isfinite(terminal).all()):
                    raise ValueError('malformed terminal observation before guidance suffix')
            frame = self._latest(terminal) if terminal is not None else reset_frame
            projection = info.get('projection')
            diagnostics = []
            episode = (info.get('session_id'), info.get('episode_id'))
            previous_episode = self._previous_episode[index]
            episode_changed = previous_episode is not None and episode != previous_episode
            baseline_reset = (not isinstance(projection, dict) or projection.get('discontinuity', False)
                              or projection.get('baseline_reset', False) or episode_changed
                              or (done and terminal is None))
            if not isinstance(projection, dict):
                diagnostics.append('Missing damage projection; guidance bonuses disabled for this transition.')
            if baseline_reset or self._eliminated(frame):
                self._progress[index] = 0
                self._previous_potential[index] = None
                diagnostics.append('Guidance baseline reset after a discontinuity or inactive player.')
            position_bonus = combo_bonus = 0.0
            potential = self._potential(frame, active)
            completed = False
            if not baseline_reset and not self._eliminated(frame):
                completed = self._advance_combo(index, int(actions[index]), active)
                if self.learning and self._previous_potential[index] is not None and potential is not None:
                    # True episode ends enter an absorbing state with potential zero.
                    # Time-limit truncation bootstraps from the actual terminal state.
                    target = 0.0 if done and not info.get('TimeLimit.truncated', False) else potential
                    position_bonus = self.shaping_gamma * target - self._previous_potential[index]
                if active['combo'] and frame[24]:
                    diagnostics.append('Input-drill bonuses are disabled in 2v2 because damage cannot be attributed to one actor.')
                elif self.learning and not done and completed:
                    damage = projection.get('damage_dealt', 0)
                    if not isinstance(damage, bool) and isinstance(damage, (int, float)) and math.isfinite(damage) and damage > 0:
                        combo_bonus = COMBO_BONUS_LIMIT * active['strength'] * min(damage / 10.0, 1.0)
            progress_after.append(self._progress[index])
            if terminal is not None:
                info['terminal_observation'] = np.concatenate((terminal, self._goal_vector(active, self._progress[index])))
            if done:
                self._progress[index] = 0
                self._previous_episode[index] = None
                diagnostics.append('Guidance progress reset at the provider episode boundary.')
            else:
                self._previous_episode[index] = episode
            self._previous_potential[index] = self._potential(reset_frame, active) if done else potential
            if not done and self._eliminated(frame):
                self._previous_potential[index] = None
            result_rewards[index] += position_bonus + combo_bonus
            info['guidance'] = {
                'schema': GUIDANCE_SCHEMA, 'active': copy.deepcopy(active),
                'active_version': active_version, 'position_bonus': position_bonus,
                'combo_bonus': combo_bonus, 'total_bonus': position_bonus + combo_bonus,
                'combo_progress': progress_after[-1], 'drill_completed': completed,
                'diagnostics': diagnostics,
            }
        changed = self._poll_guidance()
        if changed:
            self._previous_potential = [self._potential(self._latest(row), self._guidance) for row in observations]
        for index, info in enumerate(infos):
            info['guidance']['next'] = self.guidance
            info['guidance']['next_version'] = self._version
            info['guidance']['next_combo_progress'] = self._progress[index]
            if changed:
                info['guidance']['diagnostics'].append('A new guidance goal is encoded in the next observation.')
            if self._file_warning:
                info['guidance']['diagnostics'].append(self._file_warning)
        return self._augment(observations), result_rewards, dones, infos
