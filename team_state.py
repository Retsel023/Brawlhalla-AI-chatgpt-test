"""Explicit duel/team roles with one observation shape for transferable policies.

Roles and friendly-fire settings must come from configuration or a verified
provider. Damage changes cannot identify which player caused damage, and an
elimination edge is not an episode boundary or evidence of a stock loss.
"""

from dataclasses import dataclass
import math

import numpy as np

from game_state import FEATURE_NAMES as LEGACY_FEATURE_NAMES
from game_state import GameState, PlayerState, RewardProjection
from telemetry import PLAYER_KEY, validate_players


FEATURE_NAMES = LEGACY_FEATURE_NAMES + (
    "ally_x", "ally_y", "ally_damage", "ally_eliminated",
    "ally_relative_x", "ally_relative_y", "ally_present",
    "enemy2_x", "enemy2_y", "enemy2_damage", "enemy2_eliminated",
    "enemy2_relative_x", "enemy2_relative_y", "enemy2_present",
    "is_2v2", "friendly_fire",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def _positive_scale(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a positive finite number") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class TeamState:
    self_player: PlayerState
    opponents: tuple[PlayerState, ...]
    teammate: PlayerState | None = None
    friendly_fire: bool = False

    def __post_init__(self):
        if not isinstance(self.self_player, PlayerState):
            raise ValueError("self_player must be a PlayerState")
        if not isinstance(self.opponents, (tuple, list)):
            raise ValueError("opponents must be an explicitly ordered sequence")
        object.__setattr__(self, "opponents", tuple(self.opponents))
        if not all(isinstance(player, PlayerState) for player in self.opponents):
            raise ValueError("opponents must contain PlayerState instances")
        if self.teammate is not None and not isinstance(self.teammate, PlayerState):
            raise ValueError("teammate must be a PlayerState or None")
        if not ((len(self.opponents) == 1 and self.teammate is None)
                or (len(self.opponents) == 2 and self.teammate is not None)):
            raise ValueError("use one opponent without an ally for 1v1, or two opponents and an ally for 2v2")
        if not isinstance(self.friendly_fire, bool):
            raise ValueError("friendly_fire must be a boolean")
        slots = [player.slot for player in self.selected_players]
        if len(set(slots)) != len(slots):
            raise ValueError("all selected player slots must be distinct")

    @property
    def format(self):
        return "2v2" if self.teammate is not None else "1v1"

    @property
    def selected_players(self):
        """All assigned players in stable self, enemy1, ally, enemy2 order."""
        if self.teammate is None:
            return self.self_player, self.opponents[0]
        return self.self_player, self.opponents[0], self.teammate, self.opponents[1]

    @classmethod
    def from_players(cls, players, self_slot, opponent_slots,
                     teammate_slot=None, friendly_fire=False):
        """Use assigned roles; never infer teams from payload order or names."""
        normalized = validate_players(players)
        if not isinstance(opponent_slots, (tuple, list)):
            raise ValueError("opponent_slots must be an explicitly ordered sequence")
        if not ((len(opponent_slots) == 1 and teammate_slot is None)
                or (len(opponent_slots) == 2 and teammate_slot is not None)):
            raise ValueError("use one opponent without an ally for 1v1, or two opponents and an ally for 2v2")
        slots = (self_slot, *opponent_slots)
        if teammate_slot is not None:
            slots += (teammate_slot,)
        for slot in slots:
            if not isinstance(slot, str) or not PLAYER_KEY.fullmatch(slot):
                raise ValueError("slots must be explicit Player<number> keys")
            if slot not in normalized:
                raise ValueError(f"selected slot {slot} is missing")
        if len(set(slots)) != len(slots):
            raise ValueError("all selected player slots must be distinct")
        make_player = lambda slot: PlayerState.from_payload(slot, normalized[slot])
        return cls(make_player(self_slot), tuple(make_player(slot) for slot in opponent_slots),
                   make_player(teammate_slot) if teammate_slot is not None else None,
                   friendly_fire)

    def to_vector(self, position_scale=1000.0, damage_scale=100.0):
        """Preserve the ten legacy duel features, then add masked team context.

        Each optional player has x/y/damage/eliminated/relative-x/relative-y and
        a presence bit. All seven values are zero when that role is absent.
        """
        position_scale = _positive_scale(position_scale, "position_scale")
        damage_scale = _positive_scale(damage_scale, "damage_scale")
        legacy = GameState(self.self_player, self.opponents[0]).to_vector(
            position_scale, damage_scale)
        own_x = self.self_player.x / position_scale
        own_y = self.self_player.y / position_scale

        def optional_features(player):
            if player is None:
                return (0.0,) * 7
            x, y = player.x / position_scale, player.y / position_scale
            return (x, y, player.damage / damage_scale, float(player.eliminated),
                    x - own_x, y - own_y, 1.0)

        extra = (optional_features(self.teammate)
                 + optional_features(self.opponents[1] if len(self.opponents) == 2 else None)
                 + (float(self.format == "2v2"), float(self.friendly_fire)))
        limit = float(np.finfo(np.float32).max)
        if any(not math.isfinite(value) or abs(value) > limit for value in extra):
            raise ValueError("normalized observation cannot be represented as finite float32")
        return np.concatenate((legacy, np.asarray(extra, dtype=np.float32)))


class TeamRewardTracker:
    """Measure net team damage deltas without attributing individual attacks.

    Enemy increases count as observed damage dealt; self and ally increases as
    damage received. Friendly damage cannot be distinguished with these fields.
    Any selected-player damage decrease or identity/role change rebaselines the
    entire transition with zero reward. Only the provider can end an episode.
    """

    def __init__(self, damage_reward_scale=0.01):
        self.damage_reward_scale = _positive_scale(damage_reward_scale, "damage_reward_scale")
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, state):
        if not isinstance(state, TeamState):
            raise ValueError("state must be a TeamState")
        previous = self.previous
        if previous is None:
            self.previous = state
            return RewardProjection(baseline_reset=True, diagnostics=("initial_baseline",))

        diagnostics = []
        if state.format != previous.format:
            diagnostics.append("format_changed")
        if state.friendly_fire != previous.friendly_fire:
            diagnostics.append("friendly_fire_changed")
        pairs = tuple(zip(previous.selected_players, state.selected_players))
        for before, after in pairs:
            if before.slot != after.slot:
                diagnostics.append(f"{before.slot}_slot_changed_to_{after.slot}")
            if before.identity != after.identity:
                diagnostics.append(f"{after.slot}_identity_changed")
            if after.damage < before.damage:
                diagnostics.append(f"{after.slot}_damage_decreased")
        if diagnostics:
            self.previous = state
            return RewardProjection(baseline_reset=True, discontinuity=True,
                                    diagnostics=tuple(diagnostics))

        try:
            dealt = math.fsum(max(0.0, after.damage - before.damage)
                              for before, after in zip(previous.opponents, state.opponents))
            received_deltas = [max(0.0, state.self_player.damage - previous.self_player.damage)]
            if state.teammate is not None:
                received_deltas.append(max(0.0, state.teammate.damage - previous.teammate.damage))
            received = math.fsum(received_deltas)
            reward = (dealt - received) * self.damage_reward_scale
        except OverflowError:
            raise ValueError("team damage deltas and scaled reward must be finite") from None
        if not all(math.isfinite(value) for value in (dealt, received, reward)):
            raise ValueError("team damage deltas and scaled reward must be finite")
        eliminated = tuple(after.slot for before, after in pairs
                           if not before.eliminated and after.eliminated)
        result = RewardProjection(reward, dealt, received, bool(eliminated), eliminated)
        self.previous = state
        return result
