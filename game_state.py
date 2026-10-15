"""Represent the fields actually present in Brawlhalla's spectator export.

These observations have no simulation tick, stock count, or match outcome.
Reward projections describe observed damage changes; they do not establish that
an agent's actions caused those changes. A provider controlling the game must
supply its own synchronized steps, episode boundaries, and reset confirmation.
"""

from dataclasses import dataclass
import math

import numpy as np

from telemetry import PLAYER_KEY, validate_players


FEATURE_NAMES = (
    "self_x", "self_y", "self_damage", "self_eliminated",
    "opponent_x", "opponent_y", "opponent_damage", "opponent_eliminated",
    "relative_x", "relative_y",
)


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _positive_scale(value, field):
    result = _finite_number(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


@dataclass(frozen=True)
class PlayerState:
    slot: str
    x: float
    y: float
    damage: float
    eliminated: bool
    brawlhalla_id: int | None = None
    player_name: str | None = None

    def __post_init__(self):
        if not isinstance(self.slot, str) or not PLAYER_KEY.fullmatch(self.slot):
            raise ValueError("slot must be an explicit Player<number> key")
        for field in ("x", "y", "damage"):
            object.__setattr__(self, field, _finite_number(getattr(self, field), field))
        if self.damage < 0:
            raise ValueError("damage must be nonnegative")
        if not isinstance(self.eliminated, bool):
            raise ValueError("eliminated must be a boolean")
        if self.brawlhalla_id is not None and (
                isinstance(self.brawlhalla_id, bool)
                or not isinstance(self.brawlhalla_id, int)
                or self.brawlhalla_id < 0):
            raise ValueError("brawlhalla_id must be a nonnegative integer or None")
        if self.player_name is not None and not isinstance(self.player_name, str):
            raise ValueError("player_name must be a string or None")

    @classmethod
    def from_payload(cls, slot, raw):
        player = validate_players({slot: raw})[slot]
        return cls(slot, player["x"], player["y"], player["CurrDamage"],
                   player["Eliminated"], player.get("BrawlhallaID"),
                   player.get("PlayerName"))

    @property
    def identity(self):
        """Optional exported identity; never guess identity from position/order."""
        return self.brawlhalla_id, self.player_name


@dataclass(frozen=True)
class GameState:
    self_player: PlayerState
    opponent: PlayerState

    def __post_init__(self):
        if not isinstance(self.self_player, PlayerState) or not isinstance(self.opponent, PlayerState):
            raise ValueError("self_player and opponent must be PlayerState instances")
        if self.self_player.slot == self.opponent.slot:
            raise ValueError("self and opponent slots must be different")

    @classmethod
    def from_players(cls, players, self_slot, opponent_slot):
        """Select two explicit slots, without inferring which player is the agent."""
        normalized = validate_players(players)
        if self_slot == opponent_slot:
            raise ValueError("self and opponent slots must be different")
        for slot in (self_slot, opponent_slot):
            if not isinstance(slot, str) or not PLAYER_KEY.fullmatch(slot):
                raise ValueError("slots must be explicit Player<number> keys")
            if slot not in normalized:
                raise ValueError(f"selected slot {slot} is missing")
        return cls(PlayerState.from_payload(self_slot, normalized[self_slot]),
                   PlayerState.from_payload(opponent_slot, normalized[opponent_slot]))

    def to_vector(self, position_scale=1000.0, damage_scale=100.0):
        """Return ten normalized float32 features, with opponent-relative offsets.

        Scales are positive divisors, not stage dimensions or clipping limits.
        Position signs and coordinate orientation are preserved from the export.
        """
        position_scale = _positive_scale(position_scale, "position_scale")
        damage_scale = _positive_scale(damage_scale, "damage_scale")
        own, other = self.self_player, self.opponent
        own_x, own_y = own.x / position_scale, own.y / position_scale
        other_x, other_y = other.x / position_scale, other.y / position_scale
        features = (
            own_x, own_y, own.damage / damage_scale,
            float(own.eliminated), other_x, other_y,
            other.damage / damage_scale, float(other.eliminated),
            other_x - own_x, other_y - own_y,
        )
        limit = float(np.finfo(np.float32).max)
        if any(not math.isfinite(value) or abs(value) > limit for value in features):
            raise ValueError("normalized observation cannot be represented as finite float32")
        return np.asarray(features, dtype=np.float32)


@dataclass(frozen=True)
class RewardProjection:
    """Observed deltas, with an elimination edge separate from match outcome."""
    reward: float = 0.0
    damage_dealt: float = 0.0
    damage_received: float = 0.0
    terminal_proxy: bool = False
    eliminated_slots: tuple[str, ...] = ()
    baseline_reset: bool = False
    discontinuity: bool = False
    diagnostics: tuple[str, ...] = ()


class RewardTracker:
    """Project damage deltas; never infer stocks, a winner, or a game reset.

    Any damage decrease or identity change makes the whole transition ambiguous.
    Such samples become a fresh baseline with zero reward. Eliminated false->true
    transitions are reported once as a proxy only, with no invented stock bonus.
    """

    def __init__(self, damage_scale=0.01):
        self.damage_scale = _positive_scale(damage_scale, "damage_scale")
        self.previous = None

    def reset(self):
        """Clear history when a provider confirms a new episode or loses state."""
        self.previous = None

    def update(self, state):
        if not isinstance(state, GameState):
            raise ValueError("state must be a GameState")
        previous = self.previous
        if previous is None:
            self.previous = state
            return RewardProjection(baseline_reset=True, diagnostics=("initial_baseline",))

        diagnostics = []
        pairs = ((previous.self_player, state.self_player),
                 (previous.opponent, state.opponent))
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

        dealt = max(0.0, state.opponent.damage - previous.opponent.damage)
        received = max(0.0, state.self_player.damage - previous.self_player.damage)
        reward = (dealt - received) * self.damage_scale
        if not math.isfinite(reward):
            raise ValueError("scaled damage reward must be finite")
        eliminated = tuple(after.slot for before, after in pairs
                           if not before.eliminated and after.eliminated)
        result = RewardProjection(reward, dealt, received, bool(eliminated), eliminated)
        self.previous = state
        return result
