"""Standard-library validation and schema for optional tactical training goals."""

import math


GUIDANCE_SCHEMA = 'tactical-v1'
ACTION_LABELS = (
    'idle', 'left', 'right', 'jump', 'down', 'light', 'heavy', 'dodge',
    'left_jump', 'right_jump', 'left_light', 'right_light', 'jump_light',
    'down_light', 'left_heavy', 'right_heavy',
)
OPPONENT_SPACING = ('free', 'close', 'mid', 'far')
TEAMMATE_SPACING = ('free', 'near', 'spread')
ATTACK_LABELS = frozenset(label for label in ACTION_LABELS if 'light' in label or 'heavy' in label)
DEFAULT_GUIDANCE = {'opponent_spacing': 'free', 'teammate_spacing': 'free', 'strength': 0.3, 'combo': []}
GUIDANCE_FEATURE_NAMES = (
    tuple('goal_opponent_' + value for value in OPPONENT_SPACING)
    + tuple('goal_teammate_' + value for value in TEAMMATE_SPACING)
    + ('goal_strength',)
    + tuple(f'goal_combo_{step}_{label}' for step in range(1, 7) for label in ACTION_LABELS)
    + ('goal_combo_length', 'goal_combo_progress')
)


def validate_guidance(payload):
    """Fill defaults and return a fresh canonical object; reject unknown keys.

    A combo is an exact input drill, not a verified weapon-specific game combo.
    Movement may occur between attacks, but a drill must include an attack and
    cannot contain idle steps or consist entirely of movement repetition.
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError('guidance must be a JSON object')
    if set(payload) - set(DEFAULT_GUIDANCE):
        raise ValueError('guidance contains unknown settings')
    opponent = payload.get('opponent_spacing', DEFAULT_GUIDANCE['opponent_spacing'])
    teammate = payload.get('teammate_spacing', DEFAULT_GUIDANCE['teammate_spacing'])
    if not isinstance(opponent, str) or opponent not in OPPONENT_SPACING:
        raise ValueError('opponent_spacing must be free, close, mid, or far')
    if not isinstance(teammate, str) or teammate not in TEAMMATE_SPACING:
        raise ValueError('teammate_spacing must be free, near, or spread')
    strength = payload.get('strength', DEFAULT_GUIDANCE['strength'])
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError('guidance strength must be a finite number between 0 and 1')
    try:
        strength = float(strength)
    except (OverflowError, ValueError):
        raise ValueError('guidance strength must be a finite number between 0 and 1') from None
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('guidance strength must be a finite number between 0 and 1')
    combo = payload.get('combo', [])
    if not isinstance(combo, list) or len(combo) == 1 or len(combo) > 6:
        raise ValueError('combo must be an empty list or a sequence of 2 to 6 action labels')
    if any(not isinstance(label, str) or label not in ACTION_LABELS or label == 'idle' for label in combo):
        raise ValueError('combo entries must be supported non-idle action labels')
    if combo and not any(label in ATTACK_LABELS for label in combo):
        raise ValueError('an input drill must include at least one attack')
    return {'opponent_spacing': opponent, 'teammate_spacing': teammate,
            'strength': strength, 'combo': list(combo)}
