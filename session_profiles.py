"""Standard-library configuration for selectable Brawlhalla sessions."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

PLAYER_SLOT = re.compile(r'Player[0-9]+\Z')
PROFILE_PRESETS = {
    'human_duel': {'label': 'Fight the AI — 1v1', 'description': 'Play a private 1v1 against the saved AI. Learning is off by default.',
                   'match_format': '1v1', 'opponent_kind': 'human', 'online': False, 'learning': False},
    'bot_training': {'label': 'Train against bots', 'description': 'Train continuously against bots, with checkpoints between sessions.',
                     'match_format': '1v1', 'opponent_kind': 'bot', 'online': False, 'learning': True},
    'online_duel': {'label': 'Online humans — 1v1', 'description': 'Learn from online 1v1 matches using a provider that supports online sessions.',
                    'match_format': '1v1', 'opponent_kind': 'human', 'online': True, 'learning': True},
    'online_teams': {'label': 'Online humans — 2v2', 'description': 'Learn team play while starting from compatible 1v1 skills.',
                     'match_format': '2v2', 'opponent_kind': 'human', 'online': True, 'learning': True},
}
PROFILE_FIELDS = ('profile', 'match_format', 'opponent_kind', 'online', 'self_player',
                  'teammate_player', 'opponents', 'friendly_fire')
TEAMMATE_FIELDS = ('teammate_kind', 'teammate_account_id')
TEAMMATE_KINDS = ('none', 'random', 'regular', 'ai', 'bot')


def default_teammate_kind(name, match_format):
    if match_format == '1v1':
        return 'none'
    return 'bot' if name == 'bot_training' else 'random'


def validate_profile(payload):
    if not isinstance(payload, dict):
        raise ValueError('profile must be an object')
    missing = set(PROFILE_FIELDS) - payload.keys()
    if missing:
        raise ValueError(f'Missing session profile fields: {sorted(missing)}')
    name = payload['profile']
    if not isinstance(name, str) or name not in PROFILE_PRESETS:
        raise ValueError('Select a supported session profile')
    preset = PROFILE_PRESETS[name]
    match_format = payload['match_format']
    if match_format not in ('1v1', '2v2'):
        raise ValueError('match_format must be 1v1 or 2v2')
    if name != 'bot_training' and match_format != preset['match_format']:
        raise ValueError(f'{name} requires {preset["match_format"]}')
    if type(payload['online']) is not bool or payload['online'] != preset['online']:
        raise ValueError(f'{name} requires online={preset["online"]}')
    if payload['opponent_kind'] != preset['opponent_kind']:
        raise ValueError(f'{name} requires opponent_kind={preset["opponent_kind"]}')
    if type(payload['friendly_fire']) is not bool:
        raise ValueError('friendly_fire must be a boolean')
    own, ally, enemies = payload['self_player'], payload['teammate_player'], payload['opponents']
    if not isinstance(enemies, (list, tuple)) or len(enemies) != (1 if match_format == '1v1' else 2):
        raise ValueError('Select one opponent for 1v1, or two for 2v2')
    if match_format == '1v1':
        if ally is not None or payload['friendly_fire']:
            raise ValueError('1v1 requires no teammate and friendly_fire=false')
        slots = [own, *enemies]
    else:
        slots = [own, ally, *enemies]
    if not all(isinstance(slot, str) and PLAYER_SLOT.fullmatch(slot) for slot in slots):
        raise ValueError('All player roles need explicit Player<number> keys')
    if len(set(slots)) != len(slots):
        raise ValueError('Self, teammate, and opponent roles must be different players')
    kind = payload.get('teammate_kind', default_teammate_kind(name, match_format))
    account_id = payload.get('teammate_account_id')
    if kind not in TEAMMATE_KINDS:
        raise ValueError('teammate_kind must be none, random, regular, ai, or bot')
    if (match_format == '1v1') != (kind == 'none'):
        raise ValueError('1v1 requires teammate_kind=none; 2v2 requires a teammate arrangement')
    if account_id is not None and (type(account_id) is not int or account_id < 0):
        raise ValueError('teammate_account_id must be a nonnegative integer or null')
    if account_id is not None and kind != 'regular':
        raise ValueError('teammate_account_id is only supported for a regular teammate')
    canonical = {field: (list(enemies) if field == 'opponents' else payload[field]) for field in PROFILE_FIELDS}
    canonical.update(teammate_kind=kind, teammate_account_id=account_id)
    return canonical


def controlled_slots(profile):
    """Return the explicit player slots controlled by this policy session."""
    profile = validate_profile(profile)
    slots = (profile['self_player'],)
    return slots + (profile['teammate_player'],) if profile['teammate_kind'] == 'ai' else slots


def make_profile(name, *, match_format=None, self_player='Player1', teammate_player=None,
                 opponents=None, friendly_fire=False, teammate_kind=None, teammate_account_id=None):
    if name not in PROFILE_PRESETS:
        raise ValueError(f'Unknown session profile: {name}')
    preset = PROFILE_PRESETS[name]
    match_format = match_format or preset['match_format']
    teams = match_format == '2v2'
    if not teams and (teammate_player is not None or friendly_fire):
        raise ValueError('1v1 requires no teammate and friendly_fire=false')
    return validate_profile({
        'profile': name, 'match_format': match_format, 'opponent_kind': preset['opponent_kind'],
        'online': preset['online'], 'self_player': self_player,
        'teammate_player': (teammate_player or 'Player2') if teams else None,
        'opponents': list(opponents) if opponents is not None else (['Player3', 'Player4'] if teams else ['Player2']),
        'friendly_fire': friendly_fire,
        'teammate_kind': default_teammate_kind(name, match_format) if teammate_kind is None else teammate_kind,
        'teammate_account_id': teammate_account_id,
    })


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f'{name} must be {"nonnegative" if allow_zero else "positive"}')
    return value


@dataclass(frozen=True)
class SessionOptions:
    profile: dict
    learning: bool
    steps: int = 0
    checkpoint_interval: int = 1000
    model_root: str = 'models/sessions'
    initialize_from: str | None = None
    device: str = 'auto'
    max_seconds: float = 0

    def __post_init__(self):
        object.__setattr__(self, 'profile', validate_profile(self.profile))
        if type(self.learning) is not bool:
            raise ValueError('learning must be a boolean')
        for name in ('steps', 'checkpoint_interval'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == 'checkpoint_interval' else 0):
                raise ValueError(f'{name} must be an integer {"> 0" if name == "checkpoint_interval" else ">= 0"}')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu, or cuda (AMD ROCm)')
        _positive(self.max_seconds, 'max_seconds', allow_zero=True)
        if not isinstance(self.model_root, str) or not self.model_root.strip():
            raise ValueError('model_root must be a nonempty path')
        if self.initialize_from is not None and (not isinstance(self.initialize_from, str) or not self.initialize_from.strip()):
            raise ValueError('initialize_from must be a nonempty checkpoint path or null')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError('session settings must be an object')
        return cls(**payload)


def default_options(name='bot_training', **kwargs):
    preset = PROFILE_PRESETS.get(name)
    if preset is None:
        raise ValueError(f'Unknown session profile: {name}')
    profile = make_profile(name, **{k: kwargs.pop(k) for k in tuple(kwargs)
                                  if k in {'match_format', 'self_player', 'teammate_player', 'opponents',
                                           'friendly_fire', *TEAMMATE_FIELDS}})
    return SessionOptions(profile, learning=kwargs.pop('learning', preset['learning']), **kwargs)
