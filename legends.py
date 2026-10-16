"""Game-derived legend names, without importing game or ML dependencies.

Names were extracted from the English localization shipped with the uploaded
11.00.36 build: ``Brawlhalla/languages/language.1.bin`` at commit
131b686e2a1dfdd7758feb5afc90ae14b241a013 (Git blob
98ffcb518cc4cb463175e298346a111de00b9967; SHA-256
433c4bc49b0ffd1867482dc481699e6793cc654b266feef5ff31eb521d64aee3).
The resource has a four-byte prefix, zlib-compressed data, a big-endian entry
count, and pairs of length-prefixed UTF-8 strings. Each catalog entry occurs as
``<name>ColorSchemeType_Ranked_DisplayName`` with ``<name> Skyforged`` as value.

Localization establishes the spelling of these names, not whether a legend is
active, unlocked, or selectable in a particular installation. The game-side
provider must acknowledge actual selection. No game assets are required here.
"""

import unicodedata


LEGEND_CATALOG_BUILD = '11.00.36'
LEGEND_NAMES = (
    'Ada', 'Arcadia', 'Artemis', 'Asuri', 'Aurus', 'Azoth', 'Barraza', 'Bödvar',
    'Brynn', 'Caspian', 'Cassidy', 'Cross', 'Diana', 'Dusk', 'Ember', 'Ezio',
    'Fait', 'Gnash', 'Hattori', 'Imugi', 'Isaiah', 'Jaeyun', 'Jhala', 'Jiro',
    'Kaya', 'King Zuva', 'Koji', 'Kor', 'Lady Vera', 'Lin Fei', 'Loki',
    'Lord Vraxx', 'Lucien', 'Magyar', 'Mako', 'Mirage', 'Mordex', 'Munin', 'Nix',
    'Onyx', 'Orion', 'Petra', 'Priya', 'Qinghua & Baobao', 'Queen Nai', 'Ragnir',
    'Ransom', 'Rayman', 'Red Raptor', 'Reno', 'Rupture', 'Scarlet', 'Sentinel',
    'Seven', 'Sidra', 'Sir Roland', 'Teros', 'Tezca', 'Thatch', 'Thea', 'Thor',
    'Ulgrim', 'Val', 'Vector', 'Vivi', 'Volkov', 'Wu Shang', 'Xull', 'Yumiko',
    'Zariel',
)


def _name_key(value):
    text = unicodedata.normalize('NFKD', value.replace('&', ' and ')).casefold()
    return ''.join(character for character in text if character.isalnum())


def _aliases(name):
    yield _name_key(name)
    words = name.split()
    if words[0] in ('King', 'Lady', 'Lord', 'Queen', 'Sir'):
        yield _name_key(' '.join(words[1:]))
    if ' & ' in name:
        yield _name_key(name.replace(' & ', ' '))


_NAME_INDEX = {}
for _name in LEGEND_NAMES:
    for _alias in _aliases(_name):
        _NAME_INDEX.setdefault(_alias, set()).add(_name)


def normalize_legend(value: str | None) -> str | None:
    """Return a canonical name or None; reject unknown/ambiguous requests.

    Case, accents, whitespace, punctuation, and optional titles are normalized.
    Prefixes of at least two letters are accepted only when unambiguous. An
    empty string does not silently clear an existing selection; use None.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('legend must be a name or null')
    key = _name_key(value)
    if not key:
        raise ValueError('legend must be a nonempty name or null')
    matches = _NAME_INDEX.get(key)
    if matches is None and len(key) >= 2:
        matches = {name for alias, names in _NAME_INDEX.items()
                   if alias.startswith(key) for name in names}
    if not matches:
        raise ValueError(f'Unknown legend: {value!r}; choose a catalog name')
    if len(matches) != 1:
        raise ValueError(f'Ambiguous legend: {value!r}; choose from {", ".join(sorted(matches))}')
    return next(iter(matches))
