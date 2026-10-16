"""Portable legend selection and compatibility of saved session settings."""

import json
import subprocess
import sys
import unittest

from legends import LEGEND_NAMES, normalize_legend
from session_profiles import PROFILE_PRESETS, SessionOptions, default_options


class LegendSelectionTests(unittest.TestCase):
    def test_catalog_names_round_trip_without_ml_or_game_assets(self):
        self.assertEqual(len(LEGEND_NAMES), len(set(LEGEND_NAMES)))
        for name in LEGEND_NAMES:
            with self.subTest(name=name):
                self.assertEqual(normalize_legend(name), name)
        program = '''
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'gymnasium', 'pynput', 'mss'}:
        raise AssertionError('unexpected dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from legends import normalize_legend
assert normalize_legend('asuri') == 'Asuri'
'''
        subprocess.run([sys.executable, '-c', program], check=True, capture_output=True, text=True)

    def test_spelling_titles_accents_and_unambiguous_prefixes(self):
        for value, expected in (
            (' ASURI ', 'Asuri'), ('bo\u0308dvar', 'Bödvar'), ('bodvar', 'Bödvar'),
            ('QUEEN_nai', 'Queen Nai'), ('lord-vraxx', 'Lord Vraxx'),
            ('vraxx', 'Lord Vraxx'), ('roland', 'Sir Roland'), ('zuva', 'King Zuva'),
            ('linfei', 'Lin Fei'), ('red raptor', 'Red Raptor'),
            ('Qinghua and Baobao', 'Qinghua & Baobao'), ('Asur', 'Asuri'),
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_legend(value), expected)
        self.assertIsNone(normalize_legend(None))

    def test_unknown_ambiguous_and_invalid_names_do_not_silently_select(self):
        for value in ('ra', 'lo', 've'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Ambiguous legend'):
                normalize_legend(value)
        for value in ('invented legend', 'asurii', '', '   ', '&', True, 23, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_legend(value)


class SessionOptionsCompatibilityTests(unittest.TestCase):
    def test_legacy_settings_without_new_fields_remain_readable(self):
        for name in PROFILE_PRESETS:
            with self.subTest(profile=name):
                current = default_options(name)
                legacy = current.to_dict()
                del legacy['legend']
                del legacy['guidance']
                self.assertEqual(SessionOptions.from_dict(legacy), current)
                self.assertIsNone(current.legend)
                self.assertEqual(current.guidance, {
                    'opponent_spacing': 'free', 'teammate_spacing': 'free',
                    'strength': 0.3, 'combo': [],
                })
        self.assertEqual(default_options(guidance=None), default_options())

    def test_new_fields_canonicalize_and_survive_json_round_trip(self):
        options = default_options(legend='bodvar', guidance={
            'opponent_spacing': 'far', 'combo': ['light', 'heavy'],
        })
        self.assertEqual(options.legend, 'Bödvar')
        self.assertEqual(options.guidance['opponent_spacing'], 'far')
        self.assertEqual(options.guidance['teammate_spacing'], 'free')
        saved = json.loads(json.dumps(options.to_dict()))
        self.assertEqual(SessionOptions.from_dict(saved), options)

    def test_guidance_is_not_shared_with_callers_other_options_or_saved_settings(self):
        supplied = {'combo': ['light', 'heavy']}
        first = default_options(guidance=supplied)
        second = default_options()
        supplied['combo'].append('light')
        self.assertEqual(first.guidance['combo'], ['light', 'heavy'])
        saved = first.to_dict()
        saved['guidance']['combo'].append('light')
        self.assertEqual(first.guidance['combo'], ['light', 'heavy'])
        first.guidance['combo'].append('jump')
        self.assertEqual(second.guidance['combo'], [])

    def test_invalid_guidance_and_unknown_legend_are_rejected_by_settings(self):
        for changes in (
            {'legend': 'imaginary hero'}, {'legend': 'ra'},
            {'guidance': []},
            {'guidance': {'opponent_spacing': 'teleport'}},
            {'guidance': {'strength': float('nan')}},
            {'guidance': {'combo': ['light']}},
            {'guidance': {'unrecognized': True}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                default_options(**changes)


if __name__ == '__main__':
    unittest.main()
