"""Guidance configuration stays strict and available without the ML stack."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from controls import ACTIONS
from guidance_settings import ACTION_LABELS, DEFAULT_GUIDANCE, GUIDANCE_FEATURE_NAMES, validate_guidance


class GuidanceSettingsTests(unittest.TestCase):
    def test_defaults_are_canonical_and_return_independent_lists(self):
        self.assertEqual(validate_guidance(None), DEFAULT_GUIDANCE)
        self.assertEqual(validate_guidance({}), DEFAULT_GUIDANCE)
        selected = validate_guidance({'opponent_spacing': 'mid', 'combo': ['jump', 'light']})
        self.assertEqual(set(selected), set(DEFAULT_GUIDANCE))
        self.assertEqual(selected['strength'], 0.3)
        selected['combo'].append('heavy')
        self.assertEqual(validate_guidance({})['combo'], [])
        self.assertEqual(len(GUIDANCE_FEATURE_NAMES), 106)
        self.assertEqual(len(set(GUIDANCE_FEATURE_NAMES)), 106)
        self.assertEqual(len(ACTION_LABELS), len(ACTIONS))
        for label, action in zip(ACTION_LABELS, ACTIONS):
            self.assertEqual(label, '_'.join(action) if action else 'idle')

    def test_drills_require_attacks_but_allow_intermediate_movement(self):
        for combo in (['jump', 'light'], ['dodge', 'left', 'right_light'],
                      ['light', 'down', 'heavy'], ['left_jump', 'down_light']):
            with self.subTest(combo=combo):
                self.assertEqual(validate_guidance({'combo': combo})['combo'], combo)
        for combo in (['light'], ['idle', 'light'], ['left', 'right'], ['jump', 'dodge'],
                      ['light'] * 7, ['light', 'attack'], ('light', 'heavy'), 'light heavy', [0, 'light']):
            with self.subTest(combo=combo), self.assertRaises(ValueError):
                validate_guidance({'combo': combo})

    def test_invalid_settings_are_rejected(self):
        for payload in ([], '', {'strength': True}, {'strength': -1}, {'strength': 1.1},
                        {'strength': '0.3'}, {'strength': float('nan')},
                        {'strength': float('inf')}, {'strength': 10 ** 500},
                        {'opponent_spacing': 'near'}, {'opponent_spacing': []},
                        {'teammate_spacing': 'close'}, {'unknown': 'value'}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_guidance(payload)

    def test_module_imports_without_numerical_or_desktop_dependencies(self):
        script = """
import builtins, json
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'gymnasium', 'stable_baselines3', 'pynput', 'tkinter'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from guidance_settings import validate_guidance
print(json.dumps(validate_guidance({'opponent_spacing': 'close'})))
"""
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['opponent_spacing'], 'close')


if __name__ == '__main__':
    unittest.main()
