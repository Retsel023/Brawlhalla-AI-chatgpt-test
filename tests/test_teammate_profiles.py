"""Teammate arrangements, compatible settings, and the menu's canonical form."""
import unittest

from mode_menu import TEAMMATE_LABELS, build_options, options_to_form
from session_profiles import (PROFILE_FIELDS, SessionOptions, controlled_slots, default_options,
                              make_profile, validate_profile)


class TeammateProfileTests(unittest.TestCase):
    def test_default_arrangements_and_controlled_slots(self):
        for name in ('human_duel', 'bot_training', 'online_duel'):
            profile = make_profile(name)
            self.assertEqual(profile['teammate_kind'], 'none')
            self.assertIsNone(profile['teammate_account_id'])
            self.assertEqual(controlled_slots(profile), ('Player1',))
        self.assertEqual(make_profile('bot_training', match_format='2v2')['teammate_kind'], 'bot')
        self.assertEqual(make_profile('online_teams')['teammate_kind'], 'random')

    def test_old_eight_field_profiles_and_settings_remain_readable(self):
        for name, kwargs in [('human_duel', {}), ('online_teams', {}),
                             ('bot_training', {'match_format': '2v2'})]:
            with self.subTest(mode=name):
                options = default_options(name, **kwargs)
                legacy = {field: options.profile[field] for field in PROFILE_FIELDS}
                self.assertEqual(len(legacy), 8)
                self.assertEqual(validate_profile(legacy), options.profile)
                saved = options.to_dict()
                saved['profile'] = legacy
                restored = SessionOptions.from_dict(saved)
                self.assertEqual(restored, options)
                self.assertEqual(build_options(options_to_form(restored)), options)

    def test_ai_controls_exactly_two_explicit_slots(self):
        profile = make_profile('online_teams', teammate_kind='ai', self_player='Player8',
                               teammate_player='Player6', opponents=['Player2', 'Player4'])
        self.assertEqual(controlled_slots(profile), ('Player8', 'Player6'))
        self.assertNotIn('controlled_players', profile)
        for kind in ('random', 'regular', 'bot'):
            self.assertEqual(controlled_slots(make_profile('online_teams', teammate_kind=kind)), ('Player1',))

    def test_regular_account_id_optional_or_nonnegative_integer(self):
        for account_id in (None, 0, 123456, 10 ** 40):
            with self.subTest(account_id=account_id):
                profile = make_profile('online_teams', teammate_kind='regular', teammate_account_id=account_id)
                self.assertEqual(profile['teammate_account_id'], account_id)
        for account_id in (True, False, -1, 1.5, '123', [], {}):
            with self.subTest(account_id=account_id), self.assertRaises(ValueError):
                make_profile('online_teams', teammate_kind='regular', teammate_account_id=account_id)

    def test_kind_and_account_id_must_match_arrangement(self):
        for kind in ('random', 'ai', 'bot', 'none'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                make_profile('online_teams', teammate_kind=kind, teammate_account_id=123)
        for kind in ('random', 'regular', 'ai', 'bot'):
            with self.subTest(duel_kind=kind), self.assertRaises(ValueError):
                make_profile('human_duel', teammate_kind=kind)
        for kind in ('none', 'unknown', '', True, [], None):
            with self.subTest(team_kind=kind), self.assertRaises(ValueError):
                validate_profile({**make_profile('online_teams'), 'teammate_kind': kind})

    def test_menu_round_trips_random_regular_ai_and_bot(self):
        for kind in ('random', 'regular', 'ai', 'bot'):
            for account_id in ((None, 123) if kind == 'regular' else (None,)):
                with self.subTest(kind=kind, account_id=account_id):
                    options = default_options('online_teams', teammate_kind=kind, teammate_account_id=account_id)
                    form = options_to_form(options)
                    self.assertEqual(build_options(form), options)
                    form['teammate_kind'] = TEAMMATE_LABELS[kind]
                    self.assertEqual(build_options(form), options)

    def test_menu_rejects_invalid_account_text(self):
        form = options_to_form(default_options('online_teams', teammate_kind='regular'))
        for account_id in ('-1', '1.5', 'hello', True, 1.5):
            with self.subTest(account_id=account_id), self.assertRaises(ValueError):
                build_options({**form, 'teammate_account_id': account_id})


if __name__ == '__main__':
    unittest.main()
