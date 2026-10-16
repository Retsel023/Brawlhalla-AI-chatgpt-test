"""Headless checks for the menu's settings and isolated subprocess ownership."""
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from mode_menu import SessionProcess, build_options, options_to_form
from session_profiles import PROFILE_PRESETS, default_options


class ModeFormTests(unittest.TestCase):
    def test_all_modes_round_trip(self):
        for name in PROFILE_PRESETS:
            with self.subTest(mode=name):
                options = default_options(name)
                self.assertEqual(build_options(options_to_form(options)), options)
        self.assertFalse(build_options(options_to_form(default_options('human_duel'))).learning)

    def test_module_import_does_not_require_tkinter(self):
        command = [sys.executable, '-c', "import sys; sys.modules['tkinter'] = None; import mode_menu"]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bot_teams_preserve_roles_and_indefinite_training(self):
        form = options_to_form(default_options('bot_training', match_format='2v2'))
        form.update(steps='0', max_seconds='0', checkpoint_interval='250', friendly_fire=True)
        options = build_options(form)
        self.assertEqual(options.profile['opponents'], ['Player3', 'Player4'])
        self.assertEqual(options.profile['teammate_player'], 'Player2')
        self.assertTrue(options.profile['friendly_fire'])
        self.assertEqual(options.steps, 0)
        self.assertEqual(options.checkpoint_interval, 250)

    def test_bad_numbers_and_duplicate_roles_are_rejected(self):
        form = options_to_form(default_options())
        for key, value in [('steps', '-1'), ('steps', '1.5'), ('max_seconds', 'nan'),
                           ('checkpoint_interval', '0'), ('opponents', 'Player1')]:
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                build_options({**form, key: value})


class SessionProcessTests(unittest.TestCase):
    def test_default_command_uses_argument_list_and_unique_stop_file(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            controller = SessionProcess(repo, repo / 'config with spaces.json')
            with patch('mode_menu.subprocess.Popen') as popen:
                popen.return_value.poll.return_value = None
                popen.return_value.stdout = io.StringIO('')
                first = controller.start(default_options(), check=True)
                controller.reader.join(timeout=2)
                command = popen.call_args.args[0]
                self.assertIsInstance(command, list)
                self.assertFalse(popen.call_args.kwargs['shell'])
                self.assertEqual(command[:4], [sys.executable, '-u', str(repo / 'agent.py'), 'session'])
                self.assertIn('--check', command)
                self.assertEqual(command[command.index('--stop-file') + 1], str(first / 'stop.request'))
                self.assertEqual(json.loads((first / 'options.json').read_text())['steps'], 0)
                with self.assertRaises(RuntimeError):
                    controller.start(default_options())
                controller.request_stop()
                self.assertTrue((first / 'stop.request').exists())
                unrelated = repo / 'unrelated.stop'
                self.assertFalse(unrelated.exists())
                popen.return_value.poll.return_value = 0
                popen.return_value.stdout = io.StringIO('')
                second = controller.start(default_options())
                controller.reader.join(timeout=2)
                self.assertNotEqual(first, second)
                self.assertFalse((second / 'stop.request').exists())

    def test_real_child_output_and_cooperative_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            script = repo / 'child.py'
            script.write_text("import pathlib, sys, time\n"
                              "print('child started', flush=True)\n"
                              "while not pathlib.Path(sys.argv[1]).exists():\n"
                              "    time.sleep(0.01)\n"
                              "print('child saved and stopped', flush=True)\n", encoding='utf-8')
            controller = SessionProcess(repo, repo / 'config.json', command_builder=
                                        lambda options, config, stop, check: [sys.executable, '-u', str(script), str(stop)])
            controller.start(default_options())
            try:
                controller.request_stop()
                self.assertEqual(controller.process.wait(timeout=5), 0)
                controller.reader.join(timeout=2)
                lines = []
                deadline = time.monotonic() + 3
                while 'child saved and stopped' not in ''.join(lines) and time.monotonic() < deadline:
                    lines.extend(controller.drain_output())
                    time.sleep(0.01)
                self.assertIn('child started', ''.join(lines))
                self.assertIn('child saved and stopped', ''.join(lines))
                self.assertIn('child saved and stopped', (controller.run_dir / 'session.log').read_text())
            finally:
                if controller.poll() is None:
                    controller.force_stop()
                    controller.process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
