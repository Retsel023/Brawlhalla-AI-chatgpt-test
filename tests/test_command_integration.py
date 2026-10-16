"""Check command persistence and background work without a desktop or model."""

import contextlib
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from guidance_io import write_guidance, write_json_atomic
from language_cli import main
from language_commands import CommandResult
from mode_menu import LanguageCommandTask, SessionProcess
from session_profiles import default_options


class CommandCliIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = self.directory / 'config.json'
        self.config.write_text('{}', encoding='utf-8')
        self.guidance = self.directory / 'guidance.json'
        self.initial = {
            'opponent_spacing': 'mid', 'teammate_spacing': 'spread',
            'strength': 0.7, 'combo': ['down_light', 'jump_light'],
        }
        write_guidance(self.guidance, self.initial)

    def run_cli(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            code = main(['keep more distance', '--config', str(self.config),
                         '--guidance-file', str(self.guidance), *arguments])
        return code, errors.getvalue()

    def test_live_partial_commands_preserve_current_goals_over_stale_session_file(self):
        saved = self.directory / 'session.json'
        # Saved settings can lag behind changes made through a running menu.
        saved.write_text(json.dumps(default_options().to_dict()), encoding='utf-8')
        received = []

        def interpret(_text, options, _config):
            received.append(deepcopy(options.guidance))
            updated = replace(options, guidance={**options.guidance, 'opponent_spacing': 'far'})
            return CommandResult(updated, 'Spacing goal changed.', True, False)

        with patch('language_commands.interpret_command', side_effect=interpret):
            code, errors = self.run_cli('--session-file', str(saved))
        self.assertEqual(code, 0, errors)
        self.assertEqual(received, [self.initial])
        self.assertEqual(json.loads(self.guidance.read_text(encoding='utf-8')),
                         {**self.initial, 'opponent_spacing': 'far'})
        self.assertEqual(json.loads(saved.read_text(encoding='utf-8')),
                         default_options().to_dict())

    def test_live_file_is_the_context_even_without_a_saved_session(self):
        def interpret(_text, options, _config):
            self.assertEqual(options.guidance, self.initial)
            return CommandResult(options, 'No change.', False, False)

        with patch('language_commands.interpret_command', side_effect=interpret):
            code, errors = self.run_cli()
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(self.guidance.read_text(encoding='utf-8')), self.initial)

    def test_invalid_live_file_is_rejected_before_model_request_or_output_write(self):
        output = self.directory / 'new-session.json'
        for invalid in ('{"combo": ["invented"]}', '{"opponent_spacing":'):
            with self.subTest(invalid=invalid):
                self.guidance.write_text(invalid, encoding='utf-8')
                with patch('language_commands.interpret_command') as interpret:
                    code, _ = self.run_cli('--output', str(output))
                self.assertEqual(code, 2)
                interpret.assert_not_called()
                self.assertEqual(self.guidance.read_text(encoding='utf-8'), invalid)
                self.assertFalse(output.exists())

    def test_setup_change_is_rejected_atomically_for_live_command(self):
        original = self.guidance.read_bytes()
        output = self.directory / 'new-session.json'

        def interpret(_text, options, _config):
            return CommandResult(replace(options, legend='Hattori', guidance={
                **options.guidance, 'opponent_spacing': 'far'}), 'Proposed change.', True, False)

        with patch('language_commands.interpret_command', side_effect=interpret):
            code, errors = self.run_cli('--output', str(output))
        self.assertEqual(code, 2)
        self.assertIn('Live commands can change tactical goals only', errors)
        self.assertEqual(self.guidance.read_bytes(), original)
        self.assertFalse(output.exists())

    def test_failed_atomic_replace_preserves_previous_file_and_removes_staging_file(self):
        original = self.guidance.read_bytes()
        with patch('guidance_io.os.replace', side_effect=OSError('fixture disk failure')), \
                self.assertRaisesRegex(OSError, 'fixture disk failure'):
            write_json_atomic(self.guidance, {'opponent_spacing': 'far'})
        self.assertEqual(self.guidance.read_bytes(), original)
        self.assertFalse(list(self.directory.glob('*.tmp')))


class BackgroundCommandTests(unittest.TestCase):
    def test_model_work_runs_off_thread_and_delivers_result_once(self):
        options = default_options(guidance={'opponent_spacing': 'close'})
        original = deepcopy(options.to_dict())
        entered, complete = threading.Event(), threading.Event()
        caller = threading.get_ident()
        worker_threads = []

        def interpret(_text, supplied, _config):
            worker_threads.append(threading.get_ident())
            entered.set()
            if not complete.wait(2):
                raise RuntimeError('fixture was not released')
            return CommandResult(replace(supplied, guidance={
                **supplied.guidance, 'opponent_spacing': 'far'}), 'Goal changed.', True, False)

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text('{}', encoding='utf-8')
            with patch('language_commands.interpret_command', side_effect=interpret):
                task = LanguageCommandTask('keep distance', options, config)
                try:
                    self.assertTrue(entered.wait(1))
                    self.assertIsNone(task.poll())
                finally:
                    complete.set()
                    task.thread.join(timeout=2)
        self.assertFalse(task.thread.is_alive())
        result, error = task.poll()
        self.assertIsNone(error)
        self.assertEqual(result.options.guidance['opponent_spacing'], 'far')
        self.assertEqual(options.to_dict(), original)
        self.assertEqual(task.options.to_dict(), original)
        self.assertNotEqual(worker_threads, [caller])
        self.assertIsNone(task.poll())

    def test_model_failure_returns_error_without_mutating_settings(self):
        options = default_options(legend='Hattori', guidance={'opponent_spacing': 'close'})
        original = deepcopy(options.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text('{}', encoding='utf-8')
            with patch('language_commands.interpret_command', side_effect=RuntimeError('model unavailable')):
                task = LanguageCommandTask('keep distance', options, config)
                task.thread.join(timeout=2)
        self.assertFalse(task.thread.is_alive())
        self.assertEqual(task.poll(), (None, 'model unavailable'))
        self.assertEqual(options.to_dict(), original)
        self.assertEqual(task.options.to_dict(), original)

    def test_process_accepts_live_goals_only_until_stop_or_check(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            script = repo / 'child.py'
            script.write_text(
                'from pathlib import Path\nimport sys, time\n'
                'while not Path(sys.argv[1]).exists():\n    time.sleep(0.01)\n',
                encoding='utf-8')
            process = SessionProcess(repo, repo / 'config.json', command_builder=
                                     lambda options, config, stop, check:
                                     [sys.executable, '-u', str(script), str(stop)])
            options = default_options(guidance={'opponent_spacing': 'close'})
            process.start(options)
            try:
                process.update_guidance({**options.guidance, 'opponent_spacing': 'far'})
                updated = process.guidance_path.read_bytes()
                self.assertEqual(json.loads(updated)['opponent_spacing'], 'far')
                process.request_stop()
                with self.assertRaises(RuntimeError):
                    process.update_guidance(options.guidance)
                self.assertEqual(process.guidance_path.read_bytes(), updated)
                self.assertEqual(process.process.wait(timeout=3), 0)
                process.reader.join(timeout=2)
            finally:
                process.force_stop()
                process.process.wait(timeout=3)
                process.reader.join(timeout=2)

            process.start(options, check=True)
            try:
                original = process.guidance_path.read_bytes()
                with self.assertRaises(RuntimeError):
                    process.update_guidance({'opponent_spacing': 'far'})
                self.assertEqual(process.guidance_path.read_bytes(), original)
            finally:
                process.request_stop()
                process.process.wait(timeout=3)
                process.reader.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
