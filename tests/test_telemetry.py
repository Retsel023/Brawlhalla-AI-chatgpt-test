"""Exercise collector correctness against real temporary export files."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from telemetry import TelemetryPoller, collect, main, read_snapshot, validate_players


class TelemetryTests(unittest.TestCase):
    now_ns = 1_800_000_000_000_000_000

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "PlayerStatuses.json"
        self.payload = {
            "Player1": {"BrawlhallaID": 123, "PlayerName": "One", "Eliminated": False,
                        "CurrDamage": 25.5, "x": -40, "y": 1.25},
            "Player2": {"Eliminated": False, "CurrDamage": 0, "x": 50, "y": -80},
        }

    def write(self, content=None, *, mtime_ns=None):
        if content is None:
            content = json.dumps(self.payload)
        self.source.write_text(content, encoding="utf-8")
        stamp = self.now_ns if mtime_ns is None else mtime_ns
        os.utime(self.source, ns=(stamp, stamp))

    def test_first_fresh_existing_file_and_stale_file(self):
        self.write(mtime_ns=self.now_ns - 4_000_000_000)
        poller = TelemetryPoller(self.directory, max_age_seconds=3)
        self.assertIsNone(poller.poll(now_ns=self.now_ns))
        self.assertEqual(poller.stats.stale, 1)
        self.write(mtime_ns=self.now_ns - 2_000_000_000)
        record = poller.poll(now_ns=self.now_ns)
        self.assertEqual(record["players"]["Player1"]["CurrDamage"], 25.5)
        self.assertEqual(record["source_mtime_ns"], self.now_ns - 2_000_000_000)
        self.assertTrue(record["captured_at"].endswith("Z"))
        self.assertEqual(len(record["source_sha256"]), 64)
        self.assertEqual(set(record), {"captured_at", "source_mtime_ns",
                                       "source_sha256", "players"})

    def test_partial_rewrite_is_retried_and_valid_update_is_retained(self):
        self.write('{"Player1":')
        poller = TelemetryPoller(self.directory)
        self.assertIsNone(poller.poll(now_ns=self.now_ns))
        self.assertEqual(poller.stats.invalid, 1)
        self.write(mtime_ns=self.now_ns + 1)
        self.assertIsNotNone(poller.poll(now_ns=self.now_ns + 1))
        self.assertIsNone(poller.poll(now_ns=self.now_ns + 2))
        self.payload["Player1"]["CurrDamage"] = 31
        self.write(mtime_ns=self.now_ns + 3)
        self.assertEqual(poller.poll(now_ns=self.now_ns + 3)["players"]["Player1"]["CurrDamage"], 31)
        self.assertEqual(poller.stats.recorded, 2)

    def test_identical_payload_new_mtime_is_not_discarded(self):
        self.write()
        poller = TelemetryPoller(self.directory)
        first = poller.poll(now_ns=self.now_ns)
        self.assertIsNone(poller.poll(now_ns=self.now_ns))
        os.utime(self.source, ns=(self.now_ns + 1, self.now_ns + 1))
        rewritten = poller.poll(now_ns=self.now_ns + 1)
        self.assertIsNotNone(rewritten)
        self.assertEqual(first["players"], rewritten["players"])
        self.assertEqual(first["source_sha256"], rewritten["source_sha256"])
        self.assertNotEqual(first["source_mtime_ns"], rewritten["source_mtime_ns"])

    def test_rewrite_during_read_is_retried(self):
        self.write()
        original_read = Path.read_bytes

        def read_then_rewrite(path):
            data = original_read(path)
            if path == self.source:
                self.payload["Player1"]["x"] = 999
                self.write(mtime_ns=self.now_ns + 1)
            return data

        with patch.object(Path, "read_bytes", read_then_rewrite):
            result = read_snapshot(self.source, now_ns=self.now_ns)
        self.assertEqual(result.status, "unstable")
        result = read_snapshot(self.source, now_ns=self.now_ns + 1)
        self.assertEqual(result.snapshot.players["Player1"]["x"], 999)

    def test_payload_validation_rejects_invalid_fields(self):
        invalid_values = [
            ("x", True), ("x", "4"), ("y", float("inf")),
            ("CurrDamage", float("nan")), ("CurrDamage", -1),
            ("Eliminated", 1), ("BrawlhallaID", True),
            ("BrawlhallaID", -1), ("PlayerName", None),
        ]
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                raw = dict(self.payload["Player1"])
                raw[field] = value
                with self.assertRaises(ValueError):
                    validate_players({"Player1": raw})
        for payload in (None, [], {}, {"PlayerX": self.payload["Player1"]},
                        {"Player1": []}, {"Player1": {}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_players(payload)
        raw = dict(self.payload["Player1"], stocks=3, tick=99)
        normalized = validate_players({"Player1": raw})["Player1"]
        self.assertNotIn("stocks", normalized)
        self.assertNotIn("tick", normalized)

    def test_cli_records_jsonl_and_never_overwrites_output(self):
        self.write()
        output = self.directory / "records.jsonl"
        with patch("telemetry.time.time_ns", return_value=self.now_ns), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            result = main(["--directory", str(self.directory), "--output", str(output),
                           "--duration", "0.005", "--poll-seconds", "0.001"])
        self.assertEqual(result, 0)
        saved = output.read_bytes()
        records = [json.loads(line) for line in saved.splitlines()]
        self.assertEqual(len(records), 1)
        self.assertIn("invalid 0; stale 0", errors.getvalue())
        with self.assertRaises(FileExistsError):
            collect(self.directory, output, duration=0.001)
        self.assertEqual(output.read_bytes(), saved)

    def test_no_records_and_invalid_time_arguments_return_nonzero(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            result = main(["--directory", str(self.directory),
                           "--output", str(self.directory / "empty.jsonl"),
                           "--duration", "0.001"])
        self.assertEqual(result, 1)
        self.assertIn("No fresh valid", errors.getvalue())
        for option in ("--duration", "--poll-seconds", "--max-age-seconds"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(option=option, value=value), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main([option, value])
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
