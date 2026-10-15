"""Validate installation diagnostics with harmless fixture files."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from game_install import INSPECTED_SWF_SHA256, inspect_install


MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<application xmlns="http://ns.adobe.com/air/application/51.0">
  <versionNumber>1.0</versionNumber>
  <architecture>{architecture}</architecture>
  <initialWindow><content>{content}</content></initialWindow>
</application>
"""


class GameInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.swf = self.directory / "BrawlhallaAir.swf"
        self.executable = self.directory / "Brawlhalla.exe"
        self.manifest = self.directory / "META-INF" / "AIR" / "application.xml"
        self.manifest.parent.mkdir(parents=True)
        self.executable.write_bytes(b"Harmless fixture, never execute.")
        # More than one hash chunk exercises streaming without large fixtures.
        self.swf.write_bytes(b"FWS" + b"state-fixture" * 100_000)
        self.expected = hashlib.sha256(self.swf.read_bytes()).hexdigest()
        self.write_manifest()

    def write_manifest(self, *, architecture="64", content="BrawlhallaAir.swf"):
        self.manifest.write_text(
            MANIFEST.format(architecture=architecture, content=content), encoding="utf-8")

    def test_valid_install_matches_fingerprint_and_only_inspects_files(self):
        originals = {path: path.read_bytes()
                     for path in (self.executable, self.swf, self.manifest)}
        with patch("subprocess.Popen", side_effect=AssertionError("must not execute")), \
                patch("os.system", side_effect=AssertionError("must not execute")):
            report = inspect_install(self.directory, self.expected.upper())
        self.assertTrue(report["ready"])
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["directory"], str(self.directory.resolve()))
        self.assertEqual(report["missing"], [])
        self.assertTrue(report["swf_match"])
        self.assertEqual(report["checks"]["swf"]["sha256"], self.expected)
        self.assertEqual(report["checks"]["manifest"]["architecture"], "64")
        self.assertEqual(report["checks"]["manifest"]["content"], "BrawlhallaAir.swf")
        self.assertNotIn("version", report)
        json.dumps(report, allow_nan=False)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_without_expected_hash_fingerprint_is_unverified(self):
        report = inspect_install(self.directory)
        self.assertTrue(report["ready"])
        self.assertIsNone(report["swf_match"])
        self.assertEqual(report["checks"]["swf"]["sha256"], self.expected)

    def test_changed_swf_fails_expected_build_check(self):
        self.swf.write_bytes(b"different game build")
        report = inspect_install(self.directory, self.expected)
        self.assertFalse(report["ready"])
        self.assertFalse(report["swf_match"])
        self.assertEqual(report["checks"]["swf"]["status"], "mismatch")
        self.assertIn("version-specific", report["checks"]["swf"]["message"])
        self.assertEqual(len(INSPECTED_SWF_SHA256), 64)

    def test_missing_swf_and_executable_are_reported(self):
        self.swf.unlink()
        self.executable.unlink()
        report = inspect_install(self.directory, self.expected)
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"], "missing")
        self.assertCountEqual(report["missing"], ["BrawlhallaAir.swf", "Brawlhalla.exe"])
        self.assertIsNone(report["swf_match"])

    def test_wrong_architecture_or_entry_fails_manifest_check(self):
        for architecture, content, expected_message in (
                ("32", "BrawlhallaAir.swf", "architecture 64"),
                ("64", "other.swf", "entry content BrawlhallaAir.swf"),
                ("64", "./BrawlhallaAir.swf", "entry content BrawlhallaAir.swf")):
            with self.subTest(architecture=architecture, content=content):
                self.write_manifest(architecture=architecture, content=content)
                report = inspect_install(self.directory, self.expected)
                self.assertFalse(report["ready"])
                self.assertEqual(report["checks"]["manifest"]["status"], "invalid")
                self.assertIn(expected_message, report["checks"]["manifest"]["message"])

    def test_malformed_xml_and_wrong_namespace_have_useful_diagnostics(self):
        for xml, expected_message in (
                ("<broken", "Cannot parse AIR XML"),
                ('<application><architecture>64</architecture></application>',
                 "Adobe AIR namespace")):
            with self.subTest(xml=xml):
                self.manifest.write_text(xml, encoding="utf-8")
                report = inspect_install(self.directory, self.expected)
                self.assertFalse(report["ready"])
                self.assertEqual(report["checks"]["manifest"]["status"], "invalid")
                self.assertIn(expected_message, report["checks"]["manifest"]["message"])

    def test_unreadable_swf_is_diagnostic_instead_of_exception(self):
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path == self.swf:
                raise PermissionError("fixture access denied")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            report = inspect_install(self.directory, self.expected)
        self.assertFalse(report["ready"])
        self.assertEqual(report["checks"]["swf"]["status"], "unreadable")
        self.assertIn("access denied", report["checks"]["swf"]["message"])

    def test_missing_directory_and_invalid_expected_hash_fail_cleanly(self):
        report = inspect_install(self.directory / "not-installed", self.expected)
        self.assertFalse(report["ready"])
        self.assertEqual(len(report["missing"]), 3)
        report = inspect_install(self.directory, "not-a-sha256")
        self.assertFalse(report["ready"])
        self.assertEqual(report["checks"]["expected_swf_sha256"]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
