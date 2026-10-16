"""Local HTTP fixtures validate command proposals without calling an LLM or game."""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import threading
import time
import unittest
from unittest.mock import patch

from language_commands import CommandError, interpret_command
from session_profiles import SessionOptions, make_profile


class OllamaFixture:
    """Bounded loopback HTTP peer recording every request, including redirects."""

    def __init__(self, output=None, *, envelope=None, raw=None, status=200,
                 delay=0, location=None, content_length=True, drip_interval=0):
        self.output = ({"message": "Suggested settings", "needs_clarification": False, "patch": {}}
                       if output is None else output)
        self.envelope = envelope
        self.raw = raw
        self.status = status
        self.delay = delay
        self.location = location
        self.content_length = content_length
        self.drip_interval = drip_interval
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.respond()

            def do_POST(self):
                self.respond()

            def respond(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                owner.requests.append({"method": self.command, "path": self.path,
                                       "body": json.loads(body) if body else None})
                if owner.delay:
                    time.sleep(owner.delay)
                if owner.raw is not None:
                    response = owner.raw
                else:
                    content = owner.output if isinstance(owner.output, str) else json.dumps(owner.output)
                    envelope = owner.envelope if owner.envelope is not None else {
                        "model": "fixture-model", "done": True,
                        "message": {"role": "assistant", "content": content},
                    }
                    response = json.dumps(envelope).encode("utf-8")
                try:
                    self.send_response(owner.status)
                    self.send_header("Content-Type", "application/json")
                    if owner.location is not None:
                        self.send_header("Location", owner.location)
                    if owner.content_length:
                        self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    if owner.drip_interval:
                        for offset in range(0, len(response), 4):
                            self.wfile.write(response[offset:offset + 4])
                            self.wfile.flush()
                            time.sleep(owner.drip_interval)
                    else:
                        self.wfile.write(response)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def config(self, **changes):
        settings = {"endpoint": self.endpoint, "model": "fixture-model",
                    "timeout_seconds": 0.3, "max_response_bytes": 4096}
        settings.update(changes)
        return {"language_commands": settings}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


class LanguageCommandTests(unittest.TestCase):
    def serve(self, output=None, **settings):
        fixture = OllamaFixture(output, **settings)
        self.addCleanup(fixture.close)
        return fixture

    def options(self, **overrides):
        settings = {"profile": make_profile("online_teams", self_player="Player7",
                                            teammate_player="Player12", opponents=["Player3", "Player4"],
                                            friendly_fire=True, teammate_kind="regular", teammate_account_id=777),
                    "learning": True, "steps": 123, "checkpoint_interval": 17,
                    "model_root": "models/custom", "initialize_from": "models/seed.zip",
                    "device": "cpu", "max_seconds": 9,
                    "legend": "Hattori", "guidance": {"opponent_spacing": "mid",
                                                         "teammate_spacing": "near",
                                                         "strength": 0.4, "combo": ["light", "heavy"]}}
        settings.update(overrides)
        return SessionOptions(**settings)

    def output(self, changes, **overrides):
        value = {"message": "Settings proposal", "needs_clarification": False, "patch": changes}
        value.update(overrides)
        return value

    def test_chat_request_is_nonstreaming_schema_constrained_and_contains_user_text(self):
        fixture = self.serve(self.output({"guidance": {"opponent_spacing": "far"}}))
        original = self.options()
        text = "Keep more distance from the opponents."
        result = interpret_command(text, original, fixture.config())
        self.assertTrue(result.changed)
        self.assertFalse(result.needs_clarification)
        self.assertEqual(result.options.guidance["opponent_spacing"], "far")
        self.assertEqual(len(fixture.requests), 1)
        request = fixture.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/api/chat")
        payload = request["body"]
        self.assertEqual(payload["model"], "fixture-model")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertIsInstance(payload["format"], dict)
        self.assertEqual(payload["format"]["type"], "object")
        self.assertIs(payload["format"]["additionalProperties"], False)
        self.assertEqual(set(payload["format"]["required"]), {"message", "needs_clarification", "patch"})
        self.assertTrue(any(message["role"] == "user" and text in message["content"]
                            for message in payload["messages"]))
        system = next(message["content"] for message in payload["messages"] if message["role"] == "system")
        self.assertIn("only to the primary AI player", system)
        self.assertIn("requesting two legends, is unsupported", system)
        self.assertIn("both AI actors from their own perspectives", system)
        self.assertIn("per-actor tactical instructions are unsupported", system)

    def test_setup_and_tactics_are_applied_together_to_a_valid_session_proposal(self):
        fixture = self.serve(self.output({"profile": "online_teams", "match_format": "2v2",
                                         "teammate_kind": "ai", "learning": True, "legend": "Asuri",
                                         "guidance": {"opponent_spacing": "close", "teammate_spacing": "spread",
                                                      "strength": 0.8, "combo": ["light", "heavy"]}}))
        original = SessionOptions(make_profile("bot_training"), learning=False)
        result = interpret_command("Use Asuri with an AI teammate and pressure them with light then heavy.",
                                   original, fixture.config())
        self.assertTrue(result.changed)
        self.assertEqual(result.options.profile["profile"], "online_teams")
        self.assertEqual(result.options.profile["match_format"], "2v2")
        self.assertEqual(result.options.profile["teammate_kind"], "ai")
        self.assertTrue(result.options.profile["online"])
        self.assertEqual(result.options.profile["opponent_kind"], "human")
        self.assertTrue(result.options.learning)
        self.assertEqual(result.options.legend, "Asuri")
        self.assertEqual(result.options.guidance["combo"], ["light", "heavy"])
        self.assertEqual(original.profile["profile"], "bot_training")

    def test_untouched_options_and_explicit_roles_are_preserved_without_mutating_input(self):
        original = self.options()
        before = copy.deepcopy(original.to_dict())
        fixture = self.serve(self.output({"learning": False, "legend": "Asuri"}))
        result = interpret_command("Stop learning and select Asuri.", original, fixture.config())
        expected = copy.deepcopy(before)
        expected.update(learning=False, legend="Asuri")
        self.assertEqual(result.options.to_dict(), expected)
        self.assertEqual(original.to_dict(), before)
        self.assertIsNot(result.options, original)

    def test_partial_guidance_merges_with_all_existing_guidance(self):
        original = self.options()
        fixture = self.serve(self.output({"guidance": {"opponent_spacing": "far"}}))
        result = interpret_command("Stay farther away.", original, fixture.config())
        expected = copy.deepcopy(original.to_dict())
        expected["guidance"]["opponent_spacing"] = "far"
        self.assertEqual(result.options.to_dict(), expected)

    def test_profile_changes_use_destination_learning_defaults_unless_explicitly_overridden(self):
        cases = [
            ("online_teams", True, {"profile": "human_duel"}, False),
            ("human_duel", False, {"profile": "bot_training"}, True),
            ("human_duel", False, {"profile": "bot_training", "learning": False}, False),
            ("bot_training", True, {"profile": "human_duel", "learning": True}, True),
            ("human_duel", True, {"profile": "human_duel"}, True),
            ("bot_training", False, {"guidance": {"opponent_spacing": "far"}}, False),
        ]
        for source, learning, changes, expected in cases:
            with self.subTest(source=source, learning=learning, changes=changes):
                original = SessionOptions(make_profile(source), learning=learning)
                before = copy.deepcopy(original.to_dict())
                fixture = self.serve(self.output(changes))
                result = interpret_command("Change the selected training setup.", original, fixture.config())
                self.assertFalse(result.needs_clarification)
                self.assertEqual(result.options.profile["profile"], changes.get("profile", source))
                self.assertIs(result.options.learning, expected)
                self.assertEqual(original.to_dict(), before)

    def test_empty_patch_and_identical_patch_report_no_change(self):
        original = self.options()
        for changes in ({}, {"legend": "Hattori", "learning": True},
                        {"guidance": {"opponent_spacing": "mid"}}):
            with self.subTest(changes=changes):
                fixture = self.serve(self.output(changes))
                result = interpret_command("Keep the current settings.", original, fixture.config())
                self.assertFalse(result.changed)
                self.assertFalse(result.needs_clarification)
                self.assertEqual(result.options.to_dict(), original.to_dict())

    def test_clarification_is_atomic_even_when_model_proposes_a_partial_change(self):
        original = self.options()
        for changes in ({}, {"learning": False, "guidance": {"opponent_spacing": "far"}}):
            with self.subTest(changes=changes):
                fixture = self.serve(self.output(changes, message="Which opponent do you mean?",
                                                needs_clarification=True))
                result = interpret_command("Change it for that player.", original, fixture.config())
                self.assertTrue(result.needs_clarification)
                self.assertFalse(result.changed)
                self.assertEqual(result.options.to_dict(), original.to_dict())
                self.assertTrue(result.message.startswith("No settings changed. "))
                self.assertIn("Which opponent do you mean?", result.message)

    def test_teammate_spacing_without_a_teammate_requests_clarification_atomically(self):
        for source in ("human_duel", "online_teams"):
            for spacing in ("near", "spread"):
                with self.subTest(source=source, spacing=spacing):
                    original = SessionOptions(make_profile(source), learning=False, legend="Hattori")
                    before = copy.deepcopy(original.to_dict())
                    changes = {"profile": "human_duel", "learning": True, "legend": "Asuri",
                               "guidance": {"opponent_spacing": "far", "teammate_spacing": spacing,
                                            "strength": 0.8, "combo": ["light", "heavy"]}}
                    fixture = self.serve(self.output(changes))
                    result = interpret_command("Play a duel but keep this teammate spacing.",
                                               original, fixture.config())
                    self.assertTrue(result.needs_clarification)
                    self.assertFalse(result.changed)
                    self.assertEqual(result.options.to_dict(), before)
                    self.assertEqual(original.to_dict(), before)

    def test_model_text_cannot_claim_game_actions_were_executed(self):
        claimed = "I pressed heavy, won your match, and changed the running game."
        fixture = self.serve(self.output({"learning": False}, message=claimed))
        result = interpret_command("Stop learning.", self.options(), fixture.config())
        self.assertTrue(result.changed)
        self.assertIsInstance(result.message, str)
        self.assertTrue(result.message.strip())
        self.assertNotIn(claimed, result.message)
        self.assertNotIn("won your match", result.message)

    def test_unknown_or_invalid_output_fields_fail_instead_of_being_silently_applied(self):
        invalid = [
            {"message": "ok", "needs_clarification": False, "patch": {}, "execute": "heavy"},
            {"message": "ok", "needs_clarification": False},
            {"message": 42, "needs_clarification": False, "patch": {}},
            {"message": "ok", "needs_clarification": 0, "patch": {}},
            {"message": "ok", "needs_clarification": False, "patch": []},
            self.output({"unknown": True}), self.output({"self_player": "Player9"}),
            self.output({"learning": 1}), self.output({"legend": 12}),
            self.output({"profile": []}), self.output({"teammate_kind": "unknown"}),
        ]
        original = self.options()
        before = copy.deepcopy(original.to_dict())
        for output in invalid:
            with self.subTest(output=output):
                fixture = self.serve(output)
                with self.assertRaises(CommandError):
                    interpret_command("Change settings.", original, fixture.config())
                self.assertEqual(original.to_dict(), before)
                self.assertEqual(len(fixture.requests), 1)

    def test_invalid_guidance_fields_and_combinations_are_rejected(self):
        invalid = [None, [], {"opponent_spacing": "very far"}, {"teammate_spacing": "behind"},
                   {"strength": True}, {"strength": -0.1}, {"strength": 1.1},
                   {"combo": "light,heavy"}, {"combo": ["not-an-action"]},
                   {"combo": ["light"] * 7}, {"combo": [1]},
                   {"opponent_spacing": "far", "extra": "ignored"}]
        for guidance in invalid:
            with self.subTest(guidance=guidance):
                fixture = self.serve(self.output({"guidance": guidance}))
                with self.assertRaises(CommandError):
                    interpret_command("Change tactics.", self.options(), fixture.config())

    def test_invalid_final_profile_or_unknown_legend_is_rejected(self):
        invalid = [{"profile": "unknown"}, {"match_format": "3v3"},
                   {"profile": "online_teams", "match_format": "1v1"},
                   {"profile": "online_duel", "teammate_kind": "ai"},
                   {"legend": "Definitely Not A Brawlhalla Legend"}]
        for changes in invalid:
            with self.subTest(changes=changes):
                fixture = self.serve(self.output(changes))
                with self.assertRaises(CommandError):
                    interpret_command("Change the match.", self.options(), fixture.config())

    def test_duplicate_json_fields_and_nonfinite_numbers_are_refused(self):
        outputs = [
            '{"message":"a","message":"b","needs_clarification":false,"patch":{}}',
            '{"message":"ok","needs_clarification":false,"patch":{"learning":true,"learning":false}}',
            '{"message":"ok","needs_clarification":false,"patch":{"guidance":{"strength":NaN}}}',
            '{"message":"ok","needs_clarification":false,"patch":{"guidance":{"strength":Infinity}}}',
            '{"message":"ok","needs_clarification":false,"patch":{"guidance":{"strength":1e999}}}',
            '```json\n{"message":"ok","needs_clarification":false,"patch":{}}\n```',
            '{"message":"ok","needs_clarification":false,"patch":{}} trailing text',
        ]
        for output in outputs:
            with self.subTest(output=output):
                fixture = self.serve(output)
                with self.assertRaises(CommandError):
                    interpret_command("Change tactics.", self.options(), fixture.config())

    def test_invalid_or_incomplete_ollama_envelopes_are_reported(self):
        content = json.dumps(self.output({}))
        invalid = [[], {}, {"done": False, "message": {"role": "assistant", "content": content}},
                   {"done": True, "message": {"role": "user", "content": content}},
                   {"done": True, "message": {"role": "assistant", "content": {}}},
                   {"done": True, "error": "model failed"},
                   {"done": True, "error": "model failed",
                    "message": {"role": "assistant", "content": content}}]
        for envelope in invalid:
            with self.subTest(envelope=envelope):
                fixture = self.serve(envelope=envelope)
                with self.assertRaises(CommandError):
                    interpret_command("Train.", self.options(), fixture.config())
        for raw in (b"not json", b'{"done":true,"done":true,"message":{}}',
                    b'{"done":true,"message":{"role":"assistant","content":"{}"},"other":NaN}'):
            with self.subTest(raw=raw):
                fixture = self.serve(raw=raw)
                with self.assertRaises(CommandError):
                    interpret_command("Train.", self.options(), fixture.config())

    def test_http_errors_and_unavailable_local_server_are_reported_without_retry(self):
        for status in (404, 500):
            with self.subTest(status=status):
                fixture = self.serve(status=status)
                with self.assertRaises(CommandError):
                    interpret_command("Train.", self.options(), fixture.config())
                self.assertEqual(len(fixture.requests), 1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        config = {"language_commands": {"endpoint": f"http://127.0.0.1:{port}", "model": "fixture-model",
                                         "timeout_seconds": 0.05, "max_response_bytes": 4096}}
        with self.assertRaises(CommandError):
            interpret_command("Train.", self.options(), config)

    def test_http_response_is_bounded_with_and_without_content_length(self):
        for declared_length in (True, False):
            with self.subTest(declared_length=declared_length):
                fixture = self.serve(raw=b" " * 4097, content_length=declared_length)
                with self.assertRaises(CommandError):
                    interpret_command("Train.", self.options(), fixture.config(max_response_bytes=4096))
                self.assertEqual(len(fixture.requests), 1)

    def test_http_timeout_is_bounded_and_not_retried(self):
        fixture = self.serve(delay=0.2)
        started = time.monotonic()
        with self.assertRaises(CommandError):
            interpret_command("Train.", self.options(), fixture.config(timeout_seconds=0.04))
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertEqual(len(fixture.requests), 1)

    def test_absolute_deadline_stops_a_body_that_keeps_arriving_before_idle_timeout(self):
        fixture = self.serve(drip_interval=0.015)
        started = time.monotonic()
        with self.assertRaises(CommandError):
            interpret_command("Train.", self.options(), fixture.config(timeout_seconds=0.06))
        self.assertLess(time.monotonic() - started, 0.6)
        self.assertEqual(len(fixture.requests), 1)

    def test_remote_and_decorated_endpoints_are_refused_before_any_request(self):
        fixture = self.serve()
        endpoints = ["http://example.com:11434", "http://127.0.0.1.evil.test:11434",
                     "http://127.0.0.2:11434", "https://127.0.0.1:11434", "file:///tmp/model",
                     "http://user:pass@127.0.0.1:11434", fixture.endpoint + "/api/chat",
                     fixture.endpoint + "?query=1", fixture.endpoint + "#fragment",
                     fixture.endpoint + "?", fixture.endpoint + "#",
                     "http://2130706433:11434", "http://localhost.evil.test:11434"]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises((CommandError, ValueError)):
                    interpret_command("Train.", self.options(), fixture.config(endpoint=endpoint))
        self.assertEqual(fixture.requests, [])

    def test_loopback_requests_ignore_environment_proxy_settings(self):
        fixture = self.serve(self.output({"learning": False}))
        proxy_environment = {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1",
                             "HTTPS_PROXY": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1",
                             "ALL_PROXY": "http://127.0.0.1:1", "all_proxy": "http://127.0.0.1:1",
                             "NO_PROXY": "", "no_proxy": ""}
        with patch.dict(os.environ, proxy_environment):
            result = interpret_command("Stop learning.", self.options(), fixture.config())
        self.assertTrue(result.changed)
        self.assertEqual(len(fixture.requests), 1)

    def test_redirect_is_rejected_without_contacting_its_target(self):
        target = self.serve(self.output({"learning": False}))
        source = self.serve(status=302, location=target.endpoint + "/api/chat")
        with self.assertRaises(CommandError):
            interpret_command("Train.", self.options(), source.config())
        self.assertEqual(len(source.requests), 1)
        self.assertEqual(target.requests, [])


if __name__ == "__main__":
    unittest.main()
