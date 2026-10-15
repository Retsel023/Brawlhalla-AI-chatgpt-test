"""Record Brawlhalla's existing spectator JSON export without controlling the game."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
import time


PLAYER_KEY = re.compile(r"Player[0-9]+\Z")


def positive_seconds(value):
    """Argparse conversion shared by the three time limits."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive finite number") from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def validate_players(payload):
    """Return supported, validated fields; never infer missing game state."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("expected a nonempty object of Player<number> entries")
    players = {}
    for key, raw in payload.items():
        if not isinstance(key, str) or not PLAYER_KEY.fullmatch(key):
            raise ValueError("invalid player key")
        if not isinstance(raw, dict):
            raise ValueError("player data must be an object")
        try:
            player = {name: _finite_number(raw[name], name)
                      for name in ("x", "y", "CurrDamage")}
            eliminated = raw["Eliminated"]
        except KeyError:
            raise ValueError("missing required player field") from None
        if player["CurrDamage"] < 0:
            raise ValueError("CurrDamage must be nonnegative")
        if not isinstance(eliminated, bool):
            raise ValueError("Eliminated must be a boolean")
        player["Eliminated"] = eliminated
        if "BrawlhallaID" in raw:
            identity = raw["BrawlhallaID"]
            if isinstance(identity, bool) or not isinstance(identity, int) or identity < 0:
                raise ValueError("BrawlhallaID must be a nonnegative integer")
            player["BrawlhallaID"] = identity
        if "PlayerName" in raw:
            if not isinstance(raw["PlayerName"], str):
                raise ValueError("PlayerName must be a string")
            player["PlayerName"] = raw["PlayerName"]
        players[key] = player
    return players


@dataclass(frozen=True)
class Snapshot:
    mtime_ns: int
    size: int
    sha256: str
    players: dict

    @property
    def signature(self):
        return self.mtime_ns, self.size, self.sha256


@dataclass(frozen=True)
class ReadResult:
    status: str
    snapshot: Snapshot | None = None


def _stat_signature(info):
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def read_snapshot(path, max_age_seconds=3.0, *, now_ns=None):
    """Read a fresh, apparently stable file; failures are retryable results.

    The game overwrites this file in place. Before/after stat checks catch
    obvious concurrent writes, but cannot make the game's export atomic.
    """
    max_age_seconds = positive_seconds(max_age_seconds)
    path = Path(path)
    current_ns = time.time_ns() if now_ns is None else now_ns
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            return ReadResult("invalid")
        if current_ns - before.st_mtime_ns > max_age_seconds * 1_000_000_000:
            return ReadResult("stale")
        content = path.read_bytes()
        after = path.stat()
    except FileNotFoundError:
        return ReadResult("missing")
    except OSError:
        return ReadResult("invalid")
    if _stat_signature(before) != _stat_signature(after) or len(content) != after.st_size:
        return ReadResult("unstable")
    current_ns = time.time_ns() if now_ns is None else now_ns
    if current_ns - after.st_mtime_ns > max_age_seconds * 1_000_000_000:
        return ReadResult("stale")
    try:
        players = validate_players(json.loads(content.decode("utf-8")))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return ReadResult("invalid")
    return ReadResult("valid", Snapshot(after.st_mtime_ns, after.st_size,
                                        hashlib.sha256(content).hexdigest(), players))


@dataclass
class CollectionStats:
    recorded: int = 0
    invalid: int = 0
    stale: int = 0
    missing: int = 0
    unstable: int = 0
    unchanged: int = 0
    interrupted: bool = False


class TelemetryPoller:
    """Poll one source and retain identical-content rewrites with a new mtime."""

    def __init__(self, directory, max_age_seconds=3.0):
        self.source = Path(directory) / "PlayerStatuses.json"
        self.max_age_seconds = positive_seconds(max_age_seconds)
        self.last_signature = None
        self.stats = CollectionStats()

    def poll(self, *, now_ns=None):
        captured_ns = time.time_ns() if now_ns is None else now_ns
        result = read_snapshot(self.source, self.max_age_seconds, now_ns=captured_ns)
        if result.snapshot is None:
            setattr(self.stats, result.status, getattr(self.stats, result.status) + 1)
            return None
        snapshot = result.snapshot
        if snapshot.signature == self.last_signature:
            self.stats.unchanged += 1
            return None
        self.last_signature = snapshot.signature
        self.stats.recorded += 1
        captured_at = datetime.fromtimestamp(captured_ns / 1_000_000_000,
                                             timezone.utc).isoformat(timespec="microseconds")
        return {
            "captured_at": captured_at.replace("+00:00", "Z"),
            "source_mtime_ns": snapshot.mtime_ns,
            "source_sha256": snapshot.sha256,
            "players": snapshot.players,
        }


def collect(directory, output, *, duration=60.0, poll_seconds=0.1, max_age_seconds=3.0):
    """Write JSONL using exclusive creation and return aggregate polling counts."""
    duration = positive_seconds(duration)
    poll_seconds = positive_seconds(poll_seconds)
    poller = TelemetryPoller(directory, max_age_seconds)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        deadline = time.monotonic() + duration
        try:
            while True:
                record = poller.poll()
                if record is not None:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(poll_seconds, remaining))
        except KeyboardInterrupt:
            poller.stats.interrupted = True
    return poller.stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path.home() / "BrawlhallaStatsLive",
                        help="directory containing PlayerStatuses.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    parser.add_argument("--output", type=Path, default=Path("runs") / f"telemetry-{stamp}.jsonl",
                        help="new JSONL file; existing files are never overwritten")
    parser.add_argument("--duration", type=positive_seconds, default=60.0)
    parser.add_argument("--poll-seconds", type=positive_seconds, default=0.1)
    parser.add_argument("--max-age-seconds", type=positive_seconds, default=3.0)
    args = parser.parse_args(argv)
    try:
        stats = collect(args.directory, args.output, duration=args.duration,
                        poll_seconds=args.poll_seconds, max_age_seconds=args.max_age_seconds)
    except OSError as error:
        print(f"Cannot create or write telemetry output: {error}", file=sys.stderr)
        return 2
    print(f"Recorded {stats.recorded}; invalid {stats.invalid}; stale {stats.stale}; "
          f"missing {stats.missing}; unstable {stats.unstable}; unchanged {stats.unchanged}.",
          file=sys.stderr)
    if stats.recorded:
        print(f"Saved {args.output}")
    else:
        print("No fresh valid player statuses were recorded. This collector reads the "
              "existing spectator export; it does not enable telemetry or control the game.",
              file=sys.stderr)
    if stats.interrupted:
        return 130
    return 0 if stats.recorded else 1


if __name__ == "__main__":
    raise SystemExit(main())
