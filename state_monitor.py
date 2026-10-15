"""Read-only structured observations and reward diagnostics from spectator files."""

from dataclasses import asdict
import json
import time

from game_state import GameState, RewardTracker
from telemetry import TelemetryPoller, positive_seconds


class StateFeedError(RuntimeError):
    pass


def monitor(directory, *, self_slot, opponent_slot, duration=60.0,
            timeout=3.0, poll_seconds=0.1, position_scale=1000.0,
            damage_scale=100.0, damage_reward_scale=0.01, emit=None):
    """Emit derived data without pressing keys or updating a policy.

    Reward projections describe what happened to the selected players. They
    do not establish which actions caused it. File times are not game ticks.
    """
    duration = positive_seconds(duration)
    timeout = positive_seconds(timeout)
    poll_seconds = positive_seconds(poll_seconds)
    emit = emit or (lambda record: print(json.dumps(record, allow_nan=False), flush=True))
    poller = TelemetryPoller(directory, max_age_seconds=timeout)
    tracker = RewardTracker(damage_scale=damage_reward_scale)
    started = time.monotonic()
    deadline = started + duration
    last_update = started
    count = 0
    while True:
        record = poller.poll()
        if record is not None:
            try:
                state = GameState.from_players(record['players'], self_slot, opponent_slot)
                vector = state.to_vector(position_scale=position_scale, damage_scale=damage_scale)
                projection = tracker.update(state)
            except ValueError as error:
                raise StateFeedError(f'Cannot observe the selected players: {error}') from error
            emit({**record, 'source': 'spectator', 'self_player': self_slot,
                  'opponent_player': opponent_slot, 'observation': vector.tolist(),
                  'reward_projection': asdict(projection)})
            last_update = time.monotonic()
            count += 1
        now = time.monotonic()
        if now - last_update >= timeout:
            raise StateFeedError(
                f'No fresh valid player update for {timeout:g}s in {poller.source}. '
                'Use -writestats and a spectated private match to check the built-in export. '
                'This build has no verified export path for the controlled offline player.')
        if now >= deadline:
            break
        time.sleep(min(poll_seconds, deadline - now))
    if count == 0:
        raise StateFeedError(f'No usable player observations found in {poller.source}.')
    return poller.stats
