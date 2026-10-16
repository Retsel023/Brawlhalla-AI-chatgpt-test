# Protocol 2: session provider

This is the contract for a separately implemented Windows game-side provider. It is not a Brawlhalla network endpoint. The Python client is implemented; live instrumentation, input routing, bot setup, and online matchmaking remain provider work.

The current CLI uses protocol 2 for duel sessions and [protocol 3](team-control-provider.md) for 2v2 teammate selection and optional dual-character control. This document remains the underlying single-control lifecycle and the legacy protocol-2 API.

Transport is the same bounded, loopback-only JSONL transport as [protocol 1](state-bridge.md). Each response echoes the exact integer `request_id`. The protocol-1 client and commands remain supported separately; a session client requires protocol 2.

## Hello

Request:

```json
{"command":"hello","request_id":1,"protocol":2,"observation_schema":"brawlhalla-team-v2"}
```

Response for offline sessions:

```json
{"request_id":1,"protocol":2,"observation_schema":"brawlhalla-team-v2","mode":"offline_controlled","game_build":"11.00.36","controlled_player":"Player1","capabilities":["state","actions","session","next_episode"]}
```

Online sessions require `mode: "online_controlled"`. The declared build must match configuration, and `controlled_player` must match the selected AI slot. The client trusts this declaration as a contract; Windows validation must establish that actions really affect that player.

## Configure a session

```json
{"command":"configure","request_id":2,"profile":"online_teams","match_format":"2v2","opponent_kind":"human","online":true,"self_player":"Player1","teammate_player":"Player2","opponents":["Player3","Player4"],"friendly_fire":false}
```

The response supplies a nonempty `session_id` and echoes all eight profile fields exactly. Profile names and constraints are defined in `session_profiles.py`; the provider must arrange or verify that real session, not merely echo configuration without applying it. A configured client cannot switch profiles. A mode change creates a new client.

## Wait for a playable match

```json
{"command":"next_episode","request_id":3,"session_id":"session-001","after_episode_id":null,"seed":null,"profile":"online_teams","match_format":"2v2","opponent_kind":"human","online":true,"self_player":"Player1","teammate_player":"Player2","opponents":["Player3","Player4"],"friendly_fire":false}
```

The provider waits until the appropriate new match is ready, or reports an error. For bot sessions it can establish a new bot match; for human sessions it must respect the actual match lifecycle. This command does not grant a way to restart somebody else's match. `seed` can be used where the game supports reproducible setup; a provider must document limitations rather than pretend resets are deterministic.

The response contains:

- The echoed request ID, `session_id`, and all eight profile fields.
- A nonempty `episode_id` not previously used in that session, plus a nonnegative simulation `tick`.
- `players`, with exactly the configured two or four player keys. Fields follow `PlayerStatuses.json`: `x`, `y`, `CurrDamage`, `Eliminated`, optional `BrawlhallaID` and `PlayerName`.
- Explicit `terminated: false` and `truncated: false`. All selected players must be active initially.

For later matches, `after_episode_id` identifies the previous match. The provider must never return the same match as a successful new episode. Player/account identities may change between episodes, but not silently during a match. Role ordering and team assignments remain exactly as configured.

## Step and match completion

`step` is unchanged: a semantic action list from `controls.ACTIONS` plus positive `hold_seconds`. The response is a coherent state envelope containing the session/profile fields above, the same episode ID, and a strictly greater simulation tick. Only confirmed post-action observations become learning transitions.

`terminated` identifies actual match completion established by the provider. `truncated` identifies an external cutoff. The client never derives these flags from damage dropping, eliminated-state edges, or a file timestamp. After completion, the next operation is `next_episode`; there is no legacy `reset` command in this session protocol.

The normal response timeout defaults to three seconds. `next_episode` has a separate configurable wait limit of 120 seconds in the application config. Both waits poll stop requests at most every 0.1 seconds. Commands are not replayed on timeout or connection failure.

## Input ownership

The provider must release keys after each step, on `release`, on disconnect, and on its own watchdog timeout. A human-versus-AI session needs distinct input routing for the human and controlled AI slot. The client sends `release` on close, but a broken connection cannot guarantee delivery. A force-ended menu subprocess relies on the provider watchdog.

The provider must not treat spectator JSON as confirmation that actions or match transitions were applied. That export has no action acknowledgments and its verified path in this build is spectator-only. A synthetic provider exists in tests solely to verify protocol, training, transfer, and session orchestration.
