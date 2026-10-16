# Protocol 3: teammate setup and dual control

Protocol 3 extends the [session protocol](session-provider.md) with explicit teammate arrangements and controlled-player ownership. Current 2v2 session presets require it. The Python client and dual-actor environment are implemented; a live Windows state/input provider is still required.

## Extended profile

All configure and state acknowledgments must echo the existing eight session-profile fields plus:

```json
{"teammate_kind":"regular","teammate_account_id":null}
```

Kinds are `random`, `regular`, `ai`, or `bot` for 2v2, and `none` for 1v1. The account ID is null except for an optional nonnegative integer in regular mode. Omitting these fields is insufficient in protocol 3. Old saved eight-field settings can still be loaded by the menu, which supplies the documented defaults.

Random partners may change account identity between matches. Regular mode requires a reported `BrawlhallaID`; if no ID is configured, the first observed ID is pinned for the entire configured session. A later missing or changed ID fails the transition. This is a check of provider-reported data, not an independent authentication mechanism. Within-match identity continuity remains required for every arrangement.

## Hello and input ownership

For dual control, the provider responds to hello protocol 3 with:

```json
{"request_id":1,"protocol":3,"observation_schema":"brawlhalla-team-v2","mode":"online_controlled","game_build":"11.00.36","controlled_players":["Player1","Player2"],"capabilities":["state","actions","session","next_episode","team_setup","joint_actions","independent_inputs"],"input_routes":{"Player1":"keyboard:device-1","Player2":"controller:device-1"}}
```

`controlled_players` must contain exactly the requested AI slots, without duplicates; order is immaterial. An optional scalar `controlled_player` identifies the primary AI slot. In single-control random/regular/bot arrangements, only the primary AI slot belongs in the list. The human or game-bot teammate remains outside the AI's input ownership.

All providers require `state`, `actions`, `session`, `next_episode`, and `team_setup` capabilities. Dual AI additionally requires `joint_actions`, `independent_inputs`, and an `input_routes` mapping with one distinct nonempty route ID for each AI slot. Single-control providers may also report routing, which is validated when present.

Route IDs are provider-defined declarations, not drivers implemented by this client. The example represents a keyboard plus a secondary controller. A provider could instead route two controllers or separately addressable keyboards. It must verify that these routes reach distinct characters and can independently hold and release each character's inputs. Two aliases that feed the same character or merged action stream do not satisfy that requirement, even if their strings differ.

For your regular-teammate setup, the human keeps their keyboard and the AI can use a secondary controller route, for example `controlled_players: ["Player2"]` and `input_routes: {"Player2": "controller:device-1"}`. For two AI characters, both routes belong to the policy. Standard `pynput`/`SendInput` keyboard events do not provide per-keyboard device targeting. The game's `-multikeyboard` option and native keyboard-ID handling are evidence of multiple physical keyboard support, not proof that a Python input injector can impersonate two such devices.

The provider owns hardware/virtual-input setup and releases every controlled route after actions, on `release`, on disconnect, and on its watchdog timeout. The client prints the declared input routes but cannot verify device delivery from a hello response alone.

## Joint action

The shared policy receives two role-relative observations from one coherent world snapshot. Its decisions are sent in one request:

```json
{"command":"step_joint","request_id":4,"actions":{"Player1":["left","light"],"Player2":["right","jump"]},"hold_seconds":0.1}
```

The keys must exactly match the two controlled roles, and each value must be a supported semantic action from `controls.ACTIONS`. The provider applies both action streams together for that duration, releases their inputs, and returns one coherent state envelope. The response echoes the request ID, all profile fields, session ID and episode ID, and supplies a strictly greater simulation tick from that one joint world transition. Missing, extra, duplicated, sequentially substituted, or stale actor actions are not valid acknowledgments of a joint step.

Ordinary `step` is accepted only for a single controlled role. Dual AI cannot fall back to sending two ordinary steps, since that would let each character observe a different world transition.

## Shared learning and match lifecycle

The dual environment exposes two actor samples to one parameter-shared PPO policy. Each actor has its own self-relative features, reward tracker, and observation history. Both samples share the same team reward objective, simulation clock, and authoritative completion flags. This is a coupled multi-agent setting, not two independent Brawlhalla instances.

On match completion, both actor samples receive their own terminal observations. The client requests one new playable match and clears both histories together. It does not request a reset for each actor, and it never derives match completion from one character's elimination edge. Session statistics count that match once; PPO timesteps count two actor decisions per joint transition.

Fixed per-actor observation/action shapes allow a compatible team checkpoint to resume with one or two controlled actors. Shared weights provide a common starting point; separate inputs and observations let the characters act differently. This does not establish learned coordination until performance is evaluated in the actual game.
