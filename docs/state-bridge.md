# Controlled game-state bridge

This is a **project-defined protocol**, not a built-in Brawlhalla server. The Python client and Gym environment are implemented and tested. A Windows provider that obtains live offline game state, applies actions, and resets the game is still required. The shipped spectator JSON export cannot implement this interface by itself.

The agent connects to a local provider using TCP, default `127.0.0.1:8765`. Settings live under `bridge` in `config.json`. Requests and responses are UTF-8 JSON objects, one per line. Each request has a strictly increasing integer `request_id`; its response must echo that exact ID. Only one request is outstanding. Responses are bounded to 1 MiB and three seconds by default. There are no automatic command retries.

## Handshake

Client:

```json
{"command":"hello","request_id":1,"protocol":1,"observation_schema":"brawlhalla-players-v1"}
```

Provider:

```json
{"request_id":1,"protocol":1,"observation_schema":"brawlhalla-players-v1","mode":"offline_controlled","game_build":"11.00.36","controlled_player":"Player1","capabilities":["state","actions","reset"]}
```

`controlled_player` must identify the player whose inputs the provider actually applies. It must match the agent's explicit `--self-player` setting. Merely observing another player is insufficient. The expected game build is checked against configuration. These declarations are a provider contract; they do not prove the game-side integration works. Validate it on the Windows host.

## Reset

Client:

```json
{"command":"reset","request_id":2,"seed":null}
```

Provider:

```json
{"request_id":2,"episode_id":"match-001-attempt-1","tick":0,"players":{"Player1":{"BrawlhallaID":10,"PlayerName":"Agent","x":-100,"y":20,"CurrDamage":0,"Eliminated":false},"Player2":{"BrawlhallaID":20,"PlayerName":"Opponent","x":100,"y":20,"CurrDamage":0,"Eliminated":false}},"terminated":false,"truncated":false}
```

The provider must restore a real playable scenario, release inputs, and return a coherent initial state with both selected players active. Each reset must produce a new nonempty `episode_id`, including resets requested before a match ends. `seed` is an integer or null; if the game cannot seed deterministic resets, document that limitation in the provider rather than pretending otherwise.

The `players` entries use fields observed in `PlayerStatuses.json`; the surrounding envelope supplies metadata absent from the built-in export. `x/y` use the same stage-centered coordinate convention. `tick` is a nonnegative simulation counter, not a file timestamp. Account/name identity fields are optional, but must remain consistent within an episode when supplied.

## Step

Client:

```json
{"command":"step","request_id":3,"action":["left","light"],"hold_seconds":0.1}
```

The provider applies the action to the declared controlled player for the requested duration and releases the keys. It then replies with the same state envelope as reset, echoing request ID 3. The snapshot must reflect simulation after that action, retain the same episode ID, and strictly increase `tick`. Arbitrary snapshots from a running spectator feed cannot establish that relationship.

Actions come from `controls.ACTIONS`: idle, directional movement, jump, crouch, light/heavy attack, dodge, and selected combinations. The bridge receives semantic key names; its own binding layer maps those names to the configured local game controls.

The Python environment normalizes ten features in this order:

```text
self_x, self_y, self_damage, self_eliminated,
opponent_x, opponent_y, opponent_damage, opponent_eliminated,
relative_x, relative_y
```

Positions and relative offsets are divided by `state.position_scale` (default 1000), damage by `state.damage_scale` (default 100). These are divisors, not inferred stage bounds. PPO uses `MlpPolicy` with a separate state checkpoint. Rewards are scaled changes in opponent damage minus changes in own damage. Ambiguous damage decreases give zero reward and reset the damage baseline; no remaining-stock count is inferred.

`terminated` must represent a genuine terminal condition established by the provider. `truncated` represents an external episode cutoff, such as a training time limit. Neither comes from file age or the diagnostic elimination-edge flag. When either is true, the next operation must reset. Stable-Baselines3 automatically invokes reset when episodes end.

## Errors and input cleanup

The provider can reply with `{"request_id":3,"error":"reason"}`. Missing/invalid fields, invalid player identities, stale ticks, wrong episodes, timeout, disconnection, and incompatible build/schema cause the client to stop rather than return a training transition.

On close or failure, the client attempts a bounded final message:

```json
{"command":"release","request_id":4}
```

It then closes the socket without waiting for a reply. **The provider must release all inputs after each step, on `release`, on disconnect, and on its own watchdog timeout.** A broken socket cannot guarantee delivery of a release request. The provider must not leave a key held while waiting for the next client command. Ctrl+C interrupts the client and triggers cleanup/checkpoint save; a socket failure also stops the rollout.

The tests contain a synthetic provider for protocol and PPO validation. It generates artificial observations and is not a Brawlhalla integration or a substitute for live testing.
