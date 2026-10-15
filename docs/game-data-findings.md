# Direct game data: findings from the uploaded build

The upload contains analyzable game code. Structured state would be a better training input than screenshots, but a fast, usable state connection for a controlled player still needs to be established.

## What was inspected

Inspected commit `a89e51c45aa1f86ad25ca7309e925ef64f8d0b50`, before removal of identical root asset copies. The canonical `Brawlhalla/` folder is unchanged by that cleanup. Relevant files:

- `Brawlhalla/META-INF/AIR/application.xml`: 64-bit Adobe AIR 51 desktop application, entry point `BrawlhallaAir.swf`.
- `Brawlhalla/BrawlhallaAir.swf`: compressed SWF containing ActionScript bytecode. Its own build strings report `11.00.36`.
- `Engine.swz`, `Game.swz`, `Init.swz`, and `Dynamic.swz`: packaged resources handled by the `RawData` native extension.

Static inspection used JPEXS FFDec 26.3.0, downloaded from its official GitHub release and checked against that release's SHA-256 digest. All 865 script exports completed. Analysis outputs are outside the repository at `/workspace/game-analysis/`; game executables and native libraries were not run or changed. Decompiled method names are build-specific evidence, not a stable supported API.

## Actual built-in JSON export

The command-line parser recognizes `-writestats`. The exporter writes files beneath AIR's `File.userDirectory/BrawlhallaStatsLive`, normally `%USERPROFILE%\BrawlhallaStatsLive` on Windows.

`PlayerStatuses.json` maps keys such as `Player1` to:

| Field | Meaning observed in the code |
| --- | --- |
| `BrawlhallaID` | Player account identifier |
| `PlayerName` | Display name |
| `Eliminated` | Whether the entity is in the eliminated state |
| `CurrDamage` | Current damage, rounded to two decimal places |
| `x`, `y` | Position relative to the center of the stage's referenced rectangle |

Other export paths include `Deaths.json` (`DeathList`, with `Time`, `Victim`, optional `Killer`, and optional rollback marking), loadouts, map picks, and end-of-game statistics. These files are written separately; they are not one synchronized snapshot.

**The important limitation is activation.** In this uploaded build, the code path that activates live export is the spectator-loading completion handler. `-writestats` sets the requested-export flag; that alone does not prove the feed runs in offline training, replay playback, or a match where the client controls a player. No such activation path was established in this inspection.

The position/damage writer normally runs after the opening 6,000 game-time units, then when more than 1,000 game-time units have passed since its previous update. Damage-related changes can trigger an earlier write. This is approximately one update per second during ordinary real-time play, with event-triggered additions; it is not a per-frame control interface.

The player-status file contains no simulation timestamp, tick, match ID, remaining-stock count, velocity, grounded state, weapon, or attack state. Files are overwritten directly rather than atomically replaced, and write exceptions are swallowed. A reader must tolerate missing, empty, and partially written files. Identical-content rewrites can be useful cadence evidence; content hashes alone should not deduplicate them.

### Static trace for follow-up investigation

In the decompiled scripts outside the repository:

- `§_-i2H§.as`, `§_-k4K§`: `-writestats` sets `§_-v8§`.
- `§_-h3U§.as`, `§_-34G§` / `§_-K4A§`: spectator entry/completion, then `§_-A1S§ = §_-v8§`. Training and replay use other states.
- `§_-s3m§.as`, `§_-a1k§`: player-status construction; `§_-A16§`: JSON file writing; `Tick`: cadence and export gating; `§_-S48§`: death events.
- `GameStats.as`: damage changes can request an early player-status update.

## Validate the files on Windows

Add `-writestats` to Brawlhalla's Steam launch options, then restart the game. Use a private match you can spectate to exercise the path identified above. A local training session can be checked separately, but an existing file is not evidence that the training session is refreshing it.

Run the collector from this repository using Python 3.12 or later; it has no third-party dependencies:

```powershell
py -3.12 telemetry.py --duration 60
```

For a different export directory:

```powershell
py -3.12 telemetry.py --directory "C:\path\to\BrawlhallaStatsLive" --duration 60
```

It reads `PlayerStatuses.json` and records fresh valid observations in a new JSONL file under `runs/`. Records include the source modification time, content hash, and local UTC capture time. Those times are provenance, not simulation ticks. The collector never controls or modifies the game. It reports zero usable records as a failure rather than implying the feed works.

Record a short session with movement and damage, then inspect the actual fields, update intervals, and behavior when spectating stops. Compare that with offline training before using the output to drive learning. Sample payload tests in the cloud validate the collector, not the game's export behavior.

## Other useful discoveries

The game has internal entity fields for positions, damage, and remaining stocks. In this build, damage is `Entity.§_-W22§`; `§_-p3V§()` and `§_-Sb§()` expose position through indexed numeric storage; `Entity.§_-V§` is initialized from stock settings and decremented on death. These are internal symbols, not verified process-memory addresses or a working Python state provider.

Replay records include timestamped input bitmasks and enforce version compatibility. Inputs alone do not provide the per-frame world state; recovering trajectories requires the matching game simulation. The literal `advanced-telemetry` appears in a compile-flag listing, which does not establish a usable launch option.

`RawData.GetData()` is used to parse resource packages into XML/text. Its name does not indicate an external live-match data feed.

## What remains for autonomous training

The agent now has a structured-state pipeline: `agent.py observe` reads the real spectator files, encodes ten normalized features, and reports automatic damage-reward projections. `agent.py train --source bridge` selects an MLP policy and uses a strict action/state/reset protocol. See the [bridge contract](state-bridge.md). The game-side provider for controlled play remains unimplemented, so spectator observations are deliberately not offered as a training environment. The original screenshot/manual-reward workflow remains available.

Direct-state training needs fresh, coherent observations after actions, reliable self/opponent identification, actual episode boundaries, and a confirmed reset workflow. The Python client now enforces those protocol requirements, and it rebaselines ambiguous damage decreases without fabricating stock rewards. The discovered spectator export is too limited to supply that protocol. The next integration step depends on implementing and validating an accessible state source while controlling the offline game on Windows. CNN checkpoints are kept separate from the new vector-observation models.
