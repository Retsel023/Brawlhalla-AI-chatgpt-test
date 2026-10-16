# Brawlhalla learning agent

Open the new mode menu on Windows with `.\.venv\Scripts\python.exe agent.py menu`. It includes human-versus-AI 1v1, continuous bot training, online human 1v1, and online 2v2 presets, plus learning, roster, checkpoint, and device settings. **Actual gameplay still requires a Windows game-side provider.** See [modes and shared skills](docs/session-modes.md) for setup, continuous sessions, 1v1-to-2v2 transfer, and current limitations.

A local PPO agent with movement, jump, dodge, and attack actions. It now supports structured game-state observations and automatic damage rewards through a controlled-state bridge, alongside the original screenshot/human-feedback workflow. This is a development prototype, not a trained competitive player.

The uploaded game files revealed a real JSON exporter, but its verified enable path is for spectating and it updates too slowly for frame-level control. The built-in exporter is supported in **observation mode**. The separate bridge training interface needs a Windows game-side provider that has not been implemented. This project currently contains no process-memory reader or game instrumentation. See [game-data findings](docs/game-data-findings.md) and the [bridge contract](docs/state-bridge.md).

| Workflow | Observations | Rewards | Current requirement |
| --- | --- | --- | --- |
| `observe` | Built-in spectator JSON, normalized player features | Damage projections for inspection | A fresh exported file |
| `train --source visual` | Four stacked screen frames | Human feedback keys | Local focused game window |
| `train --source bridge` | Structured player state | Automatic damage deltas | Separate offline game-side bridge provider |

## Read actual game data

Set the Steam launch option `-writestats`, restart Brawlhalla, and spectate a private match to exercise the exporter found in this build. Run:

```powershell
.\.venv\Scripts\python.exe agent.py observe --self-player Player1 --opponent-player Player2 --duration 60
```

Use the actual `Player<number>` keys from `%USERPROFILE%\BrawlhallaStatsLive\PlayerStatuses.json`; they identify the selected perspective, not necessarily the player you control. Add `--telemetry-dir "C:\path\to\BrawlhallaStatsLive"` if needed. No screenshots, keyboard input, PyTorch, or policy training are used by observation mode; it requires NumPy. Output is JSONL with normalized features, source timestamps, damage rewards, and discontinuity diagnostics. Missing, stale, or invalid feeds fail with an explanation instead of being treated as valid training data.

To retain raw exported observations without NumPy, use `py -3.12 telemetry.py --duration 60`.

## Train from structured state

Once a real offline game-side provider implements the [bridge contract](docs/state-bridge.md), the client can train directly from positions, damage, and elimination flags:

```powershell
.\.venv\Scripts\python.exe agent.py train --source bridge --self-player Player1 --opponent-player Player2 --device cuda --steps 10000
.\.venv\Scripts\python.exe agent.py play --source bridge --device cuda
```

These commands require that additional provider; they do not enable Brawlhalla telemetry themselves. The client requires an offline-controlled handshake, acknowledged actions, increasing simulation ticks, and confirmed episode resets. It stops on broken state continuity. Bridge models use `MlpPolicy` and `models/ppo_state.zip`; screenshot models use `CnnPolicy` and `models/ppo_brawlhalla.zip`. Old CNN weights cannot be reused as state models. Normalization, player slots, and reward scale are configurable under `state` in `config.json`; keep them stable when resuming a checkpoint. Ctrl+C stops bridge training and saves the model. The provider must release its keys on disconnect and run an input watchdog.

The automatic damage reward is `0.01 * (opponent damage gained - self damage gained)`. Damage decreases rebaseline the whole transition with zero reward, because the exported data cannot distinguish a respawn from a reset. Identity changes also rebaseline observation diagnostics and abort controlled training. An elimination edge is reported for inspection; it is not used to invent stock counts, knockout rewards, or a match result. Bridge episode endings come from the provider's explicit flags.

## Windows / Radeon RX 9070 XT

The configured game directory is `C:\Program Files (x86)\Steam\steamapps\common\Brawlhalla`. Check the local installation and compare its main SWF against the exact build inspected here:

```powershell
py -3.12 agent.py doctor
```

`doctor` uses only the Python standard library and reads files without launching the game. A missing file or changed build produces a nonzero exit status with details. Use `--game-dir "D:\SteamLibrary\steamapps\common\Brawlhalla"` if you move the installation. A successful static check establishes file/build compatibility, not live telemetry or GPU readiness. Exported JSON still lives under your Windows user directory, not the Steam installation folder.

Install Python and create a virtual environment in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

For GPU training, first follow AMD's **current Windows PyTorch installation instructions and compatibility matrix** for the RX 9070 XT:
https://rocm.docs.amd.com/projects/radeon/en/latest/index.html

Use the Python version and AMD driver version specified there; recreate the virtual environment with that Python version if necessary. Install AMD's matching ROCm SDK and PyTorch wheels into this same virtual environment using their published commands. Linux ROCm wheels and NVIDIA CUDA wheels are not interchangeable with Windows AMD wheels. GPU installation has not been verified on this cloud Linux CPU machine. The cloud network currently blocks access to AMD documentation, so no speculative wheel URLs or version combinations are provided.

Then install the agent dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__, 'HIP:', torch.version.hip, 'GPU:', torch.cuda.is_available())"
```

AMD's ROCm PyTorch uses the `torch.cuda` API name. `--device cuda` requires an available HIP backend and runs a matrix multiplication/backpropagation probe before starting. `--device auto` uses that GPU when available and otherwise prints a CPU fallback. For a CPU-only installation, install PyTorch from https://download.pytorch.org/whl/cpu before installing requirements.

## Screenshot training on Windows

1. Open Brawlhalla offline training with an opponent/bot. Keep it windowed and focused. Do not use public matchmaking for experimentation.
2. Set `capture` in `config.json` to the game client area's actual screen coordinates. Match the configured controls to your in-game keyboard bindings. Keep game resolution and capture region consistent between runs.
3. Capture a frame, inspect `capture.png`, and verify the game is correctly cropped:

```powershell
.\.venv\Scripts\python.exe agent.py capture
.\.venv\Scripts\python.exe agent.py train --device cuda --steps 10000
```

Focus the game during the five-second countdown. F8 gives +1 reward; F9 gives -1. Use these promptly to reinforce useful behavior or penalize mistakes. F10 marks an episode boundary; F12 stops training and saves. Ctrl+C in the terminal also saves and releases held keys. Periodic checkpoints are saved every 1,000 environment steps. Resume with the same train command. Evaluate without updating weights:

```powershell
.\.venv\Scripts\python.exe agent.py play --device cuda
```

Keep keyboard settings, capture geometry, and action definitions unchanged when resuming. Model checkpoints should only be loaded from trusted sources.

## What learning currently means

Screenshot training updates from rewards supplied by a person. Without feedback, that mode receives zero reward and has no useful learning objective. Its 60-second episode boundary only clears policy history; it does not reset the game. Use a continuously running training session and manually restore the scenario when needed.

Bridge training calculates damage rewards and waits for actual reset acknowledgments. Its client/environment/PPO integration is tested against a synthetic provider. A real controlled-state provider and live Windows validation remain outstanding. Improvement is not guaranteed in either mode; evaluate over repeated consistent trials before claiming it.

The GPU accelerates policy updates; screenshot capture and real-time game interaction still limit training throughput. A single game at ten actions per second makes reinforcement learning slow, even with a powerful GPU.

## Cloud checks

Run `python -m unittest discover -s tests -v` for feedback, desktop input cleanup, telemetry parsing, state/reward transitions, protocol checks, and synthetic PPO/checkpoint tests. These do not establish actual game control or GPU compatibility. Real capture, keyboard input, game exports, and GPU training must be checked on the Windows host. Each cloud task is already isolated; use this checkout directly without creating a worktree.
