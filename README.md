# Brawlhalla learning agent — initial prototype

A local screen-reading PPO agent with movement, jump, dodge, and attack actions. Training resumes from `models/ppo_brawlhalla.zip`. This is an initial human-feedback training prototype, not a trained competitive player. No game memory access is used.

## Windows / Radeon RX 9070 XT

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

## Configure and run locally

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

The policy updates from rewards supplied by a person. Without feedback, it receives zero reward and has no useful learning objective. Improvement is not guaranteed. Automated damage/stock detection and reliable match resets are **not implemented**. A 60-second episode boundary only clears policy history; it does not reset the game. Use a continuously running training session and manually restore the scenario when needed. Performance must be evaluated over repeated consistent trials before claiming improvement.

The GPU accelerates policy updates; screenshot capture and real-time game interaction still limit training throughput. A single game at ten actions per second makes reinforcement learning slow, even with a powerful GPU.

## Cloud checks

Run `python -m unittest discover -s tests -v` for feedback/action checks and the synthetic PPO smoke test. These do not establish actual game control or GPU compatibility. Real capture, keyboard input, game rewards, and GPU training must be checked on the Windows host. Each cloud task is already isolated; use this checkout directly without creating a worktree.
