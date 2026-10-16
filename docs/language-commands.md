# Local language commands

Use the menu's chat field to change session settings and suggest movement or action practice in plain language. Ollama runs the language model locally. Its response is validated before any settings change, and the PPO policy continues to choose the game actions.

**Real gameplay still requires the Windows game-side provider, which is not implemented in this repository.** Installing Ollama supplies the language interface; it does not establish the game connection or train a competitive policy.

## Windows setup

Install [Ollama for Windows](https://ollama.com/download/windows). The Windows application normally starts its local server in the background. From PowerShell, download the default model and check the server:

```powershell
ollama pull qwen2.5:3b
Invoke-RestMethod http://127.0.0.1:11434/api/version
.\.venv\Scripts\python.exe agent.py menu
```

If the Ollama application is not already running, start `ollama serve` in another terminal and leave it open. The default endpoint is `http://127.0.0.1:11434`. The command client uses Python's standard library; it needs neither an Ollama Python package nor PyTorch. Running and training the game policy still requires the project's training dependencies.

The model is configurable under `language_commands` in `config.json`:

```json
{
  "language_commands": {
    "endpoint": "http://127.0.0.1:11434",
    "model": "qwen2.5:3b",
    "timeout_seconds": 30,
    "max_response_bytes": 65536
  }
}
```

This is a section to merge into the existing configuration. Pull a different model with `ollama pull <model-name>` before selecting it. The client accepts a local loopback endpoint and rejects redirects. Requests use Ollama's `/api/chat` endpoint with nonstreaming structured output. Invalid, unsupported, incomplete, or clarification-only responses leave the settings unchanged. A successful “Updated session settings” message describes the validated change; provider status establishes whether anything was applied in the game.

### Radeon RX 9070 XT

Ollama manages its own inference backend separately from PyTorch. After an actual language request, run `ollama ps` to see whether the loaded model uses CPU or GPU. Current official documentation describes Windows Vulkan support, but this project's Windows RX 9070 XT execution remains unverified. Use Ollama's CPU fallback or a smaller model if GPU inference is unavailable. The policy's `--device cpu` or `--device cuda` setting does not select Ollama's backend.

## Commands to try

| Say | Intended settings or guidance |
| --- | --- |
| “Play as Hattori” | Select Hattori for the main AI at the next provider-supported match setup |
| “Keep more distance” | Prefer farther spacing from opponents |
| “Stay near your teammate” | Prefer nearby ally positioning in 2v2 |
| “Practise down light then jump light” | Suggest a short supported action sequence for practice |
| “Clear my suggestions” | Clear the tactical guidance |

Session choices, player roles, and legend selection are setup settings. They do not silently reconfigure a match already in progress. Tactical guidance can be updated while a session runs; the environment reads the new guidance at its next decision. An outstanding action or provider request completes before another policy decision. Unsupported requests produce a clarification instead of guessing a setting.

The command-line form can edit saved session settings without opening Tk:

```powershell
py -3.12 agent.py command "Play as Hattori" --session-file session.json --output hattori-session.json --config config.json
```

Create `session.json` with **Save settings** in the menu. Load the resulting file in the menu or pass it to `agent.py session --session-file hattori-session.json`. Language-command interpretation does not launch a match.

## What guidance changes

Guidance supports these values:

| Key | Values |
| --- | --- |
| `opponent_spacing` | `free`, `close`, `mid`, `far` |
| `teammate_spacing` | `free`, `near`, `spread` |
| `strength` | A number from `0` to `1` |
| `combo` | Empty to clear, or a list of two to six supported action labels |

Use the exact supported action labels defined in [guidance_settings.py](../guidance_settings.py). Arbitrary button names, timing scripts, and unknown labels are rejected.

For example, `["down_light", "jump_light"]` expresses the two-action drill above. A drill needs at least one light or heavy attack, may include movement, jumping, or dodging, and cannot include `idle`. Progress follows consecutive actions; another action resets it. Clearing suggestions restores `free` opponent spacing, `free` teammate spacing, strength `0.3`, and an empty drill.

During learning, the environment supplies goal features and additional shaping rewards for the selected practice goals. A frozen evaluation policy receives the goal features without updating weights or receiving training shaping rewards. A policy must learn how those goal features relate to useful actions; changing a suggestion does not guarantee an immediate change in its play. Train and evaluate each goal using consistent scenarios.

“Combo” currently means an input drill. The available observations do not include enough weapon, attack-frame, cooldown, or hit-confirm data to verify a true weapon combo. Completing an input sequence does not establish that it connected. In 1v1, its completion bonus also requires an increase in opponent damage. In 2v2, damage changes cannot reliably identify which allied character caused a hit, so that bonus is disabled.

For a console session, use `--guidance-file` to read live guidance from a local JSON file. Keep its keys within the table above. For example, a spacing-only file can contain:

```json
{
  "opponent_spacing": "far",
  "teammate_spacing": "free",
  "strength": 0.5,
  "combo": []
}
```

```powershell
.\.venv\Scripts\python.exe agent.py session --profile bot_training --legend Hattori --guidance-file guidance.json --steps 0
```

To update a running menu session from another terminal, use the run directory printed in its log:

```powershell
.\.venv\Scripts\python.exe agent.py command "Keep more distance" --session-file runs/sessions/<run-id>/options.json --guidance-file runs/sessions/<run-id>/guidance.json
```

Replace `<run-id>` with that directory's actual name. The command reads the latest guidance before applying a partial update, preserving any other active suggestions. For a console session, provide the same saved session settings and guidance file that it uses. File updates are atomic; invalid updates retain the previous goals and produce a diagnostic.

## Legend selection and provider support

Legend selection applies to the main AI only. In **AI controls both**, it does not select the teammate's legend. The provider must advertise the `legend_selection` capability and acknowledge the requested legend during match configuration. Its next returned game state must confirm that selection. Missing capability, a mismatched acknowledgment, or a mismatched state stops setup rather than pretending the character changed.

The language client cannot supply a missing provider capability. The built-in spectator exporter also cannot configure a legend, control a player, or serve as this live provider. See the [session provider](session-provider.md) and [team-control provider](team-control-provider.md) contracts.

## Validation scope and references

Command parsing, settings validation, and policy integration can be checked with local HTTP fixtures and synthetic game providers. Actual model inference was not tested in this cloud environment: Ollama is absent and its model registry is outside the current network allowlist. No model download or Windows GPU execution is implied by those tests.

Official Ollama references used for the integration:

- [API: local model listing and structured chat](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Windows installation and runtime](https://github.com/ollama/ollama/blob/main/docs/windows.mdx)
- [Hardware and backend support](https://github.com/ollama/ollama/blob/main/docs/gpu.mdx)
