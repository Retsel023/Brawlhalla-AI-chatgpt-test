# Modes, continuous training, and shared skills

The mode menu selects the type of session and how its model learns. It is a client for the game-side connection; it does not install a Windows state/control provider. Real Brawlhalla sessions still require that provider. The new runner, team observations, checkpoint transfer, and continuous-session behavior are tested with synthetic providers.

## Open the menu on Windows

From this repository, using the Python environment with the dependencies installed:

```powershell
.\.venv\Scripts\python.exe agent.py menu
```

Running `agent.py` without arguments also opens the menu. Standard Windows Python includes Tk when the Tcl/Tk option is installed. The menu itself uses only Python's standard library; its training subprocess uses PyTorch and the other repository dependencies.

| Choice | Match | Default behavior |
| --- | --- | --- |
| Fight the AI — 1v1 | Private/local human versus the AI | Evaluate the saved duel model without changing weights |
| Train against bots | Offline 1v1 or 2v2 against bots | Keep learning until stopped |
| Online humans — 1v1 | Online human opponents | Learn from successive 1v1 matches |
| Online humans — 2v2 | Online teams with one or two controlled AI slots | Learn team play, starting from compatible duel weights when available |

The learning checkbox can enable learning while you fight it, or turn learning off for evaluation in another mode. Evaluation requires an existing checkpoint. Train against bots first if no duel model exists.

Set the explicit AI, teammate, and opponent player keys to the roles actually reported by the provider. In 1v1, the default is AI `Player1` versus `Player2`. In 2v2, the default is AI `Player1`, teammate `Player2`, opponents `Player3, Player4`. These are role assignments, not guesses about which player is on your keyboard. The provider must route your inputs to the human player and the AI's inputs to its declared slot.

Friendly fire is a 2v2 setting. It is included in observations and must match the real match rules. Team rewards use changes in total opponent damage minus changes in self and teammate damage; they do not identify who caused friendly damage.

### Choose the teammate arrangement

| Teammate choice | AI controls | What changes between matches |
| --- | --- | --- |
| Random teammate | One character | A different human can occupy the teammate role each match |
| Regular teammate (you) | One character | The reported teammate account ID stays fixed across this session |
| AI controls both | Both allied characters | One shared policy makes a separate decision for each character |
| Game bot | One character | The game controls the ally; useful for offline bot training |

Bot-training teams default to a game-bot ally. Online teams default to a random teammate. The menu saves this choice along with the role assignments. For a regular teammate, enter their Brawlhalla account ID or let the client pin the first reported teammate ID. Missing or changed regular-teammate IDs stop the session. Random teammates can change identity at new-match boundaries. Every arrangement still requires stable roles within a match.

For your regular-teammate setup, you keep the keyboard and the AI uses the secondary controller slot. Set the player keys to the actual provider assignment; for example, the AI might be `Player2` and you `Player1`. The menu does not infer player keys from the physical device order.

The uploaded game code recognizes the Steam launch option `-multikeyboard` and contains device-specific keyboard IDs and mappings. That helps with multiple-keyboard arrangements. It does not make ordinary `pynput`/Windows `SendInput` events target a particular keyboard. A keyboard plus a secondary controller is also a valid arrangement for the provider to implement. The client accepts distinct input-route declarations rather than assuming all characters share one keyboard stream; the Windows provider still needs to generate the actual device-specific input. This update does not install a virtual controller or keyboard driver, edit your Steam launch options, or implement the live Windows provider.

In **AI controls both**, each character sees its own self/teammate perspective and has a separate four-observation history. The shared policy predicts both actions as a batch, then sends one atomic joint command for one world transition. It never advances the match once per character. Both actors receive the team reward, while session statistics count the shared match once. Training-step budgets count actor decisions: one joint world step supplies two decisions.

The existing team checkpoint can resume with one or two controlled actors because each actor keeps the same observation and action layout. Changing teammate arrangements does not change the neural-network input size. Policy weights are shared; learning with a regular person does not create an automatic per-person behavior model. Repeated practice may adapt the policy to that person's habits, and mixing partners helps evaluate whether those skills generalize. Use a separate model folder if you want to preserve a personalized branch.

## Session controls

- **Check connection** validates the provider handshake and controlled-player selection. It does not configure a match, request actions, or load/train a model.
- **Start session** launches the selected session in a separate process; its log reports provider and model status. A running process alone does not mean the game connection worked.
- **Stop and save** requests a cooperative stop. The runner releases the provider connection and saves any initialized learning model. The last partial PPO rollout may be discarded; completed model updates are retained.
- **Force end** becomes available if cooperative stopping takes more than ten seconds. It terminates only the session process and can lose unsaved progress. The provider must have an independent input-release watchdog.
- **Save/Load settings** keeps a reusable menu configuration in a JSON file you choose.

`Steps = 0` means continuous operation. `Time limit = 0` means no time limit. Training with a positive step budget rounds up to a complete PPO rollout; the default rollout is 256 steps. Stop and time limits are also checked while waiting for a provider response. Continuous bot training still depends on the provider starting or resetting real bot matches and returning fresh state.

Checkpoints overwrite only the active branch's latest checkpoint at the chosen interval and on exit. The model and metadata are staged before replacement; an interrupted model/metadata pair is detected rather than silently resumed. Session logs under `runs/sessions/` rotate at 5 MiB with two backups, and the GUI keeps a bounded number of visible lines. Each new menu launch gets its own run directory; old run directories are retained for inspection.

For online modes, a match ending is followed by a request to wait for the next playable match. The Python client does not assume it can reset human opponents' matches. Match joining, matchmaking, and real state/input routing belong to the provider and are not implemented by this menu. The default wait limit is 120 seconds, configured by `bridge.episode_timeout_seconds`; exceeding it stops the session with an error. Provider errors do not cause actions or match commands to be replayed automatically.

## How 1v1 experience carries into 2v2

Both formats use the same 26-feature representation. The first ten features retain the legacy self/primary-opponent observations. Additional features describe a teammate and second opponent, their relative positions, presence masks, team format, and friendly-fire setting. Missing roles in 1v1 are zero-masked. Roles stay explicit and consistent throughout each match.

The runner stacks four observations by default, giving the policy 104 inputs with a short history of positions and damage. That can support learning movement timing and action sequences. These features still omit detailed attack frames, weapon state, cooldowns, and velocities; they are not a complete combat model. A stack of recent observations is not an explicit combo library or a guarantee of good combos.

Models are organized by format:

```text
models/sessions/
  duel/latest.zip
  duel/latest.json
  teams/latest.zip
  teams/latest.json
```

1. A 1v1 session resumes the duel branch. Bot training, sparring, and online duels can all use that branch.
2. The first 2v2 learning session copies compatible duel weights into a new team branch. It starts a fresh optimizer so the new task can adapt.
3. Later 2v2 sessions resume their own weights and optimizer. They do not overwrite the duel checkpoint.
4. Returning to 1v1 resumes the preserved duel branch. Team learning is not automatically merged back into it.

This transfers learned policy weights, including any useful movement or action-timing behavior. It does not replay old 1v1 matches as if they were fresh PPO experience. Transfer quality is empirical: test performance in both formats, and do not assume that a strong duel policy is already a good teammate. Separate branches protect the duel checkpoint from being overwritten while teams adapt; they do not prove that no skills are forgotten within a branch.

Metadata records model ancestry, checkpoint hashes, feature/action ordering, history length, and normalization/reward scales. Incompatible settings fail with an explanation. To change a contract deliberately, use a separate model folder rather than replacing an existing branch with incompatible data.

### Existing structured checkpoints

For an existing 10-feature structured model from the earlier runner, initialize a new duel branch explicitly:

```powershell
.\.venv\Scripts\python.exe agent.py session --profile bot_training --initialize-from models/ppo_state.zip --steps 10000
```

The migration preserves the old self/opponent input weights in the newest history frame and initializes new role/history inputs to zero. Actor and value outputs are tested for equivalence immediately after migration. Legacy models without metadata require the known original feature order and default scales. Screenshot CNN checkpoints use a different input representation and cannot be migrated this way.

Initialization is allowed only when the destination branch does not exist. Clear the initialization field when resuming a branch. The source checkpoint is never overwritten. Only load model files you trust.

## Command-line equivalents

Validate choices without connecting to the game or importing training dependencies:

```powershell
py -3.12 agent.py session --profile online_teams --dry-run
```

Examples once a matching protocol-2 provider exists:

```powershell
.\.venv\Scripts\python.exe agent.py session --profile bot_training --steps 0
.\.venv\Scripts\python.exe agent.py session --profile human_duel --steps 0
.\.venv\Scripts\python.exe agent.py session --profile online_duel --steps 0
.\.venv\Scripts\python.exe agent.py session --profile online_teams --teammate-kind random --steps 0
.\.venv\Scripts\python.exe agent.py session --profile online_teams --teammate-kind regular --self-player Player2 --teammate-player Player1 --steps 0
.\.venv\Scripts\python.exe agent.py session --profile online_teams --teammate-kind ai --steps 0
.\.venv\Scripts\python.exe agent.py session --profile bot_training --format 2v2 --friendly-fire --steps 0
```

Ctrl+C stops console sessions and saves initialized learning models. `--no-learn` runs evaluation. `--max-seconds 3600` supplies a one-hour limit. For non-default rosters use `--self-player`, `--teammate-player`, and `--opponents`; the provider must acknowledge the exact assignment.

The original `capture`, `observe`, `train`, and `play` commands remain available. The original `train --source bridge` uses protocol 1 and a two-player, 10-feature environment. Duel session presets use protocol 2; 2v2 session presets use protocol 3 to acknowledge teammate arrangements and input ownership. Both session formats share the same team/history representation. See the [session provider contract](session-provider.md) and [team-control extension](team-control-provider.md).
