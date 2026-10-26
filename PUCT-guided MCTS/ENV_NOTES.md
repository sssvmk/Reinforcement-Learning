# ENV_NOTES: what was found in KAGGLE_DIR (read-only inspection)

`KAGGLE_DIR` = `C:\Users\username\OneDrive - XXYY\kaggle-environments-0.1.4` (28 files; nothing was modified, imported or installed).

## Contents
A **source copy of the `kaggle-environments` package, v0.1.4 (2020 BETA)**. No competition notes, no starter notebooks for ConnectX
beyond the environment's own rendering notebook, no data files.

| Path | Size (B) | What it is |
|---|---|---|
| `README.md`, `setup.py`, `requirements.txt` (gym, jsonschema, numpy, scipy), `MANIFEST.in`, `CONTRIBUTING.md`, `LICENSE`, `.gitignore`, `.gcloudignore` | 0.5k-11k | packaging / docs |
| `main.py` (5770) | | CLI (`list/evaluate/run/step/load/http-server`); not used |
| `kaggle_environments/core.py` (19892), `utils.py`, `errors.py`, `schemas.json`, `status_codes.json`, `__init__.py` | | generic environment runner, schemas |
| `kaggle_environments/envs/connectx/connectx.py` (7058) | | **ConnectX rules, reference `random` and `negamax` agents, interpreter** |
| `.../connectx/connectx.json` (1325) | | **ConnectX specification** (config defaults, observation, action, reward) |
| `.../connectx/connectx.js` (8149), `connectx.ipynb` (72885), `test_connectx.py` (6603) | | renderer, demo notebook, env tests |
| `.../envs/identity/*`, `.../envs/tictactoe/*`, `static/player.html` | | other envs / viewer (irrelevant) |

## Confirmed from local files
* Defaults: `rows=6`, `columns=7`, `inarow=4` (`connectx.json`). Configurable (the env's own tests use 4x5, inarow 3).
* Observation: `board` = flat list length `rows*columns`, **row-major starting at the TOP row**, `0` empty, `1`/`2` players; `mark` = own player number (`connectx.json`, `connectx.py`). Gravity: `play()` drops into the max empty row index of the column.
* Action: integer column in `[0, columns)`; legal iff `board[column] == 0` (top cell empty). Player 1 moves first (`reset` gives marks 1 and 2; first ACTIVE agent is agent 0).
* Reward: 1 win, 0 loss, 0.5 draw. Invalid column -> that agent's status becomes `Invalid column: ...` and it loses.
* Win = `inarow` consecutive in a row/column/either diagonal through the last move; draw when board full.
* **Time limit in this copy**: `configuration.timeout` (integer seconds an agent may run per action), **default 2**; also `steps` default 1000. There is **no `actTimeout` / `remainingOverageTime`** in this v0.1.4 source (`core.py` `__run_agent` wraps the agent in `timeout(..., seconds=configuration.timeout)`; the exception -> TIMEOUT status).
* Agent signature: `agent(observation, configuration)` (core passes `args[:agent.__code__.co_argcount]`, so `(obs, config)` works); attribute access `obs.board`, `obs.mark`, `config.columns` is used in the reference agents (structify). My submission accepts both dicts and attribute objects.
* Observation in this copy has **no `step` field** (only `board`, `mark`); the task text mentions `step`, so it is treated as optional and never required.
* Minor quirk in the reference interpreter: the invalid-move check is `active.action > columns` (not `>=`), so `column == columns` raises instead of cleanly forfeiting. Irrelevant for legal agents.
* Reference `negamax_agent` there is depth 4 with a crude proximity heuristic and no pruning; my own alpha-beta is independent and stronger.

## UNVERIFIED (not confirmable from local files; defaults used)
1. The **actual Kaggle competition** time limits. Newer kaggle-environments versions use `actTimeout` (commonly 2 s) plus a pool of overage time (`remainingOverageTime`, commonly 60 s); neither exists in this local copy. The submission reads `actTimeout`, then `timeout` from the configuration, else falls back to 2 s, and only uses about half of it.
2. The **submission format** (single `submission.py` with a top-level `agent` function vs. notebook vs. archive) and the **packages available** on the Kaggle runner (NumPy assumed present; nothing else is used). Maximum submission file size is unknown (mine is ~0.6 MB).
3. That the `step` key exists in the real observation (not needed).
4. That the live competition uses the 6x7 / 4-in-a-row defaults (the local spec says so).
5. Exact CPU speed of the Kaggle runner (the search is time-bounded, so strength, not legality, depends on it).
6. Whether the module-level Python state persists between calls to `agent` in the real runner (used only for optional subtree reuse; code is correct if it does not persist).

## Environment check (step 0.4)
Python 3.13.3, **PyTorch 2.14.1+cpu importable**, `torch.get_num_threads() == 8`, `os.cpu_count() == 16`; numpy 2.2.6, matplotlib 3.10.8, pytest 9.1.1, PyYAML 6.0.3 already installed. Nothing was installed.
