# ConnectX — Step 0 Notes

**`kaggle_environments` runtime status: NOW AVAILABLE (update).**
Initially the package was not importable in this environment (`import gym`
failed), and the user declined an install when asked. The user subsequently
installed `kaggle-environments`/`gym` themselves outside this session; as of
this update `import kaggle_environments` succeeds and `make("connectx")`
works. All "assumed" items below have now been **live-confirmed** against
the real runtime (see the "Live-verified" section). The original
source-reading findings are kept for reference and were all confirmed
correct by the live check — no corrections were needed.

Consequences:
- Part A (`connectx_state.py`, `mcts_core.py`, `agents.py`, `tournament.py`)
  was implemented and tested entirely standalone first, per the task's
  explicit allowance, and is unaffected.
- Runtime-dependent checks (`pytest.importorskip("kaggle_environments")`
  tests: cross-check vs. real board, matches vs. built-in `random`/
  `negamax` agents) are **now enabled** and have been run — see
  `connectx/tests/test_agents.py` and the Part A tournament results.

## Live-verified (`kaggle_environments.make("connectx")`)
```
env.configuration = {'episodeSteps': 1000, 'actTimeout': 2, 'runTimeout': 1200,
                      'columns': 7, 'rows': 6, 'inarow': 4,
                      'agentTimeout': 60, 'timeout': 2}
```
- Confirms the assumed `timeout=2`, `steps=1000` defaults exactly, plus
  previously-undocumented `runTimeout=1200`, `agentTimeout=60`.
- Fresh `env.reset()` observation: player 1 (`mark=1`) is `ACTIVE` with
  `step=0` and an all-zero `board` of length 42; player 2 (`mark=2`) is
  `INACTIVE` and only gets a `mark` key until it's their turn (no `board`/
  `step` until active) — confirms turn-based single-active-agent observation
  delivery, not simultaneous.
- After player 1 plays column 2 then player 2 plays column 3: board index
  37 (`row=5,col=2`, since `37 = 5*7+2`) and index 38 (`row=5,col=3`) are
  set to `1` and `2` respectively — both pieces land in **row 5**, the
  bottom row (rows are `0..5`). **Confirms row 0 = top, row `rows-1` =
  bottom**, i.e. our `from_observation`/`apply()` orientation is correct.
- `remainingOverageTime: 60` appears per-agent in the observation (ties to
  `agentTimeout`), additional to the per-move `actTimeout`.

## `CONNECTX_DIR` contents
Recursive listing (top-level): the full `kaggle_environments` 0.1.4 source
tree — `kaggle_environments/{core.py, utils.py, errors.py, schemas.json,
main.py, envs/connectx/{connectx.py, connectx.json, connectx.js,
connectx.ipynb, test_connectx.py}}`, plus `setup.py`, `requirements.txt`,
`README.md`. There is no separate competition-data folder; for ConnectX,
as documented, the "data" is the environment/package source itself.

`kaggle_environments/requirements.txt` lists: `gym`, `jsonschema`, `numpy`,
`scipy`.

## Configuration (from `envs/connectx/connectx.json`)
Confirmed directly from the JSON schema file (not a live object):
- `columns`: default **7**
- `rows`: default **6**
- `inarow`: default **4**
- `agents`: `[1, 2]` for 2-player ConnectX
- `reward`: enum **{0 (loss), 0.5 (draw), 1 (win)}** — note this is NOT a
  `{-1, 0, 1}` zero-sum reward; it is a `{0, 0.5, 1}` win/draw/loss scale.
  Part A's MCTS value backup is implemented using a `{-1, 0, 1}` internal
  convention (standard for UCT sign-flipping) and the Kaggle-compatible
  agent wrapper is just the action (column index), so this reward-scale
  difference does not affect the wrapper, but is noted for anyone wiring
  up `kaggle_environments.evaluate()` scoring directly.
- `observation.board`: flat array of length `rows*columns`, values
  `0` = empty, `1` = player 1, `2` = player 2.
- `observation.mark`: enum `{1, 2}` — the calling agent's own mark.
- `reset.observation` = `[{mark: 1}, {mark: 2}]`, confirming player 1 is
  assigned mark 1 and moves first, player 2 is assigned mark 2.
- **`steps` / `timeout` / `actTimeout` / `runTimeout` are NOT present** in
  this version's `connectx.json` (no per-episode step cap or per-move time
  limit is declared in the schema). `core.py` references
  `self.configuration.steps` and a generic `timeout()` helper, implying
  these are supplied by a default elsewhere (possibly a newer package
  version, or environment-agnostic defaults not present in this 0.1.4
  schema file). **Since this cannot be verified without running the
  package, we proceed with commonly documented Kaggle ConnectX defaults as
  an explicit assumption**: `timeout = 2` seconds (`actTimeout`),
  `steps = 1000` (episode cap, effectively unreachable since the board
  fills/terminates after at most 42 moves). `tournament.py` enforces and
  reports its own per-move wall-clock budget (the simulation/time budget
  passed to MCTS) regardless of this assumption.

## Board orientation (from `envs/connectx/connectx.py`)
- `play()`: `row = max(r for r in range(rows) if board[col + r*columns] == EMPTY)`
  — i.e. a dropped piece lands at the **largest** row index that is still
  empty in that column. Since board is indexed `col + row*columns`, and a
  piece should fall to the bottom, this confirms **row 0 is the top row**
  and **row `rows-1` is the bottom row**, matching the task's documented
  orientation. `from_observation` in `connectx_state.py` assumes this
  orientation and is unit-tested against it using constructed boards.
- `is_win()` checks vertical, horizontal, and both diagonals for `inarow`
  consecutive marks, generalized to any `rows`/`columns`/`inarow` — our own
  `connectx_state.py` win-check is cross-tested against an independent
  reference implementation (see `connectx/tests/`), not against this file
  directly (since importing it requires `gym`), but the algorithm was read
  and matched conceptually.
- Illegal move (occupied/out-of-range column) forfeits the game for the
  acting agent (`active.status = "Invalid column: ..."`).

## Built-in agents (`envs/connectx/connectx.py`)
- `random_agent`: picks a uniformly random legal (non-full) column.
- `negamax_agent`: fixed-depth-4 negamax with a cell-proximity heuristic at
  the depth cutoff, and an immediate-win short-circuit. Not runnable here
  (needs the package); Part A implements its own `negamax` baseline
  (alpha-beta, configurable depth) independently, matching this style but
  not this exact code.

## Summary of what Step 0 confirmed vs. assumed
| Item | Status |
|---|---|
| `rows=6, columns=7, inarow=4` | Confirmed (connectx.json) |
| Board flat array, 0/1/2, top-row-first | Confirmed (connectx.json + connectx.py `play()` logic) |
| `mark` enum {1,2}, player 1 moves first | Confirmed (connectx.json `reset.observation`) |
| `reward` enum {0, 0.5, 1} | Confirmed (connectx.json) |
| `steps`, `timeout`/`actTimeout` defaults | **Live-confirmed** (`episodeSteps=1000, actTimeout=2, runTimeout=1200, agentTimeout=60, timeout=2`) |
| Win-check correctness, live runtime parity | **Live-confirmed** — `from_observation` round-trip and random-game board parity tests pass against the real runtime (see `test_agents.py`) |

## Update log
- Initial pass: package not importable, user declined install, all findings
  source-derived.
- Follow-up: user installed `kaggle-environments`/`gym` independently. Live
  checks above confirm every earlier assumption was correct; no corrections
  to the data/orientation description were needed, only confidence upgraded
  from "assumed" to "confirmed". Runtime-dependent tests un-skipped and run.

## Runtime quirk discovered during live verification
In this installed version, when **manually inspecting** `env.state` after
`env.step()` (not via `env.run()` with real agent callables), the two agent
slots' observation dicts are asymmetric: `state[0]` (mark=1) always carries
a populated `board`/`step`, even on turns where it is `INACTIVE`;
`state[1]` (mark=2) never includes `board`/`step` in its observation dict,
even while `ACTIVE`. Verified by stepping through a full
`env.run(["random","random"])` game and printing `"board" in
s["observation"]` for both slots at every step: `state[0]` is `True` at all
39 steps, `state[1]` is `False` at all of them.

**This does not affect real agent calls.** When a genuine Python agent
function is passed to `env.run([...])` (as our wrapped agents are), the
framework's internal invocation always supplies that agent's own full
`observation` (with `board`) at call time regardless of seat — confirmed by
running `env.run([random_agent, mcts_agent(n_sims=20)])` with our MCTS
agent in the **mark=2** seat: it played normally and the game completed
(final rewards `[-1, 1]`, statuses `["DONE","DONE"]`) with no errors. The
asymmetry is purely an artifact of `env.state`'s snapshot bookkeeping for
external inspection, not of the agent-calling contract. Our cross-check
test (`test_from_observation_matches_real_runtime_random_games`), which
does inspect `env.state` directly rather than going through `env.run`,
works around it by reading `board` from `state[0]` and the acting `mark`
from whichever slot has `status == "ACTIVE"`.
