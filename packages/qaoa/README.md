# qaoa

QAOA (Quantum Approximate Optimization Algorithm) for MaxCut, implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package, with a real classical-quantum optimization loop.

## Default graph

A 5-node cycle graph C5: edges `(0,1), (1,2), (2,3), (3,4), (4,0)`. C5 is an *odd* cycle, so it's not bipartite -- no partition cuts all 5 edges, and the true MaxCut value is 4 (e.g. `{0,2}` vs `{1,3,4}`). That makes it a genuinely nontrivial instance, unlike an even cycle (which has a perfect, trivially-findable solution) -- a real test of whether the optimization loop is doing something.

## What's here

- `cost_hamiltonian_layer(qs, gamma)` / `mixer_layer(qs, beta)` -- the two QAOA layers, for the default graph. `build_cost_hamiltonian_layer(n_nodes, graph)` / `build_mixer_layer(n_nodes)` are the factories that build them for *any* graph (see "Gotchas" below for why a factory, not a runtime `graph` argument).
- `qaoa_circuit(qs, gammas, betas)` -- the full p-layer circuit (`p` inferred from the length of `gammas`/`betas`): uniform superposition, then p rounds of cost + mixer.
- `run_qaoa(gammas, betas, shots=200, seed=0)` -- compiles and runs the circuit for given parameters (radians), returns `(average_measured_cut_value, bitstrings)`. The quantum half of the classical-quantum loop.
- `optimize_qaoa(p, shots=150, maxiter=20)` -- the actual hybrid loop: `scipy.optimize.minimize` (Nelder-Mead) searching for `(gammas, betas)` that maximize the average measured cut, each evaluation a real `run_qaoa` call. Returns `(best_gammas, best_betas, best_avg_cut)`.
- `cut_value(bitstring, graph=EDGES)` -- plain-Python classical MaxCut objective.

## Install

From this directory:

```
pip install -e .
```

or add `qaoa @ file:///path/to/guppy-registry/packages/qaoa` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from qaoa import optimize_qaoa, run_qaoa

gammas, betas, avg_cut = optimize_qaoa(p=2)
print(f"found gamma={gammas}, beta={betas}, avg cut={avg_cut}")

avg_cut, bitstrings = run_qaoa(gammas, betas, shots=1000)
```

`gamma`/`beta` are in **radians** (the standard QAOA convention). See `qaoa.py`'s module docstring for how that maps onto guppylang's `angle` type (which stores half-turns, not radians) -- this was pinned down empirically against an exact matrix-exponential reference, not just derived by hand, and is worth reading before adding rotation gates to a future package.

## Gotchas

- **`graph` is a factory-time parameter, not a call-time one.** `cost_hamiltonian_layer(qs, gamma)` (no `graph` argument) is what you actually call; `build_cost_hamiltonian_layer(n_nodes, graph)` is what builds it. Guppy compiles a fixed circuit structure, so the edge list has to be known when the guppy function is *defined* (traced via `@guppy.comptime`, iterating a plain Python list of edges), not passed in at the call site. Use `build_qaoa_circuit(n_nodes, graph)` for a different graph than the default 5-cycle.
- **Explicit generic call syntax (`func[T](...)`) doesn't work inside `@guppy.comptime`** (raises `GuppyComptimeError`), unlike in a plain `@guppy` function (see `packages/grover`'s CLAUDE.md gotcha #15). Not needed here -- `qaoa_circuit`'s `p` is always inferred from array length -- but worth knowing.
- **Comparing this package's circuits against a hand-built reference needs full complex phase alignment**, not just a +-1 sign flip like `packages/qft`/`packages/grover` use (their circuits are real-valued; QAOA's `rz`/`rx` gates are genuinely complex). See `tests/test_correctness.py`'s `_phase_align` and its module docstring.

Full writeup, including why this package's heavy use of `@guppy.comptime` (three levels of comptime-calling-comptime) was checked against `packages/qft`'s confirmed comptime bug and found *not* to apply here, is in the root [CLAUDE.md](../../CLAUDE.md).

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Tests run real circuits on Selene's Quest (statevector) emulator: `cost_hamiltonian_layer`/`mixer_layer` verified against hand-derived phases for a small case, the full circuit verified against an exact `scipy.linalg.expm` reference for several parameter sets and p = 1, 2, 3 layers, and -- the actual evidence QAOA is doing something -- the measured cut distribution shown to improve from the p=0 baseline through p=1 to p=2, plus a separate, small *live* `optimize_qaoa` run confirming the hybrid loop itself beats baseline. Also confirms (not just assumes) that Selene's Stim backend rejects this circuit outright. See `tests/test_correctness.py` for the full methodology.

If a test fails with `OSError: ... Application Control policy has blocked this file` (Windows only), that's an environmental issue unrelated to this package's correctness -- see CLAUDE.md gotcha #3 -- just rerun.

## Examples

- `examples/basic_qaoa.py` -- runs the classical-quantum loop for p=0, 1, 2 layers on the default graph and prints the average cut value and most-sampled partition at each, next to the true MaxCut value.

Run with `python examples/basic_qaoa.py` from this directory. Takes about a minute (real optimization loop, not a precomputed result).
