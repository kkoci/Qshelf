# grover_multi

Grover's search with a **multi-item** oracle (marking more than one item), on the same 3-qubit, 8-item register [`packages/grover`](../grover/) uses. A second-round addition to this registry, implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package.

## Why a separate package, not an extension of `packages/grover`

`packages/grover`'s single-marked-item API (`oracle[marked]`, `grover_search[marked, iterations]`) is already published, tested, and documented on its own. Rather than change its signature (breaking anyone already using it, and complicating its existing README/CLAUDE.md history), this package adds multi-item marking as a new, independent package that reuses the same validated building block. See the root [CLAUDE.md](../../CLAUDE.md) for the full reasoning.

## What's here

- `mark_item[item](qs)` -- the reusable single-item phase-flip gadget, factored out of what was `packages/grover`'s `oracle[marked]` (that package inlined this logic directly, since it only ever needed one marked item).
- `oracle_2items[item1, item2](qs)` / `oracle_3items[item1, item2, item3](qs)` -- call `mark_item[...]` once per marked item. That's it -- see "The finding" below.
- `diffuser(qs)` -- unchanged from `packages/grover` (duplicated, not imported -- this registry's packages don't depend on each other).
- `grover_2items[item1, item2, iterations](qs)` / `grover_3items[item1, item2, item3, iterations](qs)` -- the full circuit: uniform superposition, then `iterations` rounds of oracle + diffuser.
- `optimal_iterations(n_items=8, n_marked=1)` -- unchanged from `packages/grover` (the formula already generalized to `n_marked > 1`; that package just never called it that way).

## The finding: the existing gadget just composes by repetition

The task this package answers: now that `packages/grover`'s gotcha #12 (`with control(q0, q1): z(target)` is broken in guppylang 1.0.2) and its `H . multi-controlled-X . H` workaround are understood, does marking more than one item need a *different* gate structure, or can the same gadget just be repeated across multiple marked states?

**Confirmed, not assumed: it just repeats.** `oracle_2items`/`oracle_3items` are nothing more than 2 or 3 calls to `mark_item[...]`, with no new logic. Each single-item gadget is diagonal in the computational basis and acts as the identity everywhere except its own target, so composing several of them accumulates independent phase flips with zero interference -- verified via the exact statevector against the closed-form multi-item Grover formula for 5 different marking patterns (3 with k=2, 2 with k=3) and, separately, across a full sweep of iteration counts (baseline -> peak -> overshoot) for one pattern. No new bug turned up; the existing workaround from `packages/grover` fully covers this case.

## Install

From this directory:

```
pip install -e .
```

or add `grover-multi @ file:///path/to/guppy-registry/packages/grover_multi` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from grover_multi import grover_2items, optimal_iterations

iterations = optimal_iterations(n_items=8, n_marked=2)

@guppy
def main() -> None:
    qs = array(qubit() for _ in range(3))
    grover_2items[2, 5, iterations](qs)   # marks items 2 and 5
    ...
```

## Gotchas

None new beyond what `packages/grover` already documented (gotcha #12: multi-controlled Z was broken on guppylang 1.0.2, worked around via `H . multi-controlled-X . H`; gotcha #13: `control` disallowed in `@guppy.comptime`, so `mark_item`/the oracle-composition functions are plain `@guppy`). Gotcha #12 is now fixed upstream (tket 0.15.7 / guppylang 1.0.3, see the root [CLAUDE.md](../../CLAUDE.md)) -- `packages/grover`'s own copy of the workaround has since been removed, but `mark_item` here still uses it (at `OptimizationLevel.Minimal`, unchanged), since only `packages/grover`'s workaround removal was in scope for that fix. One confirmed constraint worth knowing if you're tempted to generalize this further: **which items are marked has to be baked in as separate, explicitly-arity-typed generic parameters** (`item1: nat, item2: nat, ...`), not iterated over from a Python list of arbitrary length -- guppy's explicit generic-instantiation syntax (`func[arg](...)`) needs each type argument to be a literal or closed-over constant, and a `@guppy.comptime` loop over a marked-items list (which *would* support arbitrary arity) isn't an option here because `control` is disallowed inside comptime. See the root [CLAUDE.md](../../CLAUDE.md) for the full writeup.

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Runs real circuits on Selene's Quest (statevector) emulator: the single-item gadget and both multi-item oracles verified in isolation (flip exactly the marked amplitudes, nothing else), the full circuits verified against the closed-form multi-item Grover formula for multiple patterns, and the probability curve verified across a full iteration sweep. See `tests/test_correctness.py` for the full methodology.

## Examples

- `examples/basic_grover_multi.py` -- searches for 2 and then 3 marked items, printing the amplified probability distribution, actual vs theoretical.

Run with `python examples/basic_grover_multi.py` from this directory.
