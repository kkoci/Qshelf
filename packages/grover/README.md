# grover

Grover's search algorithm on a 3-qubit register (8 items, 1 marked item), implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package.

## What's here

- `oracle[marked](qs: array[qubit, 3])` -- flips the phase of exactly the basis state `|marked>` (`marked` in `[0, 8)`), leaves every other state unchanged.
- `diffuser(qs: array[qubit, 3])` -- the Grover diffusion operator (inversion about the mean, `2|s><s| - I`).
- `grover_search[marked, iterations](qs: array[qubit, 3])` -- prepares the uniform superposition and applies `iterations` rounds of oracle + diffuser.
- `optimal_iterations(n_items=8, n_marked=1)` -- plain Python helper computing the iteration count that maximizes the marked-item probability (2, for the default 8-item/1-marked case).

Not yet included: an arbitrary-`n` version. The multi-controlled phase flip both `oracle` and `diffuser` need requires explicitly listing each control qubit at the `with control(...)` call site (see "Gotchas" below), so scaling to a different register size means writing a new set of functions with the right number of controls hard-coded, not just changing a generic parameter the way `qft`/`iqft` scale over `n`. A `grover4` (4-qubit, 3-control) variant would follow the exact same pattern; see CLAUDE.md.

## Install

From this directory:

```
pip install -e .
```

or add `grover @ file:///path/to/guppy-registry/packages/grover` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import qubit, discard_array
from grover import grover_search, optimal_iterations

MARKED = 5
ITERATIONS = optimal_iterations(8, 1)  # 2

@guppy
def main() -> None:
    qs = array(qubit() for _ in range(3))
    grover_search[MARKED, ITERATIONS](qs)
    state_output("out", qs)
    discard_array(qs)
```

**Important:** `optimal_iterations(...)` and any other classical/Python-level computation of a generic argument must run in plain Python *before* the `@guppy` function, then be passed in via a closed-over variable (`ITERATIONS` above) -- calling it directly inside the explicit-generic subscript (`grover_search[MARKED, optimal_iterations(8, 1)]`) fails to compile, since guppy parses that subscript as guppy source, not a Python expression to evaluate first. See CLAUDE.md.

## Gotchas

`with control(q0, q1): z(target)` (a multi-controlled Z built with 2+ control qubits via guppy's `control` modifier) was **broken** in guppylang 1.0.2 -- it produced a wrong, non-unitary-as-expected result. This is now fixed upstream (tket 0.15.7 / guppylang 1.0.3, [Quantinuum/guppylang#2251](https://github.com/Quantinuum/guppylang/issues/2251)), and `oracle`/`diffuser` use the direct `with control(...): z(...)` form. **Important nuance**: the fix only takes effect at `OptimizationLevel.Classical` or `.Default`, not `.Minimal` -- it lives in tket's `Normalize` pass, which `Minimal` never runs. This package's tests and example accordingly compile at `OptimizationLevel.Classical`, not `.with_minimal_opt()`. See `grover.py`'s module docstring and the root [CLAUDE.md](../../CLAUDE.md) (gotcha #12) for the full writeup, including how this was verified.

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Tests run real circuits on Selene's Quest (statevector) emulator and check them against the closed-form Grover amplitude-amplification formula: the oracle's and diffuser's action in isolation, the full circuit's statevector for every possible marked item at the optimal iteration count, and the marked-item probability curve across iteration counts (confirming it rises to the theoretical maximum and then falls past the optimum, direct evidence of genuine amplitude amplification). See `tests/test_correctness.py` for the full methodology, including a note on why these tests don't need the subprocess isolation `packages/qft`'s tests use.

## Examples

- `examples/basic_grover.py` -- search an 8-item space for a marked item, print the amplified probability distribution, actual vs theoretical.

Run with `python examples/basic_grover.py` from this directory.
