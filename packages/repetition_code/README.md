# repetition_code

The 3-qubit bit-flip repetition code, implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package -- the textbook first example of quantum error correction (encode, extract syndrome, correct). The final planned package in this registry's initial set.

## What's here

- `encode(q0, q1, q2)` -- encodes the logical state on `q0` (with `q1`, `q2` as fresh `|0>` qubits) into the 3-qubit code: `alpha|0>|00> + beta|1>|00>` -> `alpha|000> + beta|111>`.
- `extract_syndrome(q0, q1, q2) -> tuple[bool, bool]` -- ancilla-based parity measurement, returning `(parity(q0,q1), parity(q1,q2))` *without measuring q0/q1/q2 directly* -- so it doesn't collapse the logical state, including any superposition. See "Correctness" below.
- `correct(q0, q1, q2, s0, s1)` -- applies the single-qubit `X` correction identified by the syndrome (classically controlled -- plain `if` statements on the two measured bits, no quantum control at all).

Corrects any *single* bit-flip (X) error among the three qubits; like every classical repetition code, it does not protect against phase-flip (Z) errors, and cannot correct two or more simultaneous errors (that's a fundamental limitation of the 3-qubit code itself, not this implementation).

## Install

From this directory:

```
pip install -e .
```

or add `repetition-code @ file:///path/to/guppy-registry/packages/repetition_code` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, x, collect_measurements, measure_array
from repetition_code import encode, extract_syndrome, correct

@guppy
def main() -> None:
    qs = array(qubit() for _ in range(3))
    x(qs[0])              # prepare logical |1>
    encode(qs[0], qs[1], qs[2])
    x(qs[1])               # simulate a bit-flip error on qubit 1
    s0, s1 = extract_syndrome(qs[0], qs[1], qs[2])
    correct(qs[0], qs[1], qs[2], s0, s1)
    output("bits", collect_measurements(measure_array(qs)))  # -> 111
```

## Backend: this is the first package in this registry that's genuinely Clifford

`encode`/`extract_syndrome`/`correct` only ever use `H`, `X`, `CX`, and Z-basis measurement -- no continuous-angle rotation gate anywhere. So, unlike `qft`/`grover`/`qaoa`/`vqe_h2` (all of which need Selene's Quest statevector backend because they're non-Clifford), Selene's **Stim (stabilizer) backend** is not just usable here but the semantically appropriate, efficient choice -- confirmed directly, not just assumed from "well, it's Clifford" reasoning, by running the same circuits on both backends and checking they agree (see `tests/test_correctness.py::test_stabilizer_sim_reproduces_statevector_sim_results`).

## Correctness

Three properties are verified (see `tests/test_correctness.py` for the full methodology):

1. **Recovery**: for both logical values and all 4 single-error scenarios (no error, error on q0/q1/q2), the final corrected qubits exactly match the original logical value.
2. **Diagnosis**: the syndrome itself (before correction) matches the expected table for each error scenario.
3. **Non-collapse** (the actual defining property of a working syndrome-based code, not just "outputs the right bit"): preparing a genuine logical superposition, injecting an error, and running `extract_syndrome` + `correct` reproduces the *exact* original superposition (checked via the exact statevector, not sampled statistics) to ~1e-16. A naive "always output the majority vote" implementation could pass property 1 without this being true.

## Gotchas

Closing over Python `None` as a sentinel value and comparing it against an `int` inside a `@guppy` body (`if error_qubit == 0:` where `error_qubit`'s captured value is `None` for one test case) fails: `GuppyTypeError: Operator not defined ... for None and int`. guppy infers the *static* type of a closed-over Python value from what it actually is at that point, and `None` becomes `NoneType`, which has no `==` against `int`. Use a same-type sentinel instead (this package's tests use `-1` for "no error", not `None`). See CLAUDE.md.

Full writeup, including the check against CLAUDE.md's full bug list for mid-circuit measurement and measurement-driven classical control flow (new territory for this registry -- and, refreshingly, found to work exactly as documented, no new bugs beyond the sentinel-type gotcha above), is in the root [CLAUDE.md](../../CLAUDE.md).

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Runs real circuits on Selene's Stim backend and checks the three properties above for every logical value / error location combination.

## Examples

- `examples/basic_repetition_code.py` -- encodes a logical `|1>`, runs all 4 error scenarios, and prints the syndrome and corrected result for each.

Run with `python examples/basic_repetition_code.py` from this directory.
