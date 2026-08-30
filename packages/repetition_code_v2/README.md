# repetition_code_v2

The bit-flip repetition code, generalized to arbitrary (odd) code distance and repeated syndrome-extraction rounds, implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package. `packages/repetition_code` is fixed at distance 3, single-round; this package generalizes both axes.

## Why this package exists

`packages/repetition_code` corrects any *single* bit-flip error among exactly 3 physical qubits, once, then stops -- a good textbook first example, but a long way from how real fault-tolerant QEC actually operates: larger code distance (more physical qubits per logical qubit, correcting more simultaneous errors) and *continuous* correction across many rounds, not a single one-shot fix. This package generalizes both.

**Packaging choice, documented as the task brief asked: a new, independent `packages/repetition_code_v2`, not `packages/repetition_code` generalized in place.** Same rationale as every other second-round package in this registry (`vqe_h2_stretched` and `grover_multi` both added new packages rather than changing already-published APIs): `repetition_code`'s `encode(q0, q1, q2)` / `extract_syndrome(q0, q1, q2) -> tuple[bool, bool]` / `correct(q0, q1, q2, s0, s1)` are fixed-arity, 3-qubit-specific signatures that cannot become generic-over-distance without breaking every existing caller. This package's functions take an `array[qubit, n]` and an `array[bool, n]` instead -- a genuinely different, incompatible shape, not an overload of the original.

## What's here

- `encode[n: nat](qs: array[qubit, n])` -- encodes the logical state on `qs[0]` (with `qs[1..n-1]` fresh `|0>` qubits) into the distance-n code, generalizing `repetition_code.encode`'s star topology (`CX(q0,q1); CX(q0,q2)`) to `CX(qs[0], qs[i])` for every `i`.
- `extract_syndrome[n: nat](qs: array[qubit, n]) -> array[bool, n]` -- `n - 1` ancilla-based chain parity checks (`parity(qs[j], qs[j+1])`), generalizing `repetition_code.extract_syndrome`'s two checks. Returns an array of length **n, not n-1** -- see "Gotchas" below for why; the last slot is always `False`, unused padding.
- `correct[n: nat](qs: array[qubit, n], syndrome: array[bool, n])` -- **minimum-weight (majority-vote) decoding**, correcting up to `(n-1)//2` simultaneous bit-flip errors (distance-3: 1, distance-5: 2, distance-7: 3, ...), generalizing `repetition_code.correct`'s 4-row lookup table. For n=3, reproduces that table exactly, row for row (verified by hand -- see `repetition_code_v2.py`'s module docstring and tests).
- `correct_for_rounds[n: nat](qs: array[qubit, n], num_rounds: nat)` -- runs `num_rounds` back-to-back rounds of `extract_syndrome` + `correct`: continuous correction, not a single fix. Tests that need a *specific* error injected between particular rounds compose `extract_syndrome`/`correct` manually instead (see tests).

Like `repetition_code`, this only protects against bit-flip (X) errors, not phase-flip (Z) errors, and correcting more than `(n-1)//2` simultaneous errors in a single round can misdecode -- a fundamental limit of the code itself (confirmed directly, see "Correctness" below), not a bug.

## Install

From this directory:

```
pip install -e .
```

or add `repetition-code-v2 @ file:///path/to/guppy-registry/packages/repetition_code_v2` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, x, collect_measurements, measure_array
from repetition_code_v2 import encode, extract_syndrome, correct

N = 5  # distance-5: corrects up to 2 simultaneous errors

@guppy
def main() -> None:
    qs = array(qubit() for _ in range(N))
    x(qs[0])              # prepare logical |1>
    encode(qs)
    x(qs[1])               # simulate bit-flip errors on qubits 1 and 3
    x(qs[3])
    syndrome = extract_syndrome(qs)
    correct(qs, syndrome)
    output("bits", collect_measurements(measure_array(qs)))  # -> 11111
```

## The generic-over-distance question, answered

CLAUDE.md's roadmap noted this registry had never built anything generic-over-code-distance before, only generic-over-register-size (`packages/qft`'s `qft[n: nat]`). Having built it: **generic-over-code-distance turned out to be mechanically identical to generic-over-register-size** -- code distance directly equals the physical qubit count `n` here, so `encode[n: nat](qs: array[qubit, n])` is syntactically and semantically indistinguishable from `qft[n: nat](qs: array[qubit, n])`. The genuinely new wrinkle wasn't "distance" as a concept -- it was the concrete need for an *n-1*-sized second array (the syndrome). See "Gotchas" below.

## Gotchas

- **A new, general (not comptime-specific) restriction: guppylang 1.0.2 rejects arithmetic in a generic array-length type position.** `array[bool, n - 1]` (the natural type for `extract_syndrome`'s n-1 real checks) raises `GuppyError: Invalid type argument (Not a valid type argument)` -- checked in both parameter and return-type position, and in both plain `@guppy` and `@guppy.comptime` functions; all four combinations fail identically. This registry's earlier generic-over-n functions (`qft[n: nat]`, `mark_item[item: nat]`) never combined a generic `nat` with arithmetic in a type position, so this hadn't come up before. Workaround used throughout this package: size every n-related array as plain `n`, never `n - 1`, and document which slots are meaningful vs. unused padding.
- **A second, narrower finding, found while testing `correct_for_rounds`: a bare Python integer literal passed to a `nat`-typed parameter, inside a `@guppy.comptime`-traced call, can be inferred as `int` instead of `nat`**, raising `GuppyComptimeError: Type mismatch: Expected argument of type nat, got int`. Reproduced twice, deterministically, running this package's test suite under pytest -- but **two independent, honest attempts at a minimal isolated repro (a single-purpose script with the same call shape) did NOT reproduce it**, so this is recorded as real-but-not-yet-fully-isolated, not a fully characterized general rule (see CLAUDE.md for the full story, including what was and wasn't ruled out). Workaround: wrap integer-literal arguments to `nat`-typed parameters with an explicit `nat(...)` call in comptime code, which reliably avoids it either way.

Full writeup, including the check against CLAUDE.md's full bug list (gotchas #5/#12/#13/#16/#17 all confirmed not applicable, same reasoning `packages/repetition_code` gave), is in the root [CLAUDE.md](../../CLAUDE.md).

## Correctness

Matching `packages/repetition_code`'s three properties, generalized, plus the two new axes this package adds (see `tests/test_correctness.py` for the full methodology):

1. **Distance-3 backward compatibility**: this generalization's distance-3 syndrome values and recovered bits are checked against `repetition_code`'s own published 4-row table and recovery behavior -- confirms generalizing didn't silently change the base case.
2. **Distance-5/7 recovery**: every single-qubit error at both new distances, and every combination of up to `(n-1)//2` simultaneous errors (2 for distance-5, 3 for distance-7, not just 1).
3. **The fundamental limit still holds, generalized**: exceeding a distance's correction capacity (3 simultaneous errors on a distance-5 code) can misdecode -- confirmed with one concrete example, not just asserted from theory.
4. **Multi-round survival**: a *new* error injected between each of several rounds (not a single one-shot error) is corrected every round, checked via both final classical bits and, for a genuine logical superposition, the exact statevector (the actual defining property of a non-collapsing syndrome-based code, now checked across multiple rounds).
5. **Stim/Quest agreement at distance 5**: this circuit family is Clifford at every distance (H, X, CX, Z-measurement only), re-confirmed beyond distance 3 rather than assumed.

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Runs real circuits on Selene's Stim backend (this circuit family is Clifford at every distance tested). Faster than this registry's variational packages (`qaoa`/`vqe_h2`/`vqe_h2_stretched`) -- no classical optimization loop, one circuit compile per test case.

## Examples

- `examples/basic_repetition_code_v2.py` -- distance-5 and distance-7 single-round recovery (including simultaneous errors within each distance's budget), and a multi-round run with a fresh error injected between each round.

Run with `python examples/basic_repetition_code_v2.py` from this directory.
