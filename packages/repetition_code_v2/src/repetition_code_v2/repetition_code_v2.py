"""The bit-flip repetition code, generalized to arbitrary (odd) code
distance and repeated syndrome-extraction rounds, in guppylang.

`packages/repetition_code` is fixed at distance 3 (3 physical qubits,
corrects any single bit-flip) and single-round (one syndrome extraction,
one correction, done). This package generalizes both axes the task brief
asked for: code distance as a parameter (distance-5, distance-7, ...,
distance-n for any odd n >= 3), and repeated correction over multiple
rounds -- closer to how a real fault-tolerant QEC experiment actually runs
(continuous syndrome extraction, correcting errors as they accumulate,
not a single one-shot correction).

**Packaging choice, documented as the task brief asked: a new, independent
`packages/repetition_code_v2`, not `packages/repetition_code` generalized
in place.** Same rationale as every other second-round package in this
registry (`vqe_h2` didn't touch `qaoa`; `grover_multi` and
`vqe_h2_stretched` both added new packages rather than changing
`packages/grover`/`packages/vqe_h2`'s already-published, distance-3-only
API): `repetition_code`'s `encode(q0, q1, q2)`/`extract_syndrome(q0, q1,
q2)`/`correct(q0, q1, q2, s0, s1)` signatures are fixed-arity (3 positional
qubits, a 2-tuple syndrome) and could not become generic-over-n without
breaking every existing caller. This package's `encode`/`extract_syndrome`/
`correct` take an `array[qubit, n]` and an `array[bool, n]` instead --
a genuinely different, incompatible signature, not an overload.

Code distance and physical qubit count
----------------------------------------
A distance-n repetition code (n odd) uses n physical qubits in a line and
corrects up to `(n - 1) // 2` simultaneous bit-flip errors (distance-3
corrects 1, distance-5 corrects 2, distance-7 corrects 3, ...) -- the
standard classical-coding-theory result for a length-n repetition code
under minimum-distance (majority-vote) decoding. The task brief phrases
this as "n = 2k + 1 physical qubits for logical distance k"; the code below
uses `n` (the physical qubit / array-length parameter, always odd) as its
one generic parameter throughout, matching this registry's established
generic-over-register-size convention (`packages/qft`'s `qft[n: nat]`) --
see "Is this really a new kind of generic use?" below for why code-distance
genericity turned out to be mechanically identical to that, once one
concrete blocker (below) was worked around.

encode
------
`encode[n: nat](qs: array[qubit, n])`: `CX(qs[0], qs[i])` for i in 1..n-1 --
a direct generalization of `repetition_code.encode`'s star topology
(`CX(q0,q1); CX(q0,q2)`), unchanged in spirit, just looped over `n - 1`
targets instead of hardcoded to 2. Maps `alpha|0>|0>^(n-1) + beta|1>|0>^(n-1)`
(logical state on `qs[0]`, the rest fresh `|0>` ancillas) to
`alpha|0>^n + beta|1>^n`.

extract_syndrome
-----------------
`extract_syndrome[n: nat](qs: array[qubit, n]) -> array[bool, n]`: computes
`n - 1` ancilla-based chain parity checks -- `parity(qs[j], qs[j+1])` for
j = 0..n-2, a direct generalization of `repetition_code.extract_syndrome`'s
two checks (parity(q0,q1), parity(q1,q2)) to n-1 checks over n qubits.
Returns an array of length **n, not n-1** -- see "A new, general (not
comptime-specific) gotcha" below for exactly why, and note the last slot
(`syndrome[n-1]`) is always `False`, unused padding; every real check lives
at `syndrome[0..n-2]`. Like the original, this never measures `qs` itself
directly, so it doesn't collapse the logical state (verified via the exact
statevector across multiple rounds, not just single-shot classical bits --
see tests).

correct
-------
`correct[n: nat](qs: array[qubit, n], syndrome: array[bool, n]) -> None`:
**minimum-weight (majority-vote) decoding**, generalizing
`repetition_code.correct`'s 4-row lookup table (which only had to handle
"no error" or "exactly one error") to arbitrary n and up to
`(n - 1) // 2` simultaneous errors. The chain of n-1 parity checks
(syndrome[j] = e_j XOR e_{j+1}, where e_i is 1 iff qubit i was flipped)
has exactly two error patterns consistent with any given syndrome: a
"candidate" pattern (built by fixing e_0 = False and integrating the
syndrome via a running XOR -- a prefix-XOR / "cumulative parity" scan) and
its bitwise complement (the same syndrome is also produced by flipping
*every* qubit relative to the candidate, since that preserves every
pairwise parity). Real bit-flip errors are assumed sparse (each qubit
independently unlikely to be flipped), so the *lower-weight* (fewer 1s) of
the two candidates is the maximum-likelihood correction -- standard
minimum-distance decoding for a repetition code, and the actual meaning of
"majority vote" here: a qubit gets corrected if flipping it belongs to the
smaller-weight explanation of the observed syndrome, not by looking at any
single qubit's value in isolation (nothing about `qs`'s actual values is
ever read classically -- decoding works from the syndrome alone, same
non-collapsing property `extract_syndrome` has). For n=3 this decoder is
verified to reproduce `repetition_code.correct`'s exact 4-row table by
hand (see tests) -- the generalization is not just "runs without crashing"
at distance 3, its output is identical to the original, row for row.

Repeated rounds
-----------------
`correct_for_rounds[n: nat](qs: array[qubit, n], num_rounds: nat) -> None`:
runs `num_rounds` rounds of `extract_syndrome` + `correct` back to back --
the "continuous correction over multiple rounds" the task brief asked for,
as a first-class exported function (not just a testing pattern). Useful
when the caller doesn't need to inject anything *between* rounds (e.g.
"defend against ambient noise for N rounds"). Tests that need a *specific*
new error injected between particular rounds (the task's actual multi-round
requirement) compose `extract_syndrome`/`correct` manually instead, with
the error injection interleaved between calls -- see tests/test_correctness.py
and "The multi-round driver pattern" below.

A new, general (not comptime-specific) gotcha
-------------------------------------------------
The natural signature for `extract_syndrome` is `-> array[bool, n - 1]`
(n-1 real checks for n qubits) -- but guppylang 1.0.2 rejects this outright:
`array[bool, n - 1]` (or any arithmetic expression -- `n - 1`, `n + 1`, ...
all tried) in a generic array-length type position raises
`GuppyError: Invalid type argument (Not a valid type argument)`. Checked
directly, not assumed to be a one-off: this happens in BOTH a return-type
position and a parameter-type position, and in BOTH plain `@guppy` and
`@guppy.comptime` functions -- so, unlike gotchas #13/#16/#17 (which are
specifically about what comptime disallows), this is a general guppylang
1.0.2 limitation on generic array types, not a comptime-specific one.
**This registry had never attempted a generic array-length *expression*
before this package** -- every earlier generic-over-n function
(`qft[n: nat]`, `mark_item[item: nat]`) only ever used the bare type
variable `n` itself as an array length, never `n` combined with arithmetic.
Workaround, used throughout this module: size every n-related array as
plain `n` (never `n - 1` or similar), and document/enforce which slots are
meaningful vs. padding in the function's own docstring and tests (e.g.
`extract_syndrome`'s last slot, `correct`'s use of only `syndrome[0..n-2]`).
Not filed upstream yet; a minimal repro is kept in this package's test
suite (see gotcha checks in CLAUDE.md).

Is this really a new kind of generic use? (the task brief's explicit question)
----------------------------------------------------------------------------------
CLAUDE.md's "Planned next packages" section noted this registry had never
built anything generic-over-code-distance before, only generic-over-
register-size (`qft[n: nat]`). Having now built it: **generic-over-code-
distance turned out to be mechanically identical to generic-over-register-
size** -- `n` (physical qubit count) directly equals the code distance
here, so `encode[n: nat](qs: array[qubit, n])` is, syntactically and
semantically, indistinguishable from `qft[n: nat](qs: array[qubit, n])`.
The one genuinely new wrinkle wasn't "distance" as a *concept* needing
special generic machinery -- it was the concrete need for an n-1-sized
*second* array (the syndrome), which is where the array-length-arithmetic
gotcha above actually came from. A hypothetical QEC code whose syndrome
array happened to be sized exactly n (not n-1, or n+k for a compile-time
constant k) would not have hit this at all.

Checked against CLAUDE.md's full bug/finding list before writing any of the
above, per the task brief and this registry's standing convention:
gotchas #5 (iqft-isolation/comptime-derived-function cross-contamination),
#12 (`with control(...): z(...)` broken), and #13 (`control`/`dagger`/
`power` forbidden in comptime) all remain not-applicable for the same
reasons `repetition_code`'s own notes gave -- `encode`/`extract_syndrome`/
`correct`/`correct_for_rounds` are all plain `@guppy` functions (no
comptime at all: `range(n)`/`range(1, n)`/`range(num_rounds)` loops over a
generic `nat` are sufficient, no Python-level list/dict needs iterating),
and none of them use `with control(...):` -- `correct`'s qubit selection is
entirely classical (an `if` on array-indexed booleans), exactly like the
original package. Gotchas #16/#17 (comptime-specific generic/array-
construction restrictions) don't apply to the package's own functions for
the same reason; they *would* apply to a test driver that needs comptime
to iterate a Python list of per-round error qubits (see tests), matching
how earlier packages already used comptime driver functions.
"""

from guppylang import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import cx, measure, qubit, x


@guppy
def encode[n: nat](qs: array[qubit, n]) -> None:
    """Encode the logical state on qs[0] (with qs[1..n-1] fresh |0> qubits)
    into the distance-n code: alpha|0>|0>^(n-1) + beta|1>|0>^(n-1) ->
    alpha|0>^n + beta|1>^n."""
    for i in range(1, n):
        cx(qs[0], qs[i])


@guppy
def extract_syndrome[n: nat](qs: array[qubit, n]) -> array[bool, n]:
    """Ancilla-based parity measurement: returns n-1 real chain-parity
    checks (parity(qs[j], qs[j+1]) for j=0..n-2) padded to length n --
    syndrome[n-1] is always False, unused (see module docstring for why
    n, not n-1). Never measures qs directly, so the logical state (any
    superposition) is not collapsed."""
    syndrome = array(False for _ in range(n))
    for j in range(n - 1):
        a = qubit()
        cx(qs[j], a)
        cx(qs[j + 1], a)
        syndrome[j] = measure(a).read()
    return syndrome


@guppy
def correct[n: nat](qs: array[qubit, n], syndrome: array[bool, n]) -> None:
    """Minimum-weight (majority-vote) decoding of `syndrome` (only
    syndrome[0..n-2] are read; syndrome[n-1] is the unused padding slot --
    see `extract_syndrome`), applying `X` to whichever qubits the
    lower-weight of the two syndrome-consistent error patterns identifies.
    Corrects up to (n-1)//2 simultaneous bit-flip errors. See module
    docstring for the full derivation; for n=3 this reproduces
    `repetition_code.correct`'s 4-row table exactly (verified by hand and
    in tests, not just "still passes")."""
    candidate = array(False for _ in range(n))
    for i in range(1, n):
        candidate[i] = candidate[i - 1] != syndrome[i - 1]
    weight = 0
    for i in range(n):
        if candidate[i]:
            weight = weight + 1
    use_complement = weight * 2 > n
    for i in range(n):
        if candidate[i] != use_complement:
            x(qs[i])


@guppy
def correct_for_rounds[n: nat](qs: array[qubit, n], num_rounds: nat) -> None:
    """Runs `num_rounds` back-to-back rounds of extract_syndrome + correct
    -- continuous correction, not a single one-shot fix. See module
    docstring's "Repeated rounds" section for when to use this vs.
    composing extract_syndrome/correct manually (needed when a specific new
    error must be injected between particular rounds, e.g. in tests)."""
    for _ in range(num_rounds):
        syndrome = extract_syndrome(qs)
        correct(qs, syndrome)
