"""The 3-qubit bit-flip repetition code, in guppylang.

A single logical qubit alpha|0> + beta|1> is encoded into 3 physical qubits
as alpha|000> + beta|111>. If at most one of the three physical qubits
suffers a bit-flip (X) error, the original logical state can be recovered
exactly via majority-vote error correction -- the textbook first example of
quantum error correction (e.g. Nielsen & Chuang, "Quantum Computation and
Quantum Information", section 10.1). Like every classical repetition code,
this only protects against bit-flip (X) errors, not phase-flip (Z) errors;
it is not a substitute for a real, phase-protecting code (e.g. Shor's 9-qubit
code, or the surface code) -- it exists here as the simplest, canonical
worked example of the encode -> extract syndrome -> correct pattern.

Convention: qs[0] is the "primary" physical qubit -- the one the logical
state is initially prepared on before `encode` spreads it across all three
(matching the rest of this registry's `qs[0]` = most-significant/primary
convention, e.g. `packages/qft`).

encode
------
`encode(q0, q1, q2)`: `CX(q0, q1); CX(q0, q2)`. Maps alpha|0>|00> + beta|1>|00>
(logical state on q0, q1/q2 fresh ancillas in |0>) to alpha|000> + beta|111>.

extract_syndrome
-----------------
`extract_syndrome(q0, q1, q2) -> tuple[bool, bool]`: allocates two fresh
ancilla qubits and uses them, via CNOTs, to measure the *parities*
Z0*Z1 and Z1*Z2 -- which pair(s) of adjacent code qubits disagree -- without
ever directly measuring q0, q1, or q2 themselves. This is the whole point of
syndrome extraction: the ancilla-mediated parity measurement reveals which
qubit (if any) was flipped without revealing -- and so without collapsing --
the logical superposition itself. Verified directly, not just assumed (see
CLAUDE.md and tests/test_correctness.py): preparing a logical superposition,
injecting an error, running extract_syndrome + correct, and reading out the
*exact* statevector reproduces the original GHZ-like superposition
(alpha|000> + beta|111>) to ~1e-16, not just the correct value on a single
sampled shot.

Syndrome table (s0, s1) = (parity(q0,q1), parity(q1,q2)):

| s0 | s1 | meaning          |
|----|----|------------------|
| 0  | 0  | no error         |
| 1  | 0  | error on q0      |
| 1  | 1  | error on q1      |
| 0  | 1  | error on q2      |

correct
-------
`correct(q0, q1, q2, s0, s1)`: applies `X` to whichever qubit the syndrome
table above identifies, classically controlled on the two measured syndrome
bits (plain guppy `if`, not a quantum `with control(...):` -- there is no
quantum control here at all, only classical feedforward from two prior
measurement outcomes, per the task's "classically-controlled correction").
A no-error syndrome (0, 0) correctly triggers no correction.

Limitation (by design, not a bug): this code can correct any *single*
bit-flip among the three qubits, but not two or more simultaneous ones (a
double error is misdiagnosed as a single error on the *third*, undamaged
qubit, and "corrected" into the wrong codeword) -- a fundamental limitation
of the 3-qubit code itself, not a guppylang or implementation issue.

Checked against CLAUDE.md's full bug/finding list before writing any of the
above, per the task brief -- mid-circuit measurement (`extract_syndrome`
measures ancillas partway through the circuit, not at the very end) and
classical control flow driven by measurement outcomes (`correct`) are new
territory for this registry (no earlier package used either), so this was
checked carefully rather than assumed clean. Short version, see CLAUDE.md for
the full writeup: none of the comptime-related gotchas (#5, #13, #16, #17)
apply -- `encode`/`extract_syndrome`/`correct` are all plain `@guppy`
functions; this package needs no `@guppy.comptime` at all, since there is no
Python-level data structure (an edge list, a Pauli-term dict) to iterate --
the 3-qubit syndrome table is small and fixed, just three plain `if`
statements. Gotcha #12 (multi-controlled Z broken) doesn't apply -- no
`with control(...):` is used anywhere in this package; `correct`'s
qubit-to-flip selection is entirely classical (an `if` on two measured
bools), not quantum control. This is, refreshingly, the first package in
this registry where a careful check of the bug list turned up nothing new to
route around -- mid-circuit measurement and measurement-driven classical
control flow both worked exactly as documented on the first attempt.
"""

from guppylang import guppy
from guppylang.std.quantum import cx, measure, qubit, x


@guppy
def encode(q0: qubit, q1: qubit, q2: qubit) -> None:
    """Encode the logical state on q0 (with q1, q2 fresh |0> qubits) into the
    3-qubit code: alpha|0>|00> + beta|1>|00> -> alpha|000> + beta|111>."""
    cx(q0, q1)
    cx(q0, q2)


@guppy
def extract_syndrome(q0: qubit, q1: qubit, q2: qubit) -> tuple[bool, bool]:
    """Ancilla-based parity measurement: returns (parity(q0,q1), parity(q1,q2))
    without measuring q0/q1/q2 directly, so the logical state (including any
    superposition) is not collapsed. See module docstring for the syndrome
    table."""
    a0 = qubit()
    a1 = qubit()
    cx(q0, a0)
    cx(q1, a0)
    cx(q1, a1)
    cx(q2, a1)
    s0 = measure(a0).read()
    s1 = measure(a1).read()
    return s0, s1


@guppy
def correct(q0: qubit, q1: qubit, q2: qubit, s0: bool, s1: bool) -> None:
    """Apply the correction identified by the (s0, s1) syndrome (see module
    docstring's table) -- classically controlled, not quantum-controlled."""
    if s0 and not s1:
        x(q0)
    if s0 and s1:
        x(q1)
    if s1 and not s0:
        x(q2)
