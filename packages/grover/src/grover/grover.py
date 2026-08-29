"""Grover's search algorithm on a 3-qubit register (N=8 items, 1 marked item).

Convention: for a 3-qubit array `qs`, `qs[0]` is the most significant bit of
the (0..7) item index -- the same convention `packages/qft` uses.

Marked-item oracle
-------------------
`oracle[marked](qs)` flips the phase of exactly the computational basis state
`|marked>` (an integer in [0, 8)) and leaves every other basis state
unchanged: `O|x> = -|x>` if `x == marked`, else `O|x> = |x>`. It's built the
standard way: X-gate the qubits where `marked` has a 0 bit (mapping `|marked>`
to `|111>`), apply a multi-controlled Z on `|111>`, then undo the X gates.

Multi-controlled Z is implemented as `H . (multi-controlled X) . H` on the
last qubit, NOT via `with control(...): z(...)` directly -- see the "Gotcha"
note below and CLAUDE.md for why.

Diffuser
--------
`diffuser(qs)` is the standard Grover diffusion operator (inversion about the
mean): `H^3 . (phase flip about |000>) . H^3`, i.e. `2|s><s| - I` where `|s>`
is the uniform superposition. Implemented the same way as the oracle's phase
flip, just with the roles of "marked state" and `|000>` swapped (X-sandwich
every qubit unconditionally instead of conditionally on `marked`'s bits).

Gotcha: `with control(q0, q1): z(q2)` is broken in guppylang 1.0.2
--------------------------------------------------------------------
A direct multi-controlled Z built with two or more control qubits via the
`control` modifier (`with control(q0, q1): z(q2)`) produces a *wrong* unitary
-- confirmed by comparing against the exact CCZ matrix on a uniform
superposition input (see `tests/test_correctness.py` and CLAUDE.md). The same
`control` modifier with an `x` target (`with control(q0, q1): x(q2)`) is
correct and matches the built-in `toffoli` gate exactly. The workaround used
throughout this module -- `h(target); with control(...): x(target); h(target)`
-- reproduces the exact CCZ (and CCCZ, tested up to 3 controls) unitary to
floating-point precision, so it's used everywhere a multi-controlled phase
flip is needed instead of a direct multi-controlled `z`.

Also note: unlike `qft`/`iqft`, neither `oracle` nor `diffuser` is a
`@guppy.comptime` function -- the `control` modifier used here is explicitly
disallowed inside `@guppy.comptime` bodies (raises `GuppyComptimeError`), so
these are plain `@guppy` functions generic over `marked: nat` using ordinary
guppy `if`/bitwise ops, the same pattern `packages/qft`'s tests already
established as safe. This also means none of `packages/qft`'s CLAUDE.md
gotchas about `@guppy.comptime` monomorphization apply here -- confirmed by
running `grover_search` for all 8 values of `marked` sequentially in a single
process with no corruption (see CLAUDE.md's grover section).
"""

import math

from guppylang import guppy
from guppylang.std.builtins import array, control, nat
from guppylang.std.quantum import h, qubit, x


def optimal_iterations(n_items: int = 8, n_marked: int = 1) -> int:
    """Classical helper: the number of Grover iterations that maximizes the
    marked-item probability, round(pi / (4 * theta) - 1/2) with
    theta = arcsin(sqrt(n_marked / n_items)).

    Not a guppy function -- plain Python, for choosing `iterations` before
    calling `grover_search`.
    """
    theta = math.asin(math.sqrt(n_marked / n_items))
    return round(math.pi / (4 * theta) - 0.5)


@guppy
def oracle[marked: nat](qs: array[qubit, 3]) -> None:
    """Flip the phase of the basis state |marked> (marked in [0, 8))."""
    if (marked >> 2) & 1 == 0:
        x(qs[0])
    if (marked >> 1) & 1 == 0:
        x(qs[1])
    if marked & 1 == 0:
        x(qs[2])
    h(qs[2])
    with control(qs[0], qs[1]):
        x(qs[2])
    h(qs[2])
    if (marked >> 2) & 1 == 0:
        x(qs[0])
    if (marked >> 1) & 1 == 0:
        x(qs[1])
    if marked & 1 == 0:
        x(qs[2])


@guppy
def diffuser(qs: array[qubit, 3]) -> None:
    """Grover diffusion operator: inversion about the mean, 2|s><s| - I."""
    for i in range(3):
        h(qs[i])
    for i in range(3):
        x(qs[i])
    h(qs[2])
    with control(qs[0], qs[1]):
        x(qs[2])
    h(qs[2])
    for i in range(3):
        x(qs[i])
    for i in range(3):
        h(qs[i])


@guppy
def grover_search[marked: nat, iterations: nat](qs: array[qubit, 3]) -> None:
    """Prepare the uniform superposition, then apply `iterations` rounds of
    oracle + diffuser to amplify the amplitude of |marked>.

    For N=8 and 1 marked item, the optimal iteration count is 2 (see
    `grover.optimal_iterations` and CLAUDE.md for the derivation), reaching
    P(marked) ~ 0.945.
    """
    for i in range(3):
        h(qs[i])
    for _ in range(iterations):
        oracle[marked](qs)
        diffuser(qs)
