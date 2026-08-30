"""Grover's search with a multi-item oracle: amplitude amplification for more
than one marked item, on the same 3-qubit (N=8 item) register `packages/grover`
uses. A second-round addition to this registry, kept as its own package
rather than folded into `packages/grover` -- see CLAUDE.md's grover_multi
section for why.

Design question this package exists to answer (from the task brief): now
that `packages/grover`'s gotcha #12 (multi-controlled Z broken) and its
`H . multi-controlled-X . H` workaround are well understood, does marking
*more than one* item need a different gate structure per marked item, or can
the same single-item phase-flip gadget just be repeated/composed across
multiple marked states? **Answer, confirmed both by direct reasoning and by
exact-statevector verification against the closed-form multi-item Grover
formula for several marking patterns (see tests/test_correctness.py): the
existing gadget can simply be repeated, once per marked item, with zero
modification.** Each single-item phase-flip gadget is diagonal in the
computational basis and acts as the identity on every basis state except its
own target, so composing several of them (applying each once, for a
different marked item, in any order) accumulates independent phase flips on
each marked item with no interference between them -- no new bug, no new
gate structure, no per-combination special-casing needed.

`mark_item[item](qs)` -- exactly `packages/grover`'s single-marked-item
`oracle[marked](qs)`, factored out here as an explicitly reusable building
block (the original package inlined this logic directly into `oracle`,
since it only ever needed one). Flips the phase of exactly `|item>`
(`item` in [0, 8)), built the same way as `packages/grover`: X-gate the
qubits where `item` has a 0 bit (mapping `|item>` to `|111>`), apply the
`H . (multi-controlled X) . H` = multi-controlled-Z identity to `|111>`,
undo the X gates. See `packages/grover/CLAUDE.md`-documented gotcha #12 for
why not a direct multi-controlled `z`.

`oracle_2items[item1, item2](qs)` / `oracle_3items[item1, item2, item3](qs)`
-- call `mark_item[...]` once per marked item, nothing else. Separate,
explicitly-arity-typed functions rather than one generic-over-a-list oracle:
guppy's explicit generic-instantiation syntax (`func[arg](...)`, confirmed
working in `packages/grover`'s CLAUDE.md gotcha #15) requires each type
argument to be a literal or closed-over Python constant, not an expression
like `marked[i]` indexing into a runtime/loop-derived collection -- so which
items are marked has to be baked in as separate, named generic parameters,
not iterated over. This is a real, confirmed guppy constraint, not a
stylistic choice -- see CLAUDE.md for the full investigation, including why
a `@guppy.comptime` loop over a Python list of marked items (which *would*
support arbitrary arity) isn't an option here: `control` (needed by
`mark_item`) is disallowed inside `@guppy.comptime` (gotcha #13), so the
oracle-composition function has to be plain `@guppy`, and plain `@guppy`
can't loop over an arbitrary-length Python collection to generate one
generic-instantiated call per element the way comptime could.

`diffuser(qs)` -- unchanged from `packages/grover`, duplicated here rather
than imported (this registry's packages are independent, no cross-package
dependencies -- see root README). The diffusion operator (`2|s><s| - I`)
doesn't depend on which items are marked at all, so there was nothing to
generalize.

`grover_2items[item1, item2, iterations](qs)` /
`grover_3items[item1, item2, item3, iterations](qs)` -- prepare the uniform
superposition, then apply `iterations` rounds of oracle + diffuser, mirroring
`packages/grover`'s `grover_search[marked, iterations]`.

`optimal_iterations(n_items, n_marked)` -- the same formula
`packages/grover` uses (`round(pi / (4*theta) - 1/2)`, `theta =
arcsin(sqrt(n_marked/n_items))`), which already generalizes to any
`n_marked` -- `packages/grover` only ever called it with `n_marked=1`.
"""

from math import asin, pi, sqrt

from guppylang import guppy
from guppylang.std.builtins import array, control, nat
from guppylang.std.quantum import h, qubit, x

N_QUBITS = 3
N_ITEMS = 8  # 2**N_QUBITS


def optimal_iterations(n_items: int = N_ITEMS, n_marked: int = 1) -> int:
    """Classical helper: the number of Grover iterations that maximizes the
    marked-item probability, round(pi / (4 * theta) - 1/2) with
    theta = arcsin(sqrt(n_marked / n_items)). Not a guppy function -- plain
    Python, for choosing `iterations` before calling `grover_2items`/
    `grover_3items`. Identical to `packages/grover`'s helper of the same
    name (it already generalizes to n_marked > 1; that package just never
    called it that way)."""
    theta = asin(sqrt(n_marked / n_items))
    return round(pi / (4 * theta) - 0.5)


@guppy
def mark_item[item: nat](qs: array[qubit, N_QUBITS]) -> None:
    """Flip the phase of the basis state |item> (item in [0, 8))."""
    if (item >> 2) & 1 == 0:
        x(qs[0])
    if (item >> 1) & 1 == 0:
        x(qs[1])
    if item & 1 == 0:
        x(qs[2])
    h(qs[2])
    with control(qs[0], qs[1]):
        x(qs[2])
    h(qs[2])
    if (item >> 2) & 1 == 0:
        x(qs[0])
    if (item >> 1) & 1 == 0:
        x(qs[1])
    if item & 1 == 0:
        x(qs[2])


@guppy
def diffuser(qs: array[qubit, N_QUBITS]) -> None:
    """Grover diffusion operator: inversion about the mean, 2|s><s| - I.
    Unchanged from packages/grover -- does not depend on which items are
    marked."""
    for i in range(N_QUBITS):
        h(qs[i])
    for i in range(N_QUBITS):
        x(qs[i])
    h(qs[2])
    with control(qs[0], qs[1]):
        x(qs[2])
    h(qs[2])
    for i in range(N_QUBITS):
        x(qs[i])
    for i in range(N_QUBITS):
        h(qs[i])


@guppy
def oracle_2items[item1: nat, item2: nat](qs: array[qubit, N_QUBITS]) -> None:
    """Flip the phase of |item1> and |item2> -- the single-item gadget,
    simply called twice. See module docstring."""
    mark_item[item1](qs)
    mark_item[item2](qs)


@guppy
def oracle_3items[item1: nat, item2: nat, item3: nat](qs: array[qubit, N_QUBITS]) -> None:
    """Flip the phase of |item1>, |item2>, and |item3> -- the single-item
    gadget, called three times. See module docstring."""
    mark_item[item1](qs)
    mark_item[item2](qs)
    mark_item[item3](qs)


@guppy
def grover_2items[item1: nat, item2: nat, iterations: nat](qs: array[qubit, N_QUBITS]) -> None:
    """Prepare the uniform superposition, then apply `iterations` rounds of
    oracle_2items + diffuser to amplify |item1> and |item2>."""
    for i in range(N_QUBITS):
        h(qs[i])
    for _ in range(iterations):
        oracle_2items[item1, item2](qs)
        diffuser(qs)


@guppy
def grover_3items[item1: nat, item2: nat, item3: nat, iterations: nat](
    qs: array[qubit, N_QUBITS],
) -> None:
    """Prepare the uniform superposition, then apply `iterations` rounds of
    oracle_3items + diffuser to amplify |item1>, |item2>, and |item3>."""
    for i in range(N_QUBITS):
        h(qs[i])
    for _ in range(iterations):
        oracle_3items[item1, item2, item3](qs)
        diffuser(qs)
