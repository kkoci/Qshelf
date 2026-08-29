"""Correctness tests for the grover package, run on Selene's Quest (statevector) backend.

Methodology
-----------
Grover's algorithm is non-Clifford in general: the multi-controlled Z used by
both the oracle and the diffuser is built from a multi-controlled X (Toffoli,
itself Clifford+non-Clifford depending on decomposition) sandwiched by
Hadamards, and more fundamentally a 2-controlled Z (CCZ) is a canonical
non-Clifford gate (it's part of the standard Clifford+CCZ/Toffoli universal
gate set precisely because it lies outside the Clifford group). So, same
reasoning as `packages/qft` (see its tests and CLAUDE.md): Selene's Stim
(stabilizer) backend cannot simulate this circuit exactly, and these tests use
`statevector_sim()` (Quest) instead.

For a 3-qubit register (N=8 items, 1 marked item) we:
  1. Verify the oracle and diffuser in isolation against their exact,
     hand-derivable action (oracle flips only the marked amplitude; the
     diffuser fixes the uniform superposition; the oracle is self-inverse).
  2. Verify the full `grover_search[marked, iterations]` circuit's output
     statevector against the closed-form Grover formula
     (amplitude of |marked> = sin((2r+1)*theta), amplitude of every other
     state = cos((2r+1)*theta)/sqrt(N-1), theta = arcsin(sqrt(1/N))) for
     every possible marked item and several iteration counts, including
     that the marked-item probability follows the theoretical curve as
     iterations increase (and decreases past the optimum -- Grover
     "overshoots" if you iterate too many times, which is itself a good
     correctness signal: a broken/no-op circuit could not accidentally
     reproduce that shape).

Global phase: the full `grover_search` circuit is built from real-valued
gates (H, X, and CCZ-via-H-CCX-H) but the overall SIGN of the resulting
statevector is not fixed by the algorithm (it's a physically meaningless
global phase) and, empirically, this particular circuit's sign convention
depends on `marked` in a way we didn't bother deriving by hand. Tests that
compare against the closed-form formula align the sign first (flip the
reference vector if its inner product with the actual result is negative)
before comparing -- see `_phase_aligned`. This is standard practice for
state-vector-level tests, not a workaround for a bug: probabilities
(`|amplitude|**2`), which are what's physically observable, match without
any alignment (see `test_grover_probability_matches_theory_unaligned`).

Why no subprocess isolation here (unlike packages/qft)
--------------------------------------------------------
packages/qft's tests isolate each case in its own subprocess because of a
confirmed guppylang 1.0.2 bug where `iqft` (built by daggering/reversing
`qft`, both `@guppy.comptime`) behaves differently in isolation vs combined
with `qft` in the same process -- see CLAUDE.md gotcha #5. Neither `oracle`
nor `diffuser` here uses `@guppy.comptime` at all (the `control` modifier they
both rely on is explicitly disallowed inside comptime bodies -- see
grover.py's module docstring), so that bug's precondition doesn't apply. We
confirmed this directly: `test_grover_matches_theory_at_optimal_iterations`
below sweeps `marked` across all 8 values and `test_probability_curve_matches_theory`
separately sweeps `iterations`, both as ordinary parametrized pytest tests
running many monomorphizations of the same generic functions in one process,
and both passed reliably across repeated runs during development -- no
cross-contamination observed. See CLAUDE.md's grover section for the full
writeup, including the one new bug this package *did* find (multi-controlled
Z via `with control(...): z(...)`, unrelated to qft's bugs).

Why `_run` retries
-------------------
Same Windows Application Control policy transient-block issue documented in
CLAUDE.md gotcha #3: the Selene runner subprocess spawned by `.emulator(...).run()`
occasionally (and transiently) gets blocked the first time a fresh process
touches it, then succeeds immediately on retry. Every test below goes through
`_run()`, which retries a few times, so the suite is robust to this rather
than flaking on an unrelated environment issue.
"""

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.emulator.result import EmulatorResult
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit

from grover import diffuser, grover_search, oracle, optimal_iterations

N = 8  # 2**3, fixed register size this package implements
UNIFORM = np.full(N, 1 / np.sqrt(N))


def _run(main: GuppyFunctionDefinition, n_qubits: int = 3) -> np.ndarray:
    """Compile, run, and return the "out"-tagged statevector, retrying a
    couple of times if the OS transiently blocks the emulator subprocess
    (see module docstring)."""
    main.check()
    last_error: OSError | None = None
    for _attempt in range(3):
        try:
            shots: EmulatorResult = (
                main.with_minimal_opt()
                .emulator(n_qubits=n_qubits)
                .statevector_sim()
                .with_seed(0)
                .run()
            )
            return shots.partial_state_dicts()[0]["out"].as_single_state()
        except OSError as exc:
            last_error = exc
    raise AssertionError(f"emulator run failed after retries: {last_error}")


def _theoretical(marked: int, iterations: int) -> np.ndarray:
    theta = np.arcsin(np.sqrt(1 / N))
    amp_marked = np.sin((2 * iterations + 1) * theta)
    amp_other = np.cos((2 * iterations + 1) * theta) / np.sqrt(N - 1)
    v = np.full(N, amp_other, dtype=complex)
    v[marked] = amp_marked
    return v


def _phase_aligned(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip `reference`'s global sign to match `actual`'s, if needed."""
    if np.vdot(actual, reference).real < 0:
        return -reference
    return reference


@pytest.mark.parametrize("marked", range(N))
def test_oracle_flips_only_marked_amplitude(marked: int) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        for i in range(3):
            h(qs[i])
        oracle[marked](qs)
        state_output("out", qs)
        discard_array(qs)

    v = _run(main)
    expected = UNIFORM.astype(complex).copy()
    expected[marked] *= -1
    expected = _phase_aligned(v, expected)
    np.testing.assert_allclose(v, expected, atol=1e-8)


@pytest.mark.parametrize("marked", range(N))
def test_oracle_is_self_inverse(marked: int) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        for i in range(3):
            h(qs[i])
        oracle[marked](qs)
        oracle[marked](qs)
        state_output("out", qs)
        discard_array(qs)

    v = _run(main)
    np.testing.assert_allclose(v, UNIFORM, atol=1e-8)


def test_diffuser_fixes_uniform_superposition() -> None:
    """2|s><s| - I applied to |s> is the identity: |s> is the diffuser's
    +1 eigenvector."""

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        for i in range(3):
            h(qs[i])
        diffuser(qs)
        state_output("out", qs)
        discard_array(qs)

    v = _run(main)
    np.testing.assert_allclose(v, UNIFORM, atol=1e-8)


@pytest.mark.parametrize("marked", range(N))
def test_grover_matches_theory_at_optimal_iterations(marked: int) -> None:
    iterations = optimal_iterations(N, 1)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        grover_search[marked, iterations](qs)
        state_output("out", qs)
        discard_array(qs)

    v = _run(main)
    expected = _phase_aligned(v, _theoretical(marked, iterations))
    np.testing.assert_allclose(v, expected, atol=1e-8)


def test_grover_probability_matches_theory_unaligned() -> None:
    """Same as above, but comparing |amplitude|**2 directly -- the physically
    observable quantity, which needs no phase alignment at all."""
    marked = 5
    iterations = optimal_iterations(N, 1)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        grover_search[marked, iterations](qs)
        state_output("out", qs)
        discard_array(qs)

    v = _run(main)
    theory = _theoretical(marked, iterations)
    np.testing.assert_allclose(np.abs(v) ** 2, np.abs(theory) ** 2, atol=1e-8)


def test_probability_curve_matches_theory() -> None:
    """Sweep the iteration count and check P(marked) follows the theoretical
    amplitude-amplification curve, including "overshooting" past the optimum
    -- direct evidence of genuine amplitude amplification, not a fluke."""
    marked = 3
    for iterations in range(6):
        @guppy
        def main() -> None:
            qs = array(qubit() for _ in range(3))
            grover_search[marked, iterations](qs)
            state_output("out", qs)
            discard_array(qs)

        v = _run(main)
        actual_p = abs(v[marked]) ** 2
        theory_p = abs(_theoretical(marked, iterations)[marked]) ** 2
        assert actual_p == pytest.approx(theory_p, abs=1e-8), (
            f"iterations={iterations}: actual P(marked)={actual_p}, theory={theory_p}"
        )

    # Sanity: probability at iterations=0 is the uniform baseline 1/N, and
    # peaks near the optimal iteration count (2) before falling again.
    baseline = abs(_theoretical(marked, 0)[marked]) ** 2
    optimal_p = abs(_theoretical(marked, optimal_iterations(N, 1))[marked]) ** 2
    assert baseline == pytest.approx(1 / N, abs=1e-8)
    assert optimal_p > 0.9
    assert optimal_p > baseline


@pytest.mark.xfail(
    reason=(
        "with control(q0, q1): z(q2) is broken in guppylang 1.0.2 -- produces "
        "a wrong (non-CCZ) unitary. See CLAUDE.md grover section. Kept to "
        "document the quirk; grover.py uses the H-CCX-H workaround instead."
    ),
    strict=True,
)
def test_direct_multi_controlled_z_is_broken() -> None:
    from guppylang.std.builtins import control
    from guppylang.std.quantum import z

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        for i in range(3):
            h(qs[i])
        with control(qs[0], qs[1]):
            z(qs[2])
        state_output("out", qs)
        discard_array(qs)

    v = _run(main)
    expected = UNIFORM.astype(complex).copy()
    expected[7] *= -1  # true CCZ on |+++> only flips |111>
    np.testing.assert_allclose(v, expected, atol=1e-8)
