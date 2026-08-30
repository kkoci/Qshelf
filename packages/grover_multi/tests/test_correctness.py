"""Correctness tests for the grover_multi package, run on Selene's Quest
(statevector) backend -- same non-Clifford reasoning as `packages/grover`
(the multi-controlled phase flip is built via `H . multi-controlled-X . H`,
and while X/H/CX are individually Clifford, the *composition* used to
synthesize a multi-controlled Z is not simulable by Stim -- confirmed
empirically in `packages/grover`'s own test suite; not re-derived here).

This package's central question (from the task brief): now that
`packages/grover`'s gotcha #12 (`with control(q0, q1): z(target)` broken)
and its `H . multi-controlled-X . H` workaround are understood, does marking
*more than one* item need a different gate structure, or does the existing
single-item gadget just compose by repetition? Every test below is, in one
way or another, evidence for the answer documented in grover_multi.py's
module docstring and CLAUDE.md: **it just composes by repetition, exactly,
with no new bug and no per-combination special-casing.**

Methodology, mirroring packages/grover's rigor:
  1. `test_oracle_flips_only_marked_amplitudes`: the multi-item oracles, in
     isolation, flip *exactly* the marked amplitudes and leave every other
     basis state's amplitude untouched -- direct evidence that composing
     several single-item gadgets doesn't leak into or otherwise disturb the
     unmarked amplitudes (the risk the task brief was asking us to check
     for).
  2. `test_grover_matches_theory`: the full oracle+diffuser circuit's exact
     statevector matches the closed-form *multi-item* Grover formula
     (amplitude of each marked state = sin((2r+1)*theta)/sqrt(k), amplitude
     of each unmarked state = cos((2r+1)*theta)/sqrt(N-k), theta =
     arcsin(sqrt(k/N))) for several different marking patterns and k=2,3.
  3. `test_probability_curve_matches_theory`: sweeping the iteration count
     for a fixed multi-item pattern reproduces the theoretical curve
     (baseline -> peak -> overshoot), the actual evidence of genuine
     amplitude amplification for k>1, not just a lucky single data point.
  4. `test_direct_multi_controlled_z_is_broken`: the same regression test
     `packages/grover` keeps (this package is independent, doesn't import
     from `packages/grover`, so it's re-verified here rather than assumed).

Global phase: like `packages/grover`, every gate here is real-valued (H, X,
CX -- no rz/rx/ry), so a +-1 sign flip is all the phase alignment needed
(see CLAUDE.md gotcha #18 for why this differs for `packages/qaoa`).
"""

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit

from grover_multi import (
    N_ITEMS,
    N_QUBITS,
    diffuser,
    grover_2items,
    grover_3items,
    mark_item,
    optimal_iterations,
    oracle_2items,
    oracle_3items,
)

UNIFORM = np.full(N_ITEMS, 1 / np.sqrt(N_ITEMS))


def _theoretical(marked: tuple[int, ...], iterations: int) -> np.ndarray:
    k = len(marked)
    theta = np.arcsin(np.sqrt(k / N_ITEMS))
    amp_marked = np.sin((2 * iterations + 1) * theta) / np.sqrt(k)
    amp_other = np.cos((2 * iterations + 1) * theta) / np.sqrt(N_ITEMS - k)
    v = np.full(N_ITEMS, amp_other, dtype=complex)
    for m in marked:
        v[m] = amp_marked
    return v


def _phase_aligned(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if np.vdot(actual, reference).real < 0:
        return -reference
    return reference


TWO_ITEM_PATTERNS = [(2, 5), (0, 7), (1, 6)]
THREE_ITEM_PATTERNS = [(1, 3, 6), (0, 2, 4)]


@pytest.mark.parametrize("item", range(N_ITEMS))
def test_mark_item_flips_only_its_own_amplitude(item: int) -> None:
    """The reusable single-item building block, in isolation -- identical in
    spirit to packages/grover's oracle test, and the base case every
    multi-item composition above builds on."""

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        for i in range(N_QUBITS):
            h(qs[i])
        mark_item[item](qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    expected = UNIFORM.astype(complex).copy()
    expected[item] *= -1
    expected = _phase_aligned(v, expected)
    np.testing.assert_allclose(v, expected, atol=1e-8)


@pytest.mark.parametrize("marked", TWO_ITEM_PATTERNS)
def test_oracle_2items_flips_only_marked_amplitudes(marked: tuple[int, int]) -> None:
    m0, m1 = marked

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        for i in range(N_QUBITS):
            h(qs[i])
        oracle_2items[m0, m1](qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    expected = UNIFORM.astype(complex).copy()
    expected[m0] *= -1
    expected[m1] *= -1
    expected = _phase_aligned(v, expected)
    np.testing.assert_allclose(v, expected, atol=1e-8)


@pytest.mark.parametrize("marked", THREE_ITEM_PATTERNS)
def test_oracle_3items_flips_only_marked_amplitudes(marked: tuple[int, int, int]) -> None:
    m0, m1, m2 = marked

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        for i in range(N_QUBITS):
            h(qs[i])
        oracle_3items[m0, m1, m2](qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    expected = UNIFORM.astype(complex).copy()
    for m in marked:
        expected[m] *= -1
    expected = _phase_aligned(v, expected)
    np.testing.assert_allclose(v, expected, atol=1e-8)


def test_diffuser_fixes_uniform_superposition() -> None:
    """Unchanged from packages/grover: |s> is the diffuser's +1 eigenvector,
    independent of which items are (going to be) marked."""

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        for i in range(N_QUBITS):
            h(qs[i])
        diffuser(qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    np.testing.assert_allclose(v, UNIFORM, atol=1e-8)


@pytest.mark.parametrize("marked", TWO_ITEM_PATTERNS)
def test_grover_2items_matches_theory(marked: tuple[int, int]) -> None:
    m0, m1 = marked
    iterations = optimal_iterations(N_ITEMS, 2)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        grover_2items[m0, m1, iterations](qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    expected = _phase_aligned(v, _theoretical(marked, iterations))
    np.testing.assert_allclose(v, expected, atol=1e-8)


@pytest.mark.parametrize("marked", THREE_ITEM_PATTERNS)
def test_grover_3items_matches_theory(marked: tuple[int, int, int]) -> None:
    m0, m1, m2 = marked
    iterations = optimal_iterations(N_ITEMS, 3)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        grover_3items[m0, m1, m2, iterations](qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    expected = _phase_aligned(v, _theoretical(marked, iterations))
    np.testing.assert_allclose(v, expected, atol=1e-8)


def test_probability_curve_matches_theory_for_2items() -> None:
    """Sweep the iteration count for a fixed 2-item pattern and check
    P(either marked item) follows the theoretical amplitude-amplification
    curve, including "overshooting" past the optimum -- the actual evidence
    of genuine amplitude amplification for k=2, not a single lucky data
    point."""
    marked = (2, 5)
    m0, m1 = marked
    for iterations in range(4):
        @guppy
        def main() -> None:
            qs = array(qubit() for _ in range(N_QUBITS))
            grover_2items[m0, m1, iterations](qs)
            state_output("out", qs)
            discard_array(qs)

        main.check()
        shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
        v = shots.partial_state_dicts()[0]["out"].as_single_state()
        actual_p = sum(abs(v[m]) ** 2 for m in marked)
        theory = _theoretical(marked, iterations)
        theory_p = sum(abs(theory[m]) ** 2 for m in marked)
        assert actual_p == pytest.approx(theory_p, abs=1e-8), (
            f"iterations={iterations}: actual P={actual_p}, theory={theory_p}"
        )

    # Sanity: baseline (r=0) is 2/8, and N=8,k=2's optimum (r=1) reaches
    # exactly 1.0 -- a mathematically "perfect" Grover case (theta=pi/6, so
    # (2*1+1)*theta = pi/2 exactly), a good independent check that the
    # formula itself, not just the circuit, is right.
    baseline_p = sum(abs(_theoretical(marked, 0)[m]) ** 2 for m in marked)
    optimal_p = sum(abs(_theoretical(marked, 1)[m]) ** 2 for m in marked)
    assert baseline_p == pytest.approx(0.25, abs=1e-8)
    assert optimal_p == pytest.approx(1.0, abs=1e-8)


@pytest.mark.xfail(
    reason=(
        "with control(q0, q1): z(q2) is broken in guppylang 1.0.2 -- produces "
        "a wrong (non-CCZ) unitary. See CLAUDE.md gotcha #12. Kept to document "
        "the quirk (independently re-verified here, not assumed from "
        "packages/grover); mark_item uses the H-CCX-H workaround instead."
    ),
    strict=True,
)
def test_direct_multi_controlled_z_is_broken() -> None:
    from guppylang.std.builtins import control
    from guppylang.std.quantum import z

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        for i in range(N_QUBITS):
            h(qs[i])
        with control(qs[0], qs[1]):
            z(qs[2])
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    expected = UNIFORM.astype(complex).copy()
    expected[7] *= -1  # true CCZ on |+++> only flips |111>
    np.testing.assert_allclose(v, expected, atol=1e-8)
