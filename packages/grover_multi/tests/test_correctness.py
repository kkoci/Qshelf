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
  4. `test_direct_multi_controlled_z_matches_ccz`: the same regression test
     `packages/grover` keeps (this package is independent, doesn't import
     from `packages/grover`, so it's re-verified here rather than assumed).
  5. `test_*_on_superposition_inputs`: `mark_item`, both multi-item oracles,
     and `diffuser` on random product-state superposition inputs, as full
     complex statevectors up to one global phase, against their exact
     matrices. (1)'s input |s> is invariant under the identity and every
     basis permutation, so (1) alone passes a no-op diffuser or an oracle
     with a stray permutation inside the marked/unmarked set;
     mutation-checked that (5) catches both.

Global phase: like `packages/grover`, every gate here is real-valued (H, X,
CX -- no rz/rx/ry), so a +-1 sign flip is all the phase alignment needed
(see CLAUDE.md gotcha #18 for why this differs for `packages/qaoa`).
"""

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.optimizer import OptimizationLevel
from guppylang.std.angles import angle
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit, ry, rz

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


def test_direct_multi_controlled_z_matches_ccz() -> None:
    """Regression test for CLAUDE.md gotcha #12 -- fixed as of tket 0.15.7 /
    guppylang 1.0.3, but only at OptimizationLevel.Classical or above; at
    OptimizationLevel.Minimal (what mark_item's H-CCX-H workaround exists to
    avoid needing) the underlying bug is still present, since the fix lives
    in tket's Normalize pass, which Minimal never runs. See CLAUDE.md gotcha
    #12 for the full story."""
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
    shots = (
        main.with_opt_level(OptimizationLevel.Classical)
        .emulator(n_qubits=N_QUBITS)
        .statevector_sim()
        .with_seed(0)
        .run()
    )
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    expected = UNIFORM.astype(complex).copy()
    expected[7] *= -1  # true CCZ on |+++> only flips |111>
    np.testing.assert_allclose(v, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Superposition-input tests: full statevector, up to one global phase.
#
# Every test above starts from |s> = H^3|000>, which is invariant under the
# identity and under any basis-state permutation. So on its own it can't tell
# a correct diffuser from a no-op, or a correct oracle from one with a stray
# permutation within the marked (or unmarked) set. Here each op is applied to
# random product states (per-qubit ry then rz, seeded), input and output
# statevectors are both read back, and we check out == e^{i alpha} U_ref @ in
# against the exact matrix -- the same method as packages/qft's tests.
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2]
STATES_PER_SEED = 4
DIFFUSER_REF = 2 * np.outer(UNIFORM, UNIFORM) - np.eye(N_ITEMS)


def _oracle_ref(marked: tuple[int, ...]) -> np.ndarray:
    ref = np.eye(N_ITEMS, dtype=complex)
    for m in marked:
        ref[m, m] = -1
    return ref


def assert_equal_up_to_global_phase(
    actual: np.ndarray, expected: np.ndarray, atol: float = 1e-8
) -> None:
    """Assert actual == e^{i alpha} * expected for one global phase alpha.

    Full complex statevector comparison (not probabilities), so any relative
    phase or amplitude error between basis components fails. Copied from
    packages/qft's tests (packages don't depend on each other).
    """
    overlap = np.vdot(expected, actual)
    assert abs(overlap) > 1e-12, "states are orthogonal"
    np.testing.assert_allclose(actual, expected * (overlap / abs(overlap)), atol=atol)


def _superposition_pairs(
    apply: GuppyFunctionDefinition, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run `apply` (a @guppy fn on array[qubit, N_QUBITS]) on random product
    states; return (input_state, output_state) pairs, both read back from
    Selene so no numpy model of guppy's ry/rz conventions is needed."""
    rng = np.random.default_rng(seed)
    n = N_QUBITS
    m = STATES_PER_SEED
    # Angles in half-turns (guppy `angle` units), uniform over a full turn.
    ys = [float(v) for v in rng.uniform(0.0, 2.0, m * n)]
    zs = [float(v) for v in rng.uniform(0.0, 2.0, m * n)]

    @guppy
    def main() -> None:
        ya = array(angle(v) for v in ys)
        za = array(angle(v) for v in zs)
        for st in range(m):
            qs = array(qubit() for _ in range(n))
            for i in range(n):
                ry(qs[i], ya[st * n + i])
                rz(qs[i], za[st * n + i])
            state_output("in", qs)
            apply(qs)
            state_output("out", qs)
            discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=n).statevector_sim().with_seed(0).run()
    (states,) = shots.partial_states()
    vecs = [partial.as_single_state() for _tag, partial in states]
    return [(vecs[2 * k], vecs[2 * k + 1]) for k in range(m)]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("item", range(N_ITEMS))
def test_mark_item_on_superposition_inputs(item: int, seed: int) -> None:
    @guppy
    def apply(qs: array[qubit, N_QUBITS]) -> None:
        mark_item[item](qs)

    ref = _oracle_ref((item,))
    for state_in, state_out in _superposition_pairs(apply, seed):
        assert_equal_up_to_global_phase(state_out, ref @ state_in)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("marked", TWO_ITEM_PATTERNS)
def test_oracle_2items_on_superposition_inputs(marked: tuple[int, int], seed: int) -> None:
    item1, item2 = marked

    @guppy
    def apply(qs: array[qubit, N_QUBITS]) -> None:
        oracle_2items[item1, item2](qs)

    ref = _oracle_ref(marked)
    for state_in, state_out in _superposition_pairs(apply, seed):
        assert_equal_up_to_global_phase(state_out, ref @ state_in)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("marked", THREE_ITEM_PATTERNS)
def test_oracle_3items_on_superposition_inputs(marked: tuple[int, int, int], seed: int) -> None:
    item1, item2, item3 = marked

    @guppy
    def apply(qs: array[qubit, N_QUBITS]) -> None:
        oracle_3items[item1, item2, item3](qs)

    ref = _oracle_ref(marked)
    for state_in, state_out in _superposition_pairs(apply, seed):
        assert_equal_up_to_global_phase(state_out, ref @ state_in)


@pytest.mark.parametrize("seed", SEEDS)
def test_diffuser_on_superposition_inputs(seed: int) -> None:
    @guppy
    def apply(qs: array[qubit, N_QUBITS]) -> None:
        diffuser(qs)

    for state_in, state_out in _superposition_pairs(apply, seed):
        assert_equal_up_to_global_phase(state_out, DIFFUSER_REF @ state_in)

