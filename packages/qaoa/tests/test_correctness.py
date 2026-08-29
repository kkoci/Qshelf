"""Correctness tests for the qaoa package, run on Selene's Quest (statevector) backend.

Methodology
-----------
QAOA's cost layer applies `rz` with a continuous, generic angle (`gamma`) --
non-Clifford for almost all values (only exact multiples of pi/2 are
Clifford), so Selene's Stim (stabilizer) backend cannot simulate it in
general. Confirmed empirically, not just assumed from the QFT/Grover
precedent: running this package's circuit on `.stabilizer_sim()` raises
`EmulatorError`, with Selene's own error message spelling out why --
"RZ(...) is not representable in stabiliser form. Theta must be an
(approximate) multiple of pi/2 for Clifford operations." -- see
`test_stabilizer_sim_rejects_generic_angles`. All other tests use
`statevector_sim()` (Quest).

Global phase: unlike `packages/qft` and `packages/grover` (whose circuits
are built from real-valued gates only, so any global-phase ambiguity is a
simple +-1 sign), QAOA's `rz`/`rx` gates are genuinely complex. Comparing
against a hand-built reference therefore needs a *general* complex phase
alignment (rotate the reference by `vdot(reference, actual)`'s unit phase),
not just a sign flip -- see `_phase_align`. This was discovered the hard
way: an initial version of `test_full_circuit_matches_exact_reference` used
a sign-only alignment (copied from `packages/grover`'s `_phase_aligned`) and
failed with large, confusing diffs on the mixer layer alone, even though the
circuit was completely correct -- see CLAUDE.md for the full story.

Structure
---------
  1. `cost_hamiltonian_layer`/`mixer_layer` verified against hand-derived
     phases for a small (single-edge, 2-qubit) case.
  2. The full `qaoa_circuit` verified against an exact reference built from
     `scipy.linalg.expm` of the true cost/mixer Hamiltonians, for several
     (gammas, betas) and p = 1, 2, 3 layers -- run sequentially in one
     process (see qaoa.py's module docstring for why that's safe here,
     unlike `packages/qft`'s `iqft`).
  3. `run_qaoa` (real shot sampling, not the exact statevector) and
     `optimize_qaoa` (the actual classical-quantum loop) exercised end to
     end, checking that the measured cut distribution both beats the random
     baseline and improves as p increases with reasonable parameters --
     this is the actual evidence QAOA is doing something, per the package's
     design brief.
"""

import math
import time

import numpy as np
import pytest
import scipy.linalg as la
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit

from qaoa import (
    EDGES,
    N_NODES,
    build_cost_hamiltonian_layer,
    build_mixer_layer,
    cut_value,
    optimize_qaoa,
    qaoa_circuit,
    run_qaoa,
)

PI = math.pi


def _phase_align(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate `reference` by a general complex unit phase to match `actual`'s
    global phase convention -- needed for QAOA's complex (rz/rx) gates,
    unlike qft/grover's real-valued-circuit +-1 sign flip. See module
    docstring."""
    phase = np.vdot(reference, actual)
    return reference * (phase / abs(phase))


def _op_on(op: np.ndarray, i: int, n: int) -> np.ndarray:
    mats = [np.eye(2)] * n
    mats[i] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _exact_qaoa_state(gammas: list[float], betas: list[float], n: int, edges) -> np.ndarray:
    z = np.array([[1, 0], [0, -1]])
    x = np.array([[0, 1], [1, 0]])
    h_cost = np.zeros((2**n, 2**n))
    for i, j in edges:
        h_cost += (np.eye(2**n) - _op_on(z, i, n) @ _op_on(z, j, n)) / 2
    h_mix = sum(_op_on(x, i, n) for i in range(n))
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = la.expm(-1j * g * h_cost) @ psi
        psi = la.expm(-1j * b * h_mix) @ psi
    return psi


def _run_with_retry(emulator):
    """Retries a transient Windows Application Control policy block on the
    Selene subprocess spawn (see CLAUDE.md gotcha #3). Only retries `OSError`
    -- any other exception (e.g. Stim's real Clifford-rejection error in
    `test_stabilizer_sim_rejects_generic_angles`) propagates immediately."""
    last_error: OSError | None = None
    for attempt in range(10):
        try:
            return emulator.run()
        except OSError as exc:
            last_error = exc
            time.sleep(min(1.0 * (attempt + 1), 6.0))
    raise AssertionError(f"emulator run failed after retries: {last_error}")


def _run_state(build, n_qubits: int = N_NODES) -> np.ndarray:
    build.check()
    emulator = build.with_minimal_opt().emulator(n_qubits=n_qubits).statevector_sim().with_seed(0)
    shots = _run_with_retry(emulator)
    return shots.partial_state_dicts()[0]["out"].as_single_state()


def test_cost_layer_matches_hand_derived_phases_single_edge() -> None:
    """Hand-derived case: a single edge (0,1) on 2 qubits. exp(-i*gamma*(I-Z0*Z1)/2)
    applied to |++> is diag(1, e^{-i*gamma}, e^{-i*gamma}, 1) * |++> -- |00> and
    |11> (same side, edge not "cut" in the eigenvalue sense) keep phase 1;
    |01> and |10> pick up phase e^{-i*gamma}."""
    gamma = 0.7
    layer = build_cost_hamiltonian_layer(2, [(0, 1)])

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(2))
        h(qs[0])
        h(qs[1])
        layer(qs, angle(gamma / PI))
        state_output("out", qs)
        discard_array(qs)

    v = _run_state(main, n_qubits=2)
    expected = 0.5 * np.array([1, np.exp(-1j * gamma), np.exp(-1j * gamma), 1])
    expected = _phase_align(v, expected)
    np.testing.assert_allclose(v, expected, atol=1e-8)


def test_mixer_layer_matches_hand_derived_phases_single_qubit() -> None:
    """Hand-derived case: exp(-i*beta*X)|0> = cos(beta)|0> - i*sin(beta)|1>."""
    beta = 0.4
    layer = build_mixer_layer(1)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(1))
        layer(qs, angle(beta / PI))
        state_output("out", qs)
        discard_array(qs)

    v = _run_state(main, n_qubits=1)
    expected = np.array([np.cos(beta), -1j * np.sin(beta)])
    np.testing.assert_allclose(v, expected, atol=1e-8)


@pytest.mark.parametrize(
    "gammas,betas",
    [
        ([0.3], [0.2]),
        ([0.5], [0.4]),
        ([1.1], [0.9]),
        ([0.3, 0.6], [0.2, 0.5]),
        ([0.1, 0.4, 0.7], [0.9, 0.3, 0.5]),
    ],
)
def test_full_circuit_matches_exact_reference(gammas: list[float], betas: list[float]) -> None:
    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(N_NODES))
        gamma_angles = array(angle(g / PI) for g in gammas)
        beta_angles = array(angle(b / PI) for b in betas)
        qaoa_circuit(qs, gamma_angles, beta_angles)
        state_output("out", qs)
        discard_array(qs)

    v = _run_state(main)
    expected = _phase_align(v, _exact_qaoa_state(gammas, betas, N_NODES, EDGES))
    np.testing.assert_allclose(v, expected, atol=1e-8)


def test_stabilizer_sim_rejects_generic_angles() -> None:
    """Confirms Stim genuinely cannot run this circuit (not just "we didn't
    try") -- see module docstring."""
    from guppylang.std.builtins import output
    from guppylang.std.quantum import collect_measurements, measure_array

    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(N_NODES))
        gamma_angles = array(angle(g / PI) for g in [0.5])
        beta_angles = array(angle(b / PI) for b in [0.4])
        qaoa_circuit(qs, gamma_angles, beta_angles)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=N_NODES).stabilizer_sim().with_seed(0).with_shots(1)
    with pytest.raises(Exception, match="stabiliser|Clifford"):
        _run_with_retry(emulator)


def test_run_qaoa_beats_random_baseline() -> None:
    """A single, reasonable (gamma, beta) should give an average measured cut
    noticeably above the random baseline (2.5 out of 5 edges for C5)."""
    avg_cut, bitstrings = run_qaoa([0.5], [0.4], shots=300)
    assert len(bitstrings) == 300
    assert all(len(b) == N_NODES for b in bitstrings)
    assert avg_cut > 3.0, f"avg_cut={avg_cut} not clearly better than random baseline 2.5"


def test_p0_baseline_matches_random_search() -> None:
    """p=0 (no QAOA layers, just the uniform superposition) should measure
    close to the random-guessing baseline avg cut, 2.5."""
    avg_cut, _ = run_qaoa([], [], shots=500)
    assert avg_cut == pytest.approx(2.5, abs=0.3)


def test_cut_distribution_improves_with_p() -> None:
    """The actual evidence QAOA is doing something: the average measured cut
    value should increase from the p=0 baseline through p=1 to p=2 layers,
    approaching the true MaxCut value of 4, given reasonable parameters.

    Parameters below are real `optimize_qaoa` output from development (see
    `examples/basic_qaoa.py`'s recorded output in its own run, and
    CLAUDE.md), not hand-picked -- but hardcoded here (rather than calling
    `optimize_qaoa` live) to keep this specific test's subprocess-spawn count
    low: it's the "does the trend hold" check, so it doesn't need to
    re-discover the parameters every run. `test_live_optimization_loop_beats_baseline`
    below exercises the *live* optimizer loop instead, kept small there for
    the same reason -- this dev machine's transient subprocess-spawn issue
    (CLAUDE.md gotcha #3) compounds with call volume despite `run_qaoa`'s
    retries, and this test's whole point doesn't require re-optimizing."""
    baseline, _ = run_qaoa([], [], shots=300)
    p1_cut, _ = run_qaoa([0.42], [0.3], shots=300)
    p2_cut, _ = run_qaoa([0.4, 0.42], [0.3, 0.3], shots=300)

    assert p1_cut > baseline, f"p=1 ({p1_cut}) did not beat baseline ({baseline})"
    assert p2_cut > p1_cut, f"p=2 ({p2_cut}) did not beat p=1 ({p1_cut})"


def test_live_optimization_loop_beats_baseline() -> None:
    """Exercises the actual hybrid classical-quantum loop end to end (not
    hand-picked constants): a short, live `optimize_qaoa` run should find
    parameters that measurably beat the random baseline. Kept small (few
    shots, few optimizer iterations) -- see note on the test above."""
    baseline, _ = run_qaoa([], [], shots=150)
    _, _, found_cut = optimize_qaoa(p=1, shots=80, maxiter=6)
    assert found_cut > baseline, f"optimize_qaoa found {found_cut}, did not beat baseline {baseline}"
