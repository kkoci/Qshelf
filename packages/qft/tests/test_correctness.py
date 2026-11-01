"""Correctness tests for the qft package, run on Selene's Quest (statevector) backend.

Methodology
-----------
QFT is non-Clifford in general (controlled-phase rotations by pi/2**k for k >= 2
are outside the Clifford group), so Selene's Stim backend can't simulate it --
these tests use `statevector_sim()` (Quest).

Two complementary kinds of check, each against an exact reference
(`numpy.fft.ifft(eye, norm="ortho")` for `qft`, `numpy.fft.fft(...)` for `iqft`,
identity for round trips, diag(1, 1, 1, e^{i theta}) for `cphase`):

1. Basis-column matrices: run every basis state |k> through the op and compare
   the assembled 2**n x 2**n matrix to the reference. Caveat: Selene's state
   readout canonicalises each reported state (first non-zero amplitude made real
   and positive), so every column carries its own global phase. These tests are
   therefore BLIND to errors that multiply different basis inputs by different
   phases.

2. Superposition inputs (the strong check): random product states, read out
   before and after the op, compared as full statevectors up to a single global
   phase: out == e^{i alpha} * U_ref @ in. Because the input is a superposition,
   any relative-phase error is visible inside one reported state.

Why (2) exists: `qft` originally used `crz` where a controlled-phase gate was
needed. CRz(theta) = diag(1, 1, e^{-i theta/2}, e^{i theta/2}) differs from
CP(theta) = diag(1, 1, 1, e^{i theta}) by an rz on the control. In `qft` each
control is still a basis state when the gate fires, so the error was only a
per-input phase -- invisible to (1). It showed up only in `iqft` (whose
controls are in superposition when the gate fires), and was misdiagnosed as a
guppylang bug (guppylang#2250, closed; root cause was crz vs cphase in our
code).

Each case runs in a fresh subprocess via `_case_runner.py`; transient Windows
Application Control blocks are retried there (CLAUDE.md gotcha #3).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CASE_RUNNER = Path(__file__).parent / "_case_runner.py"
SIZES = [1, 2, 3, 4]
SEEDS = [0, 1, 2]


def _run_case(n: int, op: str, *extra: str) -> dict:
    """Run one case in a fresh subprocess and return its parsed JSON output."""
    last_error: subprocess.CalledProcessError | None = None
    for _attempt in range(3):
        try:
            result = subprocess.run(
                [sys.executable, str(CASE_RUNNER), "--n", str(n), "--op", op, *extra],
                capture_output=True,
                text=True,
                check=True,
                cwd=CASE_RUNNER.parent,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            last_error = exc
    raise AssertionError(
        f"case runner failed after retries: {last_error.stderr if last_error else ''}"
    )


def _basis_matrix(n: int, op: str) -> np.ndarray:
    data = _run_case(n, op, "--mode", "basis")
    return np.array(data["real"]) + 1j * np.array(data["imag"])


def _random_pairs(n: int, op: str, seed: int, *extra: str) -> list[tuple[np.ndarray, np.ndarray]]:
    data = _run_case(n, op, "--mode", "random", "--seed", str(seed), *extra)
    return [
        (np.array(p["in"][0]) + 1j * np.array(p["in"][1]),
         np.array(p["out"][0]) + 1j * np.array(p["out"][1]))
        for p in data["pairs"]
    ]


def assert_equal_up_to_global_phase(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-8) -> None:
    """Assert actual == e^{i alpha} * expected for a single global phase alpha.

    Full complex statevector comparison (not probabilities), so any relative
    phase error between basis components fails.
    """
    overlap = np.vdot(expected, actual)
    assert abs(overlap) > 1e-12, "states are orthogonal"
    np.testing.assert_allclose(actual, expected * (overlap / abs(overlap)), atol=atol)


def _qft_ref(n: int) -> np.ndarray:
    return np.fft.ifft(np.eye(2**n), axis=0, norm="ortho")


def _iqft_ref(n: int) -> np.ndarray:
    return np.fft.fft(np.eye(2**n), axis=0, norm="ortho")


REFERENCES = {
    "qft": _qft_ref,
    "iqft": _iqft_ref,
    "roundtrip": lambda n: np.eye(2**n),
    "roundtrip_reverse": lambda n: np.eye(2**n),
}


# ---------------------------------------------------------------------------
# (1) basis-column matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", list(REFERENCES))
@pytest.mark.parametrize("n", SIZES)
def test_basis_matrix_matches_reference(n: int, op: str) -> None:
    matrix = _basis_matrix(n, op)
    np.testing.assert_allclose(matrix, REFERENCES[op](n), atol=1e-8)
    np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(2**n), atol=1e-8)


def test_qft_hand_verified_n2() -> None:
    """Hand-verified reference case: QFT on 2 qubits.

    QFT|k> = (1/2) * sum_{j=0}^{3} exp(2*pi*i*j*k/4) |j>, i.e. the columns are
    powers of i = exp(i*pi/2):
      |0> -> (1/2)(|0> + |1> + |2> + |3>)
      |1> -> (1/2)(|0> + i|1> - |2> - i|3>)
      |2> -> (1/2)(|0> - |1> + |2> - |3>)
      |3> -> (1/2)(|0> - i|1> - |2> + i|3>)
    """
    matrix = _basis_matrix(2, "qft")
    half = 0.5
    expected = np.array(
        [
            [half, half, half, half],
            [half, half * 1j, -half, -half * 1j],
            [half, -half, half, -half],
            [half, -half * 1j, -half, half * 1j],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-8)


def test_iqft_alone_case_a() -> None:
    """iqft alone on |01> (n=2): exactly (1/2)(|0> - i|1> - |2> + i|3>).

    With the old crz-based iqft this came out as [0.5, 0.354-0.354j, -0.5,
    -0.354+0.354j] -- unitary, right magnitudes, wrong relative phases.
    """
    actual = _basis_matrix(2, "iqft")[:, 1]
    assert_equal_up_to_global_phase(actual, np.array([0.5, -0.5j, -0.5, 0.5j]))


# ---------------------------------------------------------------------------
# (2) superposition inputs, full statevector up to one global phase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("op", list(REFERENCES))
@pytest.mark.parametrize("n", SIZES)
def test_superposition_input_matches_reference(n: int, op: str, seed: int) -> None:
    ref = REFERENCES[op](n)
    for state_in, state_out in _random_pairs(n, op, seed):
        assert_equal_up_to_global_phase(state_out, ref @ state_in)


@pytest.mark.parametrize("theta_halfturns", [0.5, 0.25, 0.125, -0.25, 1.0 / 3.0])
def test_cphase_is_controlled_phase(theta_halfturns: float) -> None:
    """cphase(theta) == diag(1, 1, 1, e^{i theta}) up to global phase, on superpositions."""
    theta = np.pi * theta_halfturns
    ref = np.diag([1, 1, 1, np.exp(1j * theta)])
    for seed in SEEDS:
        pairs = _random_pairs(2, "cphase", seed, "--theta", str(theta_halfturns))
        for state_in, state_out in pairs:
            assert_equal_up_to_global_phase(state_out, ref @ state_in)


def test_superposition_check_detects_crz_vs_cphase() -> None:
    """Sanity check on the checker itself: CRz in place of CP must be rejected.

    Pure numpy -- guards against the global-phase-aligned comparison being too
    lenient to catch the exact class of bug it was added for.
    """
    theta = np.pi / 2
    crz = np.diag([1, 1, np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
    cp = np.diag([1, 1, 1, np.exp(1j * theta)])
    state = np.ones(4) / 2
    with pytest.raises(AssertionError):
        assert_equal_up_to_global_phase(crz @ state, cp @ state)
    # ...but a basis input can't tell them apart (the trap the old tests fell into).
    basis = np.array([0, 0, 0, 1], dtype=complex)
    assert_equal_up_to_global_phase(crz @ basis, cp @ basis)
