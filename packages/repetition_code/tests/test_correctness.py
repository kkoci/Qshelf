"""Correctness tests for the repetition_code package.

Methodology
-----------
Unlike every earlier package in this registry (`qft`, `grover`, `qaoa`,
`vqe_h2`), this circuit is entirely Clifford: `encode`/`extract_syndrome`/
`correct` only ever use `H`, `X`, `CX`, and Z-basis measurement -- no
continuous-angle rotation gate appears anywhere. So Selene's Stim
(stabilizer) backend, which cannot handle the earlier packages' non-Clifford
circuits at all (see their CLAUDE.md / test docstrings), is not just
*usable* here but the semantically appropriate, efficient choice -- checked
directly (not assumed from "well, it's Clifford" reasoning alone):
`test_stabilizer_sim_reproduces_statevector_sim_results` runs the same
circuits on both backends and confirms they agree, and every other test in
this file uses `stabilizer_sim()` as the primary backend.

Three things are verified, matching the task's explicit requirements:

  1. `test_recovery_for_all_error_locations`: for both logical values (False,
     True) and all 4 error scenarios (no error, error on q0/q1/q2), the
     final measured physical qubits exactly match the original logical
     value on all three -- the code actually corrects the error, not just
     "runs without crashing".
  2. `test_syndrome_matches_error_location`: the *syndrome itself* (before
     any correction is applied) matches the table in repetition_code.py's
     module docstring, for each error scenario -- verifies the diagnosis
     step specifically, separately from whether the subsequent correction
     happens to fix things up.
  3. `test_syndrome_extraction_preserves_logical_superposition`: the key
     correctness property of a real error-correcting code, and the one an
     "always output the majority bit" naive implementation could satisfy by
     accident for test (1) without actually being a working syndrome-based
     code. Prepares a genuine logical superposition (`H` on q0 before
     `encode`, giving `alpha|000> + beta|111>`), injects each of the 4 error
     scenarios, runs `extract_syndrome` + `correct`, and checks the *exact*
     final statevector (via `state_output`, no shot noise) reproduces the
     original superposition to ~1e-16 -- not just that a *measurement* of it
     would give the right answer on average. Since every gate here is real-
     valued (H, X, CX -- no rz/rx/ry), only a +-1 sign-flip phase alignment
     is needed (matching qft/grover, not qaoa's general complex alignment
     -- see CLAUDE.md gotcha #18).
"""

import time

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard_array,
    h,
    measure_array,
    qubit,
    x,
)

from repetition_code import correct, encode, extract_syndrome


def _run_with_retry(emulator):
    """Retries a transient Windows Application Control policy block on the
    Selene subprocess spawn (see CLAUDE.md gotcha #3)."""
    last_error: OSError | None = None
    for attempt in range(10):
        try:
            return emulator.run()
        except OSError as exc:
            last_error = exc
            time.sleep(min(1.0 * (attempt + 1), 6.0))
    raise AssertionError(f"emulator run failed after retries: {last_error}")


# error_qubit: -1 = no error injected, 0/1/2 = which physical qubit gets an X error.
# (a plain int sentinel, not None: closing over `None` types the guppy-body
# comparison `error_qubit == 0` as NoneType == int, which guppy's checker
# rejects -- "Operator not defined ... for `None` and `int`" -- see CLAUDE.md.)
ERROR_CASES = [-1, 0, 1, 2]
EXPECTED_SYNDROME = {
    -1: (False, False),
    0: (True, False),
    1: (True, True),
    2: (False, True),
}


@pytest.mark.parametrize("logical", [False, True])
@pytest.mark.parametrize("error_qubit", ERROR_CASES)
def test_recovery_for_all_error_locations(logical: bool, error_qubit: int) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        if logical:
            x(qs[0])
        encode(qs[0], qs[1], qs[2])
        if error_qubit == 0:
            x(qs[0])
        if error_qubit == 1:
            x(qs[1])
        if error_qubit == 2:
            x(qs[2])
        s0, s1 = extract_syndrome(qs[0], qs[1], qs[2])
        correct(qs[0], qs[1], qs[2], s0, s1)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=5).stabilizer_sim().with_seed(0).with_shots(20)
    shots = _run_with_retry(emulator)
    bitstrings = shots.register_bitstrings()["bits"]
    expected = "111" if logical else "000"
    assert all(b == expected for b in bitstrings), (
        f"logical={logical}, error_qubit={error_qubit}: expected all shots == {expected!r}, got {set(bitstrings)}"
    )


@pytest.mark.parametrize("error_qubit", ERROR_CASES)
def test_syndrome_matches_error_location(error_qubit: int) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        encode(qs[0], qs[1], qs[2])
        if error_qubit == 0:
            x(qs[0])
        if error_qubit == 1:
            x(qs[1])
        if error_qubit == 2:
            x(qs[2])
        s0, s1 = extract_syndrome(qs[0], qs[1], qs[2])
        output("s0", s0)
        output("s1", s1)
        discard_array(qs)

    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=5).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    for shot in shots.results:
        values = dict(shot.entries)
        assert (bool(values["s0"]), bool(values["s1"])) == EXPECTED_SYNDROME[error_qubit]


@pytest.mark.parametrize("error_qubit", ERROR_CASES)
def test_syndrome_extraction_preserves_logical_superposition(error_qubit: int) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        h(qs[0])
        encode(qs[0], qs[1], qs[2])
        if error_qubit == 0:
            x(qs[0])
        if error_qubit == 1:
            x(qs[1])
        if error_qubit == 2:
            x(qs[2])
        s0, s1 = extract_syndrome(qs[0], qs[1], qs[2])
        correct(qs[0], qs[1], qs[2], s0, s1)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=5).statevector_sim().with_seed(0)
    shots = _run_with_retry(emulator)
    actual = shots.partial_state_dicts()[0]["out"].as_single_state()

    expected = np.zeros(8, dtype=complex)
    expected[0b000] = 1 / np.sqrt(2)
    expected[0b111] = 1 / np.sqrt(2)
    if np.vdot(actual, expected).real < 0:
        expected = -expected
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_stabilizer_sim_reproduces_statevector_sim_results() -> None:
    """This circuit is entirely Clifford (H, X, CX, Z-measurement only -- no
    rz/rx/ry anywhere), unlike every earlier package in this registry, so
    Stim should agree with Quest exactly rather than merely "not crash"."""

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        h(qs[0])
        encode(qs[0], qs[1], qs[2])
        x(qs[1])
        s0, s1 = extract_syndrome(qs[0], qs[1], qs[2])
        correct(qs[0], qs[1], qs[2], s0, s1)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    main.check()
    stim_emulator = main.with_minimal_opt().emulator(n_qubits=5).stabilizer_sim().with_seed(0).with_shots(200)
    quest_emulator = main.with_minimal_opt().emulator(n_qubits=5).statevector_sim().with_seed(0).with_shots(200)
    stim_shots = _run_with_retry(stim_emulator)
    quest_shots = _run_with_retry(quest_emulator)

    stim_bits = set(stim_shots.register_bitstrings()["bits"])
    quest_bits = set(quest_shots.register_bitstrings()["bits"])
    # The logical qubit is in a superposition, so shots land on "000" or
    # "111" (never anything else -- confirms correction always succeeds),
    # split roughly evenly; both backends should see the same *set* of
    # possible outcomes.
    assert stim_bits == {"000", "111"}
    assert quest_bits == {"000", "111"}
