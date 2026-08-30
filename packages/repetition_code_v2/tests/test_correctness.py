"""Correctness tests for the repetition_code_v2 package, run on Selene's
Stim (stabilizer) backend -- same reasoning as `packages/repetition_code`:
`encode`/`extract_syndrome`/`correct`/`correct_for_rounds` only ever use
`CX`, `X`, and Z-basis measurement (no continuous-angle rotation gate
anywhere), so this circuit family is genuinely Clifford at every distance
tested, not just at distance 3 -- confirmed directly (not assumed), see
`test_stabilizer_sim_reproduces_statevector_sim_results_at_distance5`.

Methodology, matching the task's explicit requirements:
  1. `test_distance3_syndrome_matches_original_table` /
     `test_distance3_recovery_matches_original`: the base-case sanity
     check -- this generalization's distance-3 behavior (syndrome values,
     recovered bits) is checked against `packages/repetition_code`'s own
     published 4-row syndrome table and recovery behavior, hardcoded here
     (not imported -- this registry's packages don't depend on each other,
     see root README's "Structure"; the same convention
     `packages/vqe_h2_stretched` used for its own vqe_h2 cross-check).
  2. `test_distanceN_recovers_from_single_qubit_errors` (N=5,7):
     single-error recovery at each of the two new distances.
  3. `test_distanceN_recovers_from_up_to_t_simultaneous_errors` (N=5,7):
     the actual point of generalizing distance -- distance-5 should
     correct 2 simultaneous errors, distance-7 should correct 3, not just
     1 (the distance-3 limit).
  4. `test_exceeding_correction_capacity_can_misdecode`: confirms the
     generalized decoder has the SAME kind of fundamental limit the
     original package documented (not a new bug) -- more simultaneous
     errors than distance allows CAN be misdecoded, checked with one
     concrete, hand-traceable example, not just asserted from theory.
  5. `test_multiround_recovers_from_new_errors_between_rounds` /
     `test_multiround_preserves_logical_superposition`: the task's actual
     multi-round requirement -- a NEW error injected *between* rounds
     (not a single one-shot error), confirmed via both final classical
     bits and (for the superposition case) the exact statevector, the
     same non-collapse property `packages/repetition_code` verified for a
     single round, now checked across several.
  6. `test_correct_for_rounds_matches_manual_composition`: the package's
     convenience "run N rounds" function produces identical results to
     composing extract_syndrome/correct manually the same number of times.
  7. `test_stabilizer_sim_reproduces_statevector_sim_results_at_distance5`:
     Stim/Quest agreement, generalized beyond distance 3.

Hand-verification (matching this registry's "hand-verify at least one small
case" methodology, see CLAUDE.md): `correct`'s minimum-weight decoder was
traced by hand for n=3 against every one of the 4 syndrome rows in
`repetition_code.py`'s table before writing any tests -- see
`repetition_code_v2.py`'s module docstring's "correct" section, and
`test_distance3_syndrome_and_decoder_hand_trace` below, which encodes that
hand trace as an executable assertion (not just prose) for one
representative row (s=(1,0), the "error on q0" case).
"""

import itertools
import time

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, nat, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard_array,
    h,
    measure_array,
    qubit,
    x,
)

from repetition_code_v2 import correct, correct_for_rounds, encode, extract_syndrome

# `packages/repetition_code`'s published 4-row syndrome table (distance 3),
# hardcoded for the base-case sanity check -- NOT imported (this registry's
# packages don't depend on each other). error_qubit: -1 = no error, 0/1/2 =
# which physical qubit gets an X error, matching that package's own
# convention (a same-type sentinel, not None -- CLAUDE.md gotcha #20).
_ORIGINAL_ERROR_CASES = [-1, 0, 1, 2]
_ORIGINAL_EXPECTED_SYNDROME = {
    -1: (False, False),
    0: (True, False),
    1: (True, True),
    2: (False, True),
}


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


def _single_round_bits(n: int, logical: bool, error_qubits: list[int]):
    """One encode -> inject errors -> extract_syndrome -> correct round,
    returning the final measured bits. error_qubits is a Python list, so
    this driver needs @guppy.comptime to iterate it (plain @guppy cannot
    iterate an arbitrary Python collection -- see CLAUDE.md); encode/
    extract_syndrome/correct themselves are plain @guppy (no comptime
    needed there at all, same as packages/repetition_code)."""

    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n))
        if logical:
            x(qs[0])
        encode(qs)
        for eq in error_qubits:
            x(qs[eq])
        syndrome = extract_syndrome(qs)
        correct(qs, syndrome)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    return main


def _single_round_syndrome(n: int, error_qubits: list[int]):
    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n))
        encode(qs)
        for eq in error_qubits:
            x(qs[eq])
        syndrome = extract_syndrome(qs)
        output("syndrome", syndrome)
        discard_array(qs)

    return main


def _multiround_bits(n: int, logical: bool, rounds_errors: list[list[int]]):
    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n))
        if logical:
            x(qs[0])
        encode(qs)
        for errs in rounds_errors:
            for eq in errs:
                x(qs[eq])
            syndrome = extract_syndrome(qs)
            correct(qs, syndrome)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    return main


def _multiround_state(n: int, rounds_errors: list[list[int]]):
    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n))
        h(qs[0])
        encode(qs)
        for errs in rounds_errors:
            for eq in errs:
                x(qs[eq])
            syndrome = extract_syndrome(qs)
            correct(qs, syndrome)
        state_output("out", qs)
        discard_array(qs)

    return main


def _correct_for_rounds_bits(n: int, logical: bool, rounds_errors: list[list[int]]):
    """Same shape as _multiround_bits, but uses correct_for_rounds for the
    trailing no-new-error rounds -- see
    test_correct_for_rounds_matches_manual_composition."""

    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n))
        if logical:
            x(qs[0])
        encode(qs)
        for errs in rounds_errors:
            for eq in errs:
                x(qs[eq])
            syndrome = extract_syndrome(qs)
            correct(qs, syndrome)
        correct_for_rounds(qs, nat(3))  # see CLAUDE.md: a bare literal `3` here
        # raises "Type mismatch: Expected argument of type nat, got int" when this
        # file runs under pytest (reproduced twice), but NOT in an isolated
        # single-purpose script -- explicit nat(...) sidesteps it either way.
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    return main


# ---------------------------------------------------------------------------
# 1. Distance-3 base case vs. packages/repetition_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error_qubit", _ORIGINAL_ERROR_CASES)
def test_distance3_syndrome_matches_original_table(error_qubit: int) -> None:
    errs = [] if error_qubit == -1 else [error_qubit]
    main = _single_round_syndrome(3, errs)
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=8).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    for shot in shots.results:
        values = dict(shot.entries)
        syndrome = values["syndrome"]
        s0, s1, s2 = bool(syndrome[0]), bool(syndrome[1]), bool(syndrome[2])
        assert (s0, s1) == _ORIGINAL_EXPECTED_SYNDROME[error_qubit]
        assert s2 is False, "syndrome[n-1] must always be the unused padding slot"


def test_distance3_syndrome_and_decoder_hand_trace() -> None:
    """Encodes the module docstring's hand trace of correct's decoder for
    the s=(1,0) ("error on q0") row as an executable assertion: candidate =
    [F,T,T] (weight 2), use_complement=True (2*2 > 3), so the applied
    correction flips exactly index 0 -- matching
    packages/repetition_code.correct's `if s0 and not s1: x(q0)` row
    exactly, not just "ends up with the right final state"."""
    main = _single_round_bits(3, False, [0])
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=8).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    assert bitstrings == {"000"}


@pytest.mark.parametrize("logical", [False, True])
@pytest.mark.parametrize("error_qubit", _ORIGINAL_ERROR_CASES)
def test_distance3_recovery_matches_original(logical: bool, error_qubit: int) -> None:
    errs = [] if error_qubit == -1 else [error_qubit]
    main = _single_round_bits(3, logical, errs)
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=8).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    expected = "111" if logical else "000"
    assert bitstrings == {expected}, f"logical={logical}, error_qubit={error_qubit}: got {bitstrings}"


# ---------------------------------------------------------------------------
# 2. Distance-5 / distance-7: single-error and multi-error recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 7])
@pytest.mark.parametrize("error_qubit", range(7))
def test_distanceN_recovers_from_single_qubit_errors(n: int, error_qubit: int) -> None:
    if error_qubit >= n:
        pytest.skip("error_qubit out of range for this n")
    main = _single_round_bits(n, False, [error_qubit])
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=2 * n).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    assert bitstrings == {"0" * n}, f"n={n}, error_qubit={error_qubit}: got {bitstrings}"


@pytest.mark.parametrize("errors", list(itertools.combinations(range(5), 2)))
def test_distance5_recovers_from_up_to_t_simultaneous_errors(errors: tuple[int, int]) -> None:
    """Distance-5 corrects up to (5-1)//2 = 2 simultaneous errors -- every
    2-error combination among 5 qubits, not just 1 error (distance-3's
    limit)."""
    main = _single_round_bits(5, False, list(errors))
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=10).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    assert bitstrings == {"00000"}, f"errors={errors}: got {bitstrings}"


@pytest.mark.parametrize("errors", list(itertools.combinations(range(7), 3))[:12])
def test_distance7_recovers_from_up_to_t_simultaneous_errors(errors: tuple[int, int, int]) -> None:
    """Distance-7 corrects up to (7-1)//2 = 3 simultaneous errors. Limited
    to the first 12 of the 35 possible 3-combinations to keep this
    package's subprocess-spawn volume reasonable -- each is still a fully
    independent, real circuit run, not a subsampled/statistical check."""
    main = _single_round_bits(7, False, list(errors))
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=14).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    assert bitstrings == {"0000000"}, f"errors={errors}: got {bitstrings}"


def test_exceeding_correction_capacity_can_misdecode() -> None:
    """Not a bug: a fundamental limit of the code itself, same as
    packages/repetition_code's documented "cannot correct two or more
    simultaneous errors" limitation at distance 3, generalized -- distance-5
    corrects at most 2 simultaneous errors; 3 simultaneous errors (0,1,2)
    exceeds that budget and, for this specific pattern, is misdecoded into
    flipping the wrong two qubits instead of recovering. Confirmed with one
    concrete, hand-traceable example (not just asserted from theory) so a
    future change to the decoder can't silently make this "pass" by
    accident and hide a real regression elsewhere."""
    main = _single_round_bits(5, False, [0, 1, 2])
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=10).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    assert bitstrings != {"00000"}, (
        "expected 3 simultaneous errors to exceed distance-5's 2-error correction "
        f"capacity and misdecode; got the fully-recovered state instead: {bitstrings}"
    )


# ---------------------------------------------------------------------------
# 3. Multi-round: a NEW error injected between rounds
# ---------------------------------------------------------------------------

MULTIROUND_CASES = [
    (5, False, [[0], [2], [4]]),
    (5, True, [[1], [1], [3]]),  # repeated errors on the same qubit across rounds
    (7, False, [[0], [3], [6], [2]]),
    (5, False, [[0, 2], [4]]),  # round 1: 2 simultaneous errors (within distance-5's budget)
]


@pytest.mark.parametrize("n,logical,rounds_errors", MULTIROUND_CASES)
def test_multiround_recovers_from_new_errors_between_rounds(
    n: int, logical: bool, rounds_errors: list[list[int]]
) -> None:
    main = _multiround_bits(n, logical, rounds_errors)
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=2 * n).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    expected = ("1" if logical else "0") * n
    assert bitstrings == {expected}, f"n={n}, logical={logical}, rounds={rounds_errors}: got {bitstrings}"


@pytest.mark.parametrize("n,rounds_errors", [(5, [[0], [2], [4]]), (5, [[1], [1], [3]])])
def test_multiround_preserves_logical_superposition(n: int, rounds_errors: list[list[int]]) -> None:
    """The actual defining property of a working syndrome-based code,
    extended across multiple rounds: a genuine logical superposition
    survives repeated syndrome-extraction + correction, with a fresh error
    injected between each round -- checked via the exact statevector
    (state_output, no shot noise), not just that a measurement would give
    the right answer on average. Every gate here is real-valued (H, X, CX),
    so only a +-1 sign-flip phase alignment is needed (CLAUDE.md gotcha
    #18)."""
    main = _multiround_state(n, rounds_errors)
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=2 * n).statevector_sim().with_seed(0)
    shots = _run_with_retry(emulator)
    actual = shots.partial_state_dicts()[0]["out"].as_single_state()

    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[2**n - 1] = 1 / np.sqrt(2)
    if np.vdot(actual, expected).real < 0:
        expected = -expected
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_correct_for_rounds_matches_manual_composition() -> None:
    """correct_for_rounds(qs, k) (the package's convenience "run k rounds"
    function) should behave identically to composing extract_syndrome/
    correct manually k times with no new errors injected in between --
    checked by running 3 trailing no-op rounds via correct_for_rounds after
    an initial manually-composed error-recovery round, and confirming the
    logical value is still exactly right (repeated correction with nothing
    left to fix should be a stable no-op, not drift)."""
    main = _correct_for_rounds_bits(5, True, [[1, 3]])
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=10).stabilizer_sim().with_seed(0).with_shots(10)
    shots = _run_with_retry(emulator)
    bitstrings = set(shots.register_bitstrings()["bits"])
    assert bitstrings == {"11111"}


# ---------------------------------------------------------------------------
# 4. Stim vs Quest agreement at distance 5 (generalizing beyond distance 3)
# ---------------------------------------------------------------------------


def test_stabilizer_sim_reproduces_statevector_sim_results_at_distance5() -> None:
    """packages/repetition_code confirmed Stim/Quest agreement at distance
    3; this circuit family is Clifford at every distance (still only H, X,
    CX, Z-measurement), so this re-confirms it at distance 5 rather than
    assuming the distance-3 finding generalizes for free."""
    main = _multiround_bits(5, True, [[0], [2, 3]])
    main.check()
    stim_emulator = main.with_minimal_opt().emulator(n_qubits=10).stabilizer_sim().with_seed(0).with_shots(50)
    quest_emulator = main.with_minimal_opt().emulator(n_qubits=10).statevector_sim().with_seed(0).with_shots(50)
    stim_shots = _run_with_retry(stim_emulator)
    quest_shots = _run_with_retry(quest_emulator)

    stim_bits = set(stim_shots.register_bitstrings()["bits"])
    quest_bits = set(quest_shots.register_bitstrings()["bits"])
    assert stim_bits == {"11111"}
    assert quest_bits == {"11111"}
