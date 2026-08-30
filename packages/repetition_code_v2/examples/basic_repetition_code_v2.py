"""Worked example: the generalized bit-flip repetition code at distance 5
and 7, and a multi-round run with a fresh error injected between rounds.

Run with:  python examples/basic_repetition_code_v2.py   (from the
packages/repetition_code_v2 directory, with the package installed -- see
../README.md)

Runs on Selene's Stim (stabilizer) backend -- this circuit family is
entirely Clifford at every distance (H, X, CX, Z-measurement only), same as
packages/repetition_code.
"""

from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import collect_measurements, measure_array, qubit, x

from repetition_code_v2 import correct, encode, extract_syndrome


def run_single_round(n: int, errors: list[int]) -> None:
    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n))
        x(qs[0])  # logical |1>
        encode(qs)
        for eq in errors:
            x(qs[eq])
        syndrome = extract_syndrome(qs)
        correct(qs, syndrome)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=2 * n).stabilizer_sim().with_seed(0).with_shots(1).run()
    bits = shots.register_bitstrings()["bits"][0]
    print(f"  distance-{n}, errors on {errors or 'none'}: corrected bits={bits}  (expected {'1' * n})")


def run_multiround(n: int, rounds_errors: list[list[int]]) -> None:
    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n))
        x(qs[0])  # logical |1>
        encode(qs)
        for errs in rounds_errors:
            for eq in errs:
                x(qs[eq])
            syndrome = extract_syndrome(qs)
            correct(qs, syndrome)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=2 * n).stabilizer_sim().with_seed(0).with_shots(1).run()
    bits = shots.register_bitstrings()["bits"][0]
    print(f"  distance-{n}, {len(rounds_errors)} rounds, one fresh error each: corrected bits={bits}  (expected {'1' * n})")


def run() -> None:
    print("Distance-5 (corrects up to 2 simultaneous errors):")
    run_single_round(5, [])
    run_single_round(5, [2])
    run_single_round(5, [1, 3])  # 2 simultaneous errors -- within budget

    print("\nDistance-7 (corrects up to 3 simultaneous errors):")
    run_single_round(7, [0, 3, 6])  # 3 simultaneous errors -- within budget

    print("\nMulti-round: a fresh error injected between each round (distance-5):")
    run_multiround(5, [[0], [2], [4]])


if __name__ == "__main__":
    run()
