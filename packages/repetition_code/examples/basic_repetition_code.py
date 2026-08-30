"""Worked example: the 3-qubit bit-flip repetition code correcting a single
X error.

Run with:  python examples/basic_repetition_code.py   (from the
packages/repetition_code directory, with the package installed -- see
../README.md)

For each of the 4 possible single-error scenarios (no error, error on
qubit 0/1/2), encodes a logical |1>, injects the error, extracts the
syndrome, applies the correction, and prints the syndrome and the final
(corrected) physical qubit values next to the expected logical value.

Runs on Selene's Stim (stabilizer) backend -- this circuit is entirely
Clifford (H, X, CX, Z-measurement only), the first package in this registry
where that's the appropriate choice; see repetition_code's README/CLAUDE.md.
"""

from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import collect_measurements, measure_array, qubit, x

from repetition_code import correct, encode, extract_syndrome

ERROR_LABELS = {-1: "no error", 0: "error on q0", 1: "error on q1", 2: "error on q2"}


def run_case(error_qubit: int) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        x(qs[0])  # logical |1>
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
        correct(qs[0], qs[1], qs[2], s0, s1)
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=5).stabilizer_sim().with_seed(0).with_shots(1).run()
    values = dict(shots.results[0].entries)
    bits = "".join(str(b) for b in values["bits"])
    print(
        f"  {ERROR_LABELS[error_qubit]:<12} syndrome=({int(values['s0'])},{int(values['s1'])})"
        f"  corrected bits={bits}  (expected 111)"
    )


def run() -> None:
    print("3-qubit bit-flip repetition code, logical |1> encoded, single-error correction:")
    for error_qubit in (-1, 0, 1, 2):
        run_case(error_qubit)


if __name__ == "__main__":
    run()
