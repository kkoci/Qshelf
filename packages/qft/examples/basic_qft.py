"""Worked example: QFT on a 3-qubit register, expected vs actual.

Run with:  python examples/basic_qft.py   (from the packages/qft directory,
with the qft package installed -- see ../README.md)

Prepares the computational basis state |5> = |101> on 3 qubits, applies `qft`,
and prints the resulting statevector next to the value predicted by the
textbook QFT formula (equivalently, `numpy.fft.ifft` with `norm="ortho"`).
"""

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit, x

from qft import qft

N = 3
K = 5  # |101>


@guppy
def main() -> None:
    qs = array(qubit() for _ in range(N))
    # qs[0] is the most significant qubit; prepare the basis state |K>.
    if (K >> 2) & 1:
        x(qs[0])
    if (K >> 1) & 1:
        x(qs[1])
    if K & 1:
        x(qs[2])
    qft(qs)
    state_output("qft_out", qs)
    discard_array(qs)


def run() -> None:
    main.check()
    shots = (
        main.with_minimal_opt()
        .emulator(n_qubits=N)
        .statevector_sim()
        .with_seed(0)
        .run()
    )
    (states,) = shots.partial_state_dicts()
    actual = states["qft_out"].as_single_state()

    expected = np.zeros(2**N, dtype=complex)
    expected[K] = 1.0
    expected = np.fft.ifft(expected, norm="ortho")

    print(f"QFT|{K}> on {N} qubits (qs[0] most significant):")
    print("  actual:  ", np.round(actual, 4))
    print("  expected:", np.round(expected, 4))
    print("  max abs diff:", np.max(np.abs(actual - expected)))


if __name__ == "__main__":
    run()
