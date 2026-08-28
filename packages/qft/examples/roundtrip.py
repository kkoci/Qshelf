"""Worked example: QFT followed by inverse-QFT returns the original state.

Run with:  python examples/roundtrip.py   (from the packages/qft directory,
with the qft package installed -- see ../README.md)

Prepares each of the 4 computational basis states on a 2-qubit register,
applies `qft` then `iqft`, and prints the resulting statevector next to the
(trivially) expected result: the original basis state, exactly.

Note: this example deliberately calls `qft` and `iqft` together in the same
program. See CLAUDE.md ("iqft in isolation") for why -- calling `iqft` with no
`qft` anywhere in the same compiled program is a known-bad pattern in
guppylang 1.0.2 for this comptime-generic pair.
"""

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit, x

from qft import iqft, qft

N = 2


@guppy
def main() -> None:
    for k in range(2**N):
        qs = array(qubit() for _ in range(N))
        if (k >> 1) & 1:
            x(qs[0])
        if k & 1:
            x(qs[1])
        qft(qs)
        iqft(qs)
        state_output("out", qs)
        discard_array(qs)


def run() -> None:
    main.check()
    shots = (
        main.with_minimal_opt()
        .emulator(n_qubits=N)
        .statevector_sim()
        .with_seed(0)
        .with_shots(1)
        .run()
    )
    (states,) = shots.partial_states()
    print("iqft(qft(|k>)) for each basis state |k> on 2 qubits:")
    for k, (_tag, partial) in enumerate(states):
        actual = partial.as_single_state()
        expected = np.zeros(2**N, dtype=complex)
        expected[k] = 1.0
        print(f"  k={k}: actual={np.round(actual, 4)}  expected={expected.real.astype(int)}")


if __name__ == "__main__":
    run()
