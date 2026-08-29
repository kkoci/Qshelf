"""Standalone runner for a single QFT/IQFT correctness case.

Run in its own Python process (see test_correctness.py): guppylang 1.0.2 has a
confirmed bug where `iqft` compiled and run *alone* (no call to `qft` anywhere
in the same program) produces a different -- wrong -- unitary than the exact
same `iqft` compiled and run alongside a `qft` call on the same register
(which is exact), and this can non-deterministically corrupt other, unrelated
compilations sharing the same process. See CLAUDE.md gotcha #5 for the full
writeup; test_iqft_in_isolation_matches_numpy_fft in test_correctness.py is
kept (marked xfail) specifically to document this.

Prints the resulting 2**n x 2**n matrix as JSON: {"real": [[...]], "imag": [[...]]}
"""

import argparse
import json
import sys

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit, x

from qft import iqft, qft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--op",
        choices=["qft", "iqft", "roundtrip", "roundtrip_reverse"],
        required=True,
    )
    args = parser.parse_args()
    n = args.n

    if args.op == "qft":

        @guppy
        def driver() -> None:
            for k in range(2**n):
                qs = array(qubit() for _ in range(n))
                for i in range(n):
                    if (k >> (n - 1 - i)) & 1 == 1:
                        x(qs[i])
                qft(qs)
                state_output("out", qs)
                discard_array(qs)

    elif args.op == "iqft":

        @guppy
        def driver() -> None:
            for k in range(2**n):
                qs = array(qubit() for _ in range(n))
                for i in range(n):
                    if (k >> (n - 1 - i)) & 1 == 1:
                        x(qs[i])
                iqft(qs)
                state_output("out", qs)
                discard_array(qs)

    elif args.op == "roundtrip":

        @guppy
        def driver() -> None:
            for k in range(2**n):
                qs = array(qubit() for _ in range(n))
                for i in range(n):
                    if (k >> (n - 1 - i)) & 1 == 1:
                        x(qs[i])
                qft(qs)
                iqft(qs)
                state_output("out", qs)
                discard_array(qs)

    else:  # roundtrip_reverse

        @guppy
        def driver() -> None:
            for k in range(2**n):
                qs = array(qubit() for _ in range(n))
                for i in range(n):
                    if (k >> (n - 1 - i)) & 1 == 1:
                        x(qs[i])
                iqft(qs)
                qft(qs)
                state_output("out", qs)
                discard_array(qs)

    driver.check()
    shots = (
        driver.with_minimal_opt()
        .emulator(n_qubits=n)
        .statevector_sim()
        .with_seed(0)
        .with_shots(1)
        .run()
    )
    (states,) = shots.partial_states()
    matrix = np.zeros((2**n, 2**n), dtype=complex)
    for k, (_tag, partial) in enumerate(states):
        matrix[:, k] = partial.as_single_state()

    json.dump({"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}, sys.stdout)


if __name__ == "__main__":
    main()
