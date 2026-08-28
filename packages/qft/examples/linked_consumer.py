"""Consume the qft3.hugr package built by build_lib.py, WITHOUT importing this
package's Python source -- only a `@guppy.declare` stub with a matching
`@link_name`, and `.emulator(libs=[...])`.

This mirrors how a real consumer of a guppy-registry package would use it:
`guppylang.defs.emulator()`'s `libs: list[Package]` argument links a
separately-compiled HUGR Package into the consumer's own compiled program.

Run build_lib.py first (from the packages/qft directory), then this script.
"""

import numpy as np
from guppylang import guppy
from guppylang.library import link_name
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit, x
from hugr.package import Package

N = 3
K = 5  # |101>


@guppy.declare
@link_name("qft_registry.qft3")
def qft3(qs: array[qubit, N]) -> None: ...


@guppy
def main() -> None:
    qs = array(qubit() for _ in range(N))
    if (K >> 2) & 1:
        x(qs[0])
    if (K >> 1) & 1:
        x(qs[1])
    if K & 1:
        x(qs[2])
    qft3(qs)
    state_output("out", qs)
    discard_array(qs)


def run() -> None:
    main.check()
    with open("qft3.hugr", "rb") as f:
        lib = Package.from_bytes(f.read())

    shots = (
        main.with_minimal_opt()
        .emulator(n_qubits=N, libs=[lib])
        .statevector_sim()
        .with_seed(0)
        .run()
    )
    (states,) = shots.partial_state_dicts()
    actual = states["out"].as_single_state()

    expected = np.zeros(2**N, dtype=complex)
    expected[K] = 1.0
    expected = np.fft.ifft(expected, norm="ortho")

    print("Result of calling qft3 via a linked HUGR Package (no Python import of qft):")
    print("  actual:  ", np.round(actual, 4))
    print("  expected:", np.round(expected, 4))
    print("  max abs diff:", np.max(np.abs(actual - expected)))


if __name__ == "__main__":
    run()
