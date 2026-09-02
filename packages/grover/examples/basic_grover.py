"""Worked example: Grover's search amplifying a marked item on 3 qubits.

Run with:  python examples/basic_grover.py   (from the packages/grover
directory, with the grover package installed -- see ../README.md)

Searches an 8-item space (3 qubits) for the marked item 5 (|101>), running
the optimal number of Grover iterations (2, computed by
`grover.optimal_iterations`), and prints the resulting probability
distribution next to what plain amplitude amplification predicts:
P(marked) rising from the classical baseline 1/8 = 0.125 to ~0.945.
"""

import numpy as np
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit

from grover import grover_search, optimal_iterations

MARKED = 5  # |101>
N = 8
ITERATIONS = optimal_iterations(N, 1)


@guppy
def main() -> None:
    qs = array(qubit() for _ in range(3))
    grover_search[MARKED, ITERATIONS](qs)
    state_output("out", qs)
    discard_array(qs)


def run() -> None:
    main.check()
    shots = (
        # Classical, not Minimal: oracle/diffuser use a direct
        # `with control(...): z(...)` (CLAUDE.md gotcha #12, fixed as of
        # tket 0.15.7 / guppylang 1.0.3) -- the fix only takes effect under
        # Classical/Default opt level, not Minimal. See grover.py and
        # tests/test_correctness.py for the full story.
        main.with_opt_level(OptimizationLevel.Classical)
        .emulator(n_qubits=3)
        .statevector_sim()
        .with_seed(0)
        .run()
    )
    (states,) = shots.partial_state_dicts()
    v = states["out"].as_single_state()
    probs = np.abs(v) ** 2

    theta = np.arcsin(np.sqrt(1 / N))
    expected_marked_p = np.sin((2 * ITERATIONS + 1) * theta) ** 2

    print(f"Searching {N} items for marked item {MARKED} ({ITERATIONS} Grover iterations):")
    print("  baseline (no search) P(marked)  :", 1 / N)
    for k in range(N):
        marker = " <- marked" if k == MARKED else ""
        print(f"  P(|{k}>) = {probs[k]:.4f}{marker}")
    print("  actual   P(marked):", round(probs[MARKED], 4))
    print("  expected P(marked):", round(expected_marked_p, 4))


if __name__ == "__main__":
    run()
