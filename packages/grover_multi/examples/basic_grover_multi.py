"""Worked example: Grover's search amplifying 2 and 3 marked items on 3 qubits.

Run with:  python examples/basic_grover_multi.py   (from the
packages/grover_multi directory, with the package installed -- see
../README.md)

Searches the same 8-item space packages/grover uses, but for 2 and then 3
marked items at once, running the optimal number of Grover iterations for
each (via `optimal_iterations`), and prints the resulting probability
distribution next to what the closed-form multi-item amplitude-amplification
formula predicts.
"""

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit

from grover_multi import N_ITEMS, grover_2items, grover_3items, optimal_iterations


def run_2items(marked: tuple[int, int]) -> None:
    m0, m1 = marked
    iterations = optimal_iterations(N_ITEMS, 2)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        grover_2items[m0, m1, iterations](qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=3).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    probs = np.abs(v) ** 2

    theta = np.arcsin(np.sqrt(2 / N_ITEMS))
    expected_p = np.sin((2 * iterations + 1) * theta) ** 2

    print(f"\nSearching {N_ITEMS} items for marked items {marked} ({iterations} Grover iteration(s)):")
    for k in range(N_ITEMS):
        marker = " <- marked" if k in marked else ""
        print(f"  P(|{k}>) = {probs[k]:.4f}{marker}")
    print(f"  actual   P(either marked item): {probs[m0] + probs[m1]:.4f}")
    print(f"  expected P(either marked item): {expected_p:.4f}")


def run_3items(marked: tuple[int, int, int]) -> None:
    m0, m1, m2 = marked
    iterations = optimal_iterations(N_ITEMS, 3)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(3))
        grover_3items[m0, m1, m2, iterations](qs)
        state_output("out", qs)
        discard_array(qs)

    main.check()
    shots = main.with_minimal_opt().emulator(n_qubits=3).statevector_sim().with_seed(0).run()
    v = shots.partial_state_dicts()[0]["out"].as_single_state()
    probs = np.abs(v) ** 2

    theta = np.arcsin(np.sqrt(3 / N_ITEMS))
    expected_p = np.sin((2 * iterations + 1) * theta) ** 2

    print(f"\nSearching {N_ITEMS} items for marked items {marked} ({iterations} Grover iteration(s)):")
    for k in range(N_ITEMS):
        marker = " <- marked" if k in marked else ""
        print(f"  P(|{k}>) = {probs[k]:.4f}{marker}")
    actual_p = sum(probs[m] for m in marked)
    print(f"  actual   P(any marked item): {actual_p:.4f}")
    print(f"  expected P(any marked item): {expected_p:.4f}")


if __name__ == "__main__":
    run_2items((2, 5))
    run_3items((1, 3, 6))
