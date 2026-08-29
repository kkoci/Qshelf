"""Worked example: VQE for the H2 ground-state energy.

Run with:  python examples/basic_vqe.py   (from the packages/vqe_h2
directory, with the vqe_h2 package installed -- see ../README.md)

Runs the actual classical-quantum loop (`optimize_vqe`, real circuit
compiles + real shot sampling, not a shortcut) starting from the package's
validated default starting point, and prints the found energy next to the
Hartree-Fock baseline and the literature ground-state energy.

Expect this to take a couple of minutes -- each optimizer step is 5 real
quantum circuit compile+run+measure cycles (see vqe_h2.py's module
docstring for why 5, not 15).
"""

from vqe_h2 import LITERATURE_GROUND_STATE_ENERGY, estimate_energy, optimize_vqe


def run() -> None:
    hf_energy = estimate_energy([0.0, 0.0, 0.0, 0.0], shots=500)
    print("H2 VQE (4-qubit, Jordan-Wigner, STO-3G -- see vqe_h2.py for citation)")
    print(f"  Hartree-Fock reference energy : {hf_energy:.4f} Hartree")

    params, found_energy = optimize_vqe(shots=200, maxiter=15)
    print(f"  VQE found energy              : {found_energy:.4f} Hartree")
    print(f"  literature ground-state energy: {LITERATURE_GROUND_STATE_ENERGY:.4f} Hartree")
    print(f"  optimized parameters (radians): {[round(p, 3) for p in params]}")
    print(f"  error vs literature           : {abs(found_energy - LITERATURE_GROUND_STATE_ENERGY):.4f} Hartree")


if __name__ == "__main__":
    run()
