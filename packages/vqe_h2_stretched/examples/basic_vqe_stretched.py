"""Worked example: VQE for H2's ground-state energy at a stretched
(non-equilibrium, 2.1A) bond length.

Run with:  python examples/basic_vqe_stretched.py   (from the
packages/vqe_h2_stretched directory, with the package installed -- see
../README.md)

Runs the actual classical-quantum loop (`optimize_vqe`, real circuit
compiles + real shot sampling, not a shortcut) starting from the package's
validated default starting point, and prints the found energy next to the
params=0 baseline and the exact ground-state energy. Expect this to take
longer than `packages/vqe_h2`'s equivalent example -- this landscape's
validated default needs more optimizer iterations to converge (maxiter=60
here vs 20 there; see vqe_h2_stretched.py's module docstring).
"""

from vqe_h2_stretched import EXACT_GROUND_STATE_ENERGY, estimate_energy, optimize_vqe


def run() -> None:
    baseline_energy = estimate_energy([0.0, 0.0, 0.0, 0.0], shots=500)
    print("Stretched H2 VQE (4-qubit, Jordan-Wigner, STO-3G, bond length 2.1A -- see vqe_h2_stretched.py for citation)")
    print(f"  params=0 baseline energy      : {baseline_energy:.4f} Hartree")

    params, found_energy = optimize_vqe(shots=200, maxiter=60)
    print(f"  VQE found energy              : {found_energy:.4f} Hartree")
    print(f"  exact ground-state energy     : {EXACT_GROUND_STATE_ENERGY:.4f} Hartree")
    print(f"  optimized parameters (radians): {[round(p, 3) for p in params]}")
    print(f"  error vs exact                : {abs(found_energy - EXACT_GROUND_STATE_ENERGY):.4f} Hartree")


if __name__ == "__main__":
    run()
