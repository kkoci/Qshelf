"""Worked example: QAOA for MaxCut on a 5-node cycle graph.

Run with:  python examples/basic_qaoa.py   (from the packages/qaoa directory,
with the qaoa package installed -- see ../README.md)

Runs the actual classical-quantum loop: for p = 0, 1, 2 layers, uses
`optimize_qaoa` (a live `scipy.optimize.minimize` loop over `run_qaoa`, each
evaluation compiling and running real shots on Selene's Quest emulator) to
search for good (gamma, beta) parameters, then prints the resulting average
cut value and the most-sampled partition, next to the true MaxCut value.

Expect this to take a minute or so -- each optimizer step is a real quantum
circuit compile + emulate.
"""

from collections import Counter

from qaoa import EDGES, N_NODES, cut_value, optimize_qaoa, run_qaoa

TRUE_MAX_CUT = 4  # for the default 5-cycle graph (odd cycle, not bipartite)


def run() -> None:
    print(f"MaxCut on a {N_NODES}-node cycle, edges {EDGES} (true MaxCut = {TRUE_MAX_CUT}):")

    baseline_cut, _ = run_qaoa([], [], shots=500)
    print(f"\np=0 (no QAOA layers, uniform superposition baseline): avg cut = {baseline_cut:.3f}")

    for p in (1, 2):
        gammas, betas, avg_cut = optimize_qaoa(p=p, shots=200, maxiter=25)
        _, bitstrings = run_qaoa(gammas, betas, shots=500)
        best_bits, count = Counter(bitstrings).most_common(1)[0]
        print(
            f"\np={p}: optimized gamma={[round(g, 3) for g in gammas]}, "
            f"beta={[round(b, 3) for b in betas]}"
        )
        print(f"  avg cut = {avg_cut:.3f}  (baseline {baseline_cut:.3f}, true max {TRUE_MAX_CUT})")
        print(f"  most-sampled partition: {best_bits} ({count}/500 shots), cut = {cut_value(best_bits)}")


if __name__ == "__main__":
    run()
