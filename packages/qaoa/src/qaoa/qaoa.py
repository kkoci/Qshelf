"""QAOA (Quantum Approximate Optimization Algorithm) for MaxCut, in guppylang.

Default graph
-------------
A 5-node cycle C5: edges (0,1), (1,2), (2,3), (3,4), (4,0). C5 is an odd
cycle, so it is *not* bipartite -- no partition cuts all 5 edges, and the
true MaxCut value is 4 (e.g. {0,2} vs {1,3,4}). This makes it a genuinely
nontrivial instance (unlike an even cycle, which has a perfect, trivially-
findable solution) and a real test of whether QAOA's amplitude amplification
is doing something.

Units: gamma/beta are in RADIANS, the standard QAOA convention
(U_C(gamma) = exp(-i*gamma*H_C), U_B(beta) = exp(-i*beta*H_B)). guppylang's
`angle` type stores *half-turns* (`float(some_angle) == halfturns * pi`), so
converting a radian value `g` to the `angle` guppy expects is `angle(g / pi)`
-- see `cost_hamiltonian_layer`/`mixer_layer`/`run_qaoa` below, and CLAUDE.md
for how this was empirically pinned down (against an exact matrix-exponential
reference, not just derived by hand).

Cost Hamiltonian: H_C = sum_{(i,j) in edges} (I - Z_i Z_j) / 2 (the standard
MaxCut cost operator -- eigenvalue 1 on a computational basis state where
edge (i,j) is cut, 0 otherwise, so <psi|H_C|psi> is the expected cut value).
Mixer Hamiltonian: H_B = sum_i X_i (the standard transverse-field mixer).

Per-edge cost gate: `cx(i,j); rz(j, -gamma); cx(i,j)` implements
exp(-i*gamma*(I - Z_i Z_j)/2) *exactly* (confirmed to ~1e-16 against
`scipy.linalg.expm` of the true operator, phase-aligned -- see CLAUDE.md and
tests/test_correctness.py for why phase alignment is necessary here and
wasn't for packages/qft or packages/grover). Per-qubit mixer gate:
`rx(i, 2*beta)` implements exp(-i*beta*X_i) exactly, same verification.

Why `@guppy.comptime` (and why it's safe here -- see CLAUDE.md before reusing
this pattern blindly)
----------------------------------------------------------------------------
`cost_hamiltonian_layer` needs to iterate over `graph`, a plain Python list
of edges -- not a `nat`-generic bound or an array length, so a real (non-
comptime) `@guppy` function's `for` loop can't do it (guppy's `for` iterates
guppy iterables like `range(n)` or a guppy `array`, not an arbitrary Python
object). `@guppy.comptime` traces via real Python execution, so
`for i, j in graph: ...` inside a comptime body works and unrolls one
CX-RZ-CX block per edge, the same way `packages/qft`'s `qft[n]` unrolls one
H/CRZ block per qubit.

This means `qaoa_circuit` ends up calling comptime `cost_hamiltonian_layer`
and comptime `mixer_layer`, which are themselves called from a comptime
`main()` in `run_qaoa` below -- three levels of comptime-calling-comptime.
We checked this specifically against CLAUDE.md's gotcha #5 (the confirmed
guppylang 1.0.2 bug where a comptime function derived from another comptime
function via `dagger`/hand-reversal misbehaves when used without its
"parent" also being called on the same register) before committing to this
design: nothing here is derived from anything else via `dagger`, there is no
"parent/derived" relationship between any of these functions, and we ran the
whole chain -- `cost_hamiltonian_layer`, `mixer_layer`, and `qaoa_circuit`
for p=1, 2, and 3 layers, several different gamma/beta values, all in one
process -- against an exact `scipy.linalg.expm` reference and got ~1e-16
agreement every time (see `tests/test_correctness.py` and CLAUDE.md). No
subprocess isolation is used in this package's tests, matching
`packages/grover`'s finding (not `packages/qft`'s) that gotcha #5's
precondition (a comptime function used *only* via composition with another
comptime function derived from it) genuinely doesn't apply here.

New (to this package) finding: explicit generic-argument syntax at a call
site (`func[T](...)`, which works fine in a *plain* `@guppy` function -- see
CLAUDE.md gotcha #15 from packages/grover) raises
`GuppyComptimeError: Explicitly specifying type arguments of generic
functions in a comptime context is not supported yet` when attempted inside
a `@guppy.comptime` function. Not needed here regardless -- `qaoa_circuit`'s
`p` is always inferred from the length of the `gammas`/`betas` array
arguments -- but worth knowing this restriction exists. See CLAUDE.md.
"""

import math
import time

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat, output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    h,
    measure_array,
    qubit,
    rx,
    rz,
)

EdgeList = list[tuple[int, int]]

N_NODES = 5
EDGES: EdgeList = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # 5-cycle (C5)

_PI = math.pi


def build_cost_hamiltonian_layer(n_nodes: int, graph: EdgeList):
    """Build a guppy function implementing the QAOA cost layer
    exp(-i*gamma*sum_{(i,j) in graph} (I - Z_i Z_j)/2) for the given graph.

    A separate factory call per distinct graph, not a runtime `graph`
    argument: guppy compiles a fixed circuit structure, so the edge list
    must be known when the guppy function is *defined* (traced via
    `@guppy.comptime`), not passed in at the call site. This is the closest
    honest equivalent, under guppy's static-compilation model, to a
    `cost_hamiltonian_layer(qs, graph, gamma)` signature -- `graph` is a
    Python-level parameter of the *factory*, and the returned function is
    called as `cost_hamiltonian_layer(qs, gamma)`.
    """

    @guppy.comptime
    def cost_hamiltonian_layer(qs: array[qubit, n_nodes], gamma: angle) -> None:
        for i, j in graph:
            cx(qs[i], qs[j])
            rz(qs[j], -gamma)
            cx(qs[i], qs[j])

    return cost_hamiltonian_layer


def build_mixer_layer(n_nodes: int):
    """Build a guppy function implementing the QAOA mixer layer
    exp(-i*beta*sum_i X_i) on an `n_nodes`-qubit register."""

    @guppy.comptime
    def mixer_layer(qs: array[qubit, n_nodes], beta: angle) -> None:
        for i in range(n_nodes):
            rx(qs[i], 2.0 * beta)

    return mixer_layer


def build_qaoa_circuit(n_nodes: int, graph: EdgeList):
    """Build the full p-layer QAOA circuit for the given graph: uniform
    superposition, then p rounds of cost layer + mixer layer.

    `p` (the number of layers) *is* a real guppy `nat` generic (unlike
    `graph`) -- it's inferred from the length of the `gammas`/`betas` array
    arguments at each call site, so one `qaoa_circuit` works for any p.
    """
    cost_layer = build_cost_hamiltonian_layer(n_nodes, graph)
    mix_layer = build_mixer_layer(n_nodes)

    @guppy.comptime
    def qaoa_circuit[p: nat](
        qs: array[qubit, n_nodes],
        gammas: array[angle, p],
        betas: array[angle, p],
    ) -> None:
        for i in range(n_nodes):
            h(qs[i])
        for layer in range(p):
            cost_layer(qs, gammas[layer])
            mix_layer(qs, betas[layer])

    return qaoa_circuit


# Default instances, for the default 5-cycle graph.
cost_hamiltonian_layer = build_cost_hamiltonian_layer(N_NODES, EDGES)
mixer_layer = build_mixer_layer(N_NODES)
qaoa_circuit = build_qaoa_circuit(N_NODES, EDGES)


def cut_value(bitstring: str, graph: EdgeList = EDGES) -> int:
    """Classical MaxCut objective (plain Python, not guppy): the number of
    edges with endpoints on opposite sides of the partition `bitstring`
    encodes (bitstring[i] is node i's side, '0' or '1')."""
    return sum(1 for i, j in graph if bitstring[i] != bitstring[j])


def run_qaoa(
    gammas: list[float],
    betas: list[float],
    n_nodes: int = N_NODES,
    graph: EdgeList = EDGES,
    shots: int = 200,
    seed: int = 0,
) -> tuple[float, list[str]]:
    """Compile and run the QAOA circuit for the given gamma/beta parameters
    (radians), and return (average measured cut value, measured bitstrings).

    This is the quantum half of the classical-quantum loop: builds a fresh
    entrypoint closing over the current parameters (the same
    recompile-per-iteration pattern `packages/qft`/`packages/grover` use for
    generating per-case drivers), runs real shots, and reads back genuine
    measurement statistics via `output()` + `register_bitstrings()` -- not a
    shortcut through the exact statevector.
    """
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length (p)")
    circuit = qaoa_circuit if (n_nodes, graph) == (N_NODES, EDGES) else build_qaoa_circuit(n_nodes, graph)

    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(n_nodes))
        if gammas:
            # `array(... for x in [])` can't infer an element type from zero
            # elements ("Cannot infer the type of empty list"), so p=0 (no
            # QAOA layers -- just the uniform superposition baseline) is
            # handled directly rather than through `circuit`.
            gamma_angles = array(angle(g / _PI) for g in gammas)
            beta_angles = array(angle(b / _PI) for b in betas)
            circuit(qs, gamma_angles, beta_angles)
        else:
            for i in range(n_nodes):
                h(qs[i])
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=n_nodes).statevector_sim().with_seed(seed).with_shots(shots)
    # Retries a transient Windows Application Control policy block on the
    # Selene subprocess spawn (see CLAUDE.md gotcha #3) -- real on this dev
    # machine, and `optimize_qaoa` below calls `run_qaoa` many times per
    # optimization run, so it's worth handling here rather than in every
    # caller.
    result = None
    last_error: OSError | None = None
    for attempt in range(10):
        try:
            result = emulator.run()
            break
        except OSError as exc:
            last_error = exc
            time.sleep(min(1.0 * (attempt + 1), 6.0))
    if result is None:
        raise RuntimeError(f"emulator run failed after retries: {last_error}") from last_error
    bitstrings = result.register_bitstrings()["bits"]
    avg_cut = sum(cut_value(b, graph) for b in bitstrings) / len(bitstrings)
    return avg_cut, bitstrings


def optimize_qaoa(
    p: int,
    n_nodes: int = N_NODES,
    graph: EdgeList = EDGES,
    shots: int = 150,
    seed: int = 0,
    maxiter: int = 20,
) -> tuple[list[float], list[float], float]:
    """Classical optimization loop (plain Python, outside guppy): search for
    the (gammas, betas) that maximize the average measured cut value, using
    `scipy.optimize.minimize` (Nelder-Mead, gradient-free -- suited to a
    noisy, sampled objective). Each objective evaluation is a full
    `run_qaoa` call: compile, run real shots, average the measured cut
    value. This *is* the hybrid classical-quantum loop, not a shortcut.

    Returns (best_gammas, best_betas, best_avg_cut).
    """
    from scipy.optimize import minimize

    def objective(x: list[float]) -> float:
        gammas = list(x[:p])
        betas = list(x[p:])
        avg_cut, _ = run_qaoa(gammas, betas, n_nodes=n_nodes, graph=graph, shots=shots, seed=seed)
        return -avg_cut

    x0 = [0.4] * p + [0.3] * p
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 0.05, "fatol": 0.05},
    )
    gammas = list(result.x[:p])
    betas = list(result.x[p:])
    avg_cut, _ = run_qaoa(gammas, betas, n_nodes=n_nodes, graph=graph, shots=shots, seed=seed)
    return gammas, betas, avg_cut
