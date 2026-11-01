"""Runner for a single QFT/IQFT/cphase correctness case.

Imported by test_correctness.py (`run_case(...)`, all cases in the pytest
process). Also runnable as a script for ad-hoc inspection of one case, printing
JSON to stdout.

Two input modes:
  * ``basis``: run every computational basis state |k> through the op and return
    the 2**n x 2**n matrix whose k-th column is the output state. NOTE: Selene
    canonicalises every reported state so its first non-zero amplitude is real
    and positive, i.e. each column carries its own arbitrary global phase. This
    mode is therefore blind to errors that only multiply each basis input by a
    different phase -- which is exactly what using `crz` in place of a
    controlled-phase gate does to `qft`. Don't rely on it alone.
  * ``random``: prepare random product-state superpositions (per-qubit `ry` then
    `rz` with seeded random angles), report the input state, apply the op,
    report the output state. Relative phases between basis components are then
    visible within a single reported state, so comparing ``out`` against
    ``U_ref @ in`` up to one global phase catches relative-phase errors.

JSON output: basis -> {"real": [[...]], "imag": [[...]]};
random -> {"pairs": [{"in": [re, im], "out": [re, im]}, ...]}.
"""

import argparse
import json
import sys

import numpy as np
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit, ry, rz, x

from qft import cphase, iqft, qft

OPS = ["qft", "iqft", "roundtrip", "roundtrip_reverse", "cphase"]


def _apply_op(op: str, theta_halfturns: float):
    """Return a @guppy function applying `op` to a register (closed-over choice).

    Separate @guppy definitions per op because guppy can't dispatch through a
    Python-level callable inside a guppy body (CLAUDE.md gotcha #6).
    """
    if op == "qft":

        @guppy
        def apply[n: nat](qs: array[qubit, n]) -> None:
            qft(qs)

    elif op == "iqft":

        @guppy
        def apply[n: nat](qs: array[qubit, n]) -> None:
            iqft(qs)

    elif op == "roundtrip":

        @guppy
        def apply[n: nat](qs: array[qubit, n]) -> None:
            qft(qs)
            iqft(qs)

    elif op == "roundtrip_reverse":

        @guppy
        def apply[n: nat](qs: array[qubit, n]) -> None:
            iqft(qs)
            qft(qs)

    elif op == "cphase":
        th = theta_halfturns

        @guppy
        def apply[n: nat](qs: array[qubit, n]) -> None:
            cphase(qs[0], qs[1], angle(th))

    else:
        raise ValueError(op)
    return apply


def _run(driver, n: int):
    driver.check()
    last: Exception | None = None
    # Retry: Windows Application Control occasionally blocks the Selene runner
    # transiently on first access (CLAUDE.md gotcha #3).
    for _attempt in range(5):
        try:
            shots = (
                driver.with_minimal_opt()
                .emulator(n_qubits=n)
                .statevector_sim()
                .with_seed(0)
                .with_shots(1)
                .run()
            )
            break
        except Exception as exc:  # pragma: no cover
            if "Application Control" not in str(exc):
                raise
            last = exc
    else:
        raise RuntimeError(f"emulator blocked after retries: {last}")
    (states,) = shots.partial_states()
    return [partial.as_single_state() for _tag, partial in states]


def run_case(
    n: int, op: str, mode: str = "basis", seed: int = 0, num_states: int = 4,
    theta_halfturns: float = 0.5,
) -> dict:
    apply = _apply_op(op, theta_halfturns)

    if mode == "basis":

        @guppy
        def driver() -> None:
            for k in range(2**n):
                qs = array(qubit() for _ in range(n))
                for i in range(n):
                    if (k >> (n - 1 - i)) & 1 == 1:
                        x(qs[i])
                apply(qs)
                state_output("out", qs)
                discard_array(qs)

        cols = _run(driver, n)
        matrix = np.array(cols).T
        return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}

    if mode == "random":
        rng = np.random.default_rng(seed)
        # Angles in half-turns (guppy `angle` units), uniform over a full turn.
        ys = [float(v) for v in rng.uniform(0.0, 2.0, num_states * n)]
        zs = [float(v) for v in rng.uniform(0.0, 2.0, num_states * n)]
        m = num_states

        @guppy
        def driver() -> None:
            ya = array(angle(v) for v in ys)
            za = array(angle(v) for v in zs)
            for s in range(m):
                qs = array(qubit() for _ in range(n))
                for i in range(n):
                    ry(qs[i], ya[s * n + i])
                    rz(qs[i], za[s * n + i])
                state_output("in", qs)
                apply(qs)
                state_output("out", qs)
                discard_array(qs)

        vecs = _run(driver, n)
        pairs = [
            {"in": [vecs[2 * s].real.tolist(), vecs[2 * s].imag.tolist()],
             "out": [vecs[2 * s + 1].real.tolist(), vecs[2 * s + 1].imag.tolist()]}
            for s in range(m)
        ]
        return {"pairs": pairs}

    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--op", choices=OPS, required=True)
    parser.add_argument("--mode", choices=["basis", "random"], default="basis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-states", type=int, default=4)
    parser.add_argument("--theta", type=float, default=0.5, help="cphase angle, half-turns")
    args = parser.parse_args()
    data = run_case(args.n, args.op, args.mode, args.seed, args.num_states, args.theta)
    json.dump(data, sys.stdout)


if __name__ == "__main__":
    main()
