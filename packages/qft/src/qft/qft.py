"""Quantum Fourier Transform and inverse QFT, in-place on a qubit register.

Convention: for an n-qubit array `qs`, `qs[0]` is the most significant qubit.
`qft` implements the textbook QFT unitary

    QFT|j> = (1/sqrt(2**n)) * sum_k exp(2*pi*i*j*k / 2**n) |k>

i.e. the matrix ``numpy.fft.ifft(np.eye(2**n), norm="ortho")`` in this qubit
ordering. `iqft` is its exact inverse (adjoint).
"""

from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import cx, crz, h, qubit


@guppy(daggerable=True)
def swap(q0: qubit, q1: qubit) -> None:
    """Swap the states of two qubits using three CNOTs."""
    cx(q0, q1)
    cx(q1, q0)
    cx(q0, q1)


@guppy.comptime(daggerable=True)
def qft[n: nat](qs: array[qubit, n]) -> None:
    """In-place Quantum Fourier Transform on an n-qubit register.

    For each qubit i (from most to least significant), applies a Hadamard
    followed by controlled phase rotations from every less-significant qubit
    j > i, with angle pi / 2**(j - i). A final pass of swaps reverses qubit
    order to restore the standard output convention (qs[0] most significant).
    """
    for i in range(n):
        h(qs[i])
        for j in range(i + 1, n):
            crz(qs[j], qs[i], pi / 2.0 ** (j - i))
    for k in range(n // 2):
        swap(qs[k], qs[n - k - 1])


@guppy.comptime(daggerable=True)
def iqft[n: nat](qs: array[qubit, n]) -> None:
    """In-place inverse Quantum Fourier Transform on an n-qubit register.

    Exact adjoint of `qft`: the same swaps, then controlled phase rotations and
    Hadamards in reverse order with negated angles.
    """
    for k in range(n // 2):
        swap(qs[k], qs[n - k - 1])
    for m in range(n):
        i = n - 1 - m
        for jm in range(n - 1 - i):
            j = n - 1 - jm
            crz(qs[j], qs[i], -pi / 2.0 ** (j - i))
        h(qs[i])
