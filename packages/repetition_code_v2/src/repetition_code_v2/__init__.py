"""repetition_code_v2: the bit-flip repetition code generalized to
arbitrary distance and repeated correction rounds, for the guppy-registry."""

from repetition_code_v2.repetition_code_v2 import (
    correct,
    correct_for_rounds,
    encode,
    extract_syndrome,
)

__all__ = [
    "correct",
    "correct_for_rounds",
    "encode",
    "extract_syndrome",
]
