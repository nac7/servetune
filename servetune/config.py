# SPDX-License-Identifier: Apache-2.0
"""Serving configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Quant(str, Enum):
    """Weight quantization scheme."""

    FP16 = "fp16"
    FP8 = "fp8"
    INT4_AWQ = "int4-awq"


class KVDtype(str, Enum):
    """KV-cache dtype."""

    FP16 = "fp16"
    FP8 = "fp8"


@dataclass(frozen=True)
class ServeConfig:
    """A single candidate serving configuration."""

    quant: Quant = Quant.FP16
    spec_decode: bool = False
    kv_dtype: KVDtype = KVDtype.FP16
    max_batch: int = 8
    max_len: int = 4096

    @property
    def is_reference(self) -> bool:
        """The full-precision, lossless reference everything is compared against."""
        return (
            self.quant == Quant.FP16
            and not self.spec_decode
            and self.kv_dtype == KVDtype.FP16
        )

    def label(self) -> str:
        parts = [self.quant.value]
        if self.spec_decode:
            parts.append("spec")
        parts.append(f"kv:{self.kv_dtype.value}")
        return " + ".join(parts)
