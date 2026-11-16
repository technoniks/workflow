from enum import StrEnum
from typing import Literal

def enum_to_literal(enum: StrEnum) -> Literal[str]:
  return Literal[tuple(i.value for i in enum)]