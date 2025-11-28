from dataclasses import dataclass
from typing import Any


@dataclass
class SamplingRunnerDTO:
    model_network: Any = None
    max_sequence_length: int = None
    vocabulary: Any = None
    device: Any = None
    tokenizer: Any = None