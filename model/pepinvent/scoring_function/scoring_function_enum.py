from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringFunctions:
    WeightedAverage: str = "weighted_average"
    GeometricMean: str = "geometric_mean"

ScoringFunctionsEnum = ScoringFunctions()
