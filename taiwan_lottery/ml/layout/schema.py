"""Validated row-major boards. Client coordinates are one-based."""
from pydantic import BaseModel, Field, model_validator


class Board(BaseModel):
    rows: int = Field(ge=1, le=100)
    cols: int = Field(ge=1, le=100)
    cells: list[int | None]

    @model_validator(mode="after")
    def permutation(self):
        n = self.rows * self.cols
        if n > 400 or len(self.cells) != n:
            raise ValueError("Require rows*cols cells, at most 400 per request")
        seen = [x for x in self.cells if x is not None]
        if any(x < 1 or x > n for x in seen) or len(set(seen)) != len(seen):
            raise ValueError("Revealed values must be unique integers in 1..N")
        return self

    @property
    def unknown(self):
        return [i for i, x in enumerate(self.cells) if x is None]

    @property
    def missing(self):
        return sorted(set(range(1, len(self.cells) + 1)) - set(self.cells))

    def coordinate(self, i):
        return {"row": i // self.cols + 1, "col": i % self.cols + 1}


class PredictionRequest(Board):
    target: int = Field(ge=1)

    @model_validator(mode="after")
    def target_valid(self):
        if self.target > len(self.cells):
            raise ValueError("Target outside 1..N")
        return self


class HistoryRequest(Board):
    source: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def complete(self):
        if None in self.cells:
            raise ValueError("Training history must be a confirmed complete permutation")
        return self


class EvaluationRequest(BaseModel):
    rows: int = Field(ge=1, le=100)
    cols: int = Field(ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    revealed_fraction: float = Field(default=0.5, gt=0, lt=1)
    max_folds: int = Field(default=8, ge=1, le=30)


class Resolution(BaseModel):
    row: int = Field(ge=1)
    col: int = Field(ge=1)
