"""Per-request registry, no mutable shared model cache."""
from .models import Statistical, Position, Association, Similarity, LightGBM, CatBoost, Neural, Transformer

CLASSES = (Statistical, Position, Association, LightGBM, CatBoost, Similarity, Neural, Transformer)


def train_models(board, history):
    models = []
    for cls in CLASSES:
        m = cls()
        m.rows, m.cols = board.rows, board.cols
        m.fit(history)
        models.append(m)
    return models
