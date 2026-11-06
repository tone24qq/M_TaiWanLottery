"""Local models. Training features only see a masked board, never hidden truth."""
from abc import ABC, abstractmethod
import importlib.util
import numpy as np
from .assignment import normalize
from .schema import Board


class LayoutModel(ABC):
    name = "base"
    minimum = 3
    features = ["target_position_history"]
    package = None

    def fit(self, history):
        self.history = history
        self.reason = "insufficient_complete_layouts"
        self.ready = len(history) >= self.minimum
        if self.package and importlib.util.find_spec(self.package) is None:
            self.ready, self.reason = False, "dependency_unavailable"
        return self

    @abstractmethod
    def scores(self, board): ...


class Statistical(LayoutModel):
    name, minimum = "statistical", 0
    features = ["permutation_constraint", "uniform_random_baseline"]

    def scores(self, board):
        return normalize(np.ones((len(board.missing), len(board.unknown))))


class Position(LayoutModel):
    name = "position"

    def scores(self, board):
        counts = np.ones((len(board.missing), len(board.unknown)))
        for h in self.history:
            for r, number in enumerate(board.missing):
                p = h.index(number)
                if p in board.unknown:
                    counts[r, board.unknown.index(p)] += 1
        return normalize(counts)


class Association(Position):
    name = "association"
    features = ["visible_pair_association", "target_position_history"]

    def scores(self, board):
        visible = [(i, x) for i, x in enumerate(board.cells) if x is not None]
        a = np.ones((len(board.missing), len(board.unknown)))
        for h in self.history:
            weight = (1 + sum(h[i] == x for i, x in visible)) / (1 + len(visible))
            for r, number in enumerate(board.missing):
                p = h.index(number)
                if p in board.unknown:
                    a[r, board.unknown.index(p)] += weight
        return normalize(a)


class Similarity(Association):
    name = "similarity_knn"
    features = ["same_layout_similarity", "visible_position_matches"]

    def scores(self, board):
        visible = [i for i, x in enumerate(board.cells) if x is not None]
        neighbors = sorted(enumerate(self.history), key=lambda item: (
            sum(item[1][i] != board.cells[i] for i in visible), item[0]))[:5]
        a = np.ones((len(board.missing), len(board.unknown)))
        for _, h in neighbors:
            d = sum(h[i] != board.cells[i] for i in visible)
            for r, number in enumerate(board.missing):
                p = h.index(number)
                if p in board.unknown:
                    a[r, board.unknown.index(p)] += 1 / (1 + d)
        return normalize(a)


def candidate_features(board, number):
    n = len(board.cells)
    visible = [(i, v) for i, v in enumerate(board.cells) if v is not None]
    values = []
    for p in board.unknown:
        row, col = divmod(p, board.cols)
        neighbors = [v / n for i, v in visible
                     if abs(i // board.cols-row) + abs(i % board.cols-col) == 1]
        values.append([number/n, row/max(board.rows-1, 1), col/max(board.cols-1, 1),
                       len(visible)/n, np.mean(neighbors) if neighbors else 0,
                       len(neighbors)/4, np.mean([v/n for _, v in visible]) if visible else 0])
    return np.asarray(values, dtype=np.float32)


class Learned(LayoutModel):
    minimum = 20
    features = ["target_value", "candidate_coordinates", "visible_neighbor_mean",
                "reveal_fraction", "visible_global_mean"]

    def fit(self, history):
        super().fit(history)
        if not self.ready:
            return self
        rng = np.random.default_rng(42)
        sequences, labels = [], []
        for h in history:
            for _ in range(2):
                cells = list(h)
                hidden = rng.choice(len(h), max(1, len(h)//2), replace=False)
                for i in hidden:
                    cells[i] = None
                b = Board(rows=self.rows, cols=self.cols, cells=cells)
                number = h[int(hidden[0])]
                sequences.append(candidate_features(b, number))
                labels.append(b.unknown.index(h.index(number)))
        if self.name in ("lightgbm", "catboost"):
            x = np.concatenate(sequences)
            y = np.concatenate([np.arange(len(s)) == label for s, label in zip(sequences, labels)])
            if len(set(y)) < 2:
                self.ready, self.reason = False, "insufficient_candidate_variation"
                return self
            if self.name == "lightgbm":
                from lightgbm import LGBMClassifier
                self.estimator = LGBMClassifier(n_estimators=40, num_leaves=7,
                    min_child_samples=5, random_state=42, n_jobs=1, verbosity=-1,
                    deterministic=True, force_col_wise=True)
            else:
                from catboost import CatBoostClassifier
                self.estimator = CatBoostClassifier(iterations=40, depth=4, random_seed=42,
                    thread_count=1, verbose=False, allow_writing_files=False)
            self.estimator.fit(x, y)
        else:
            import torch
            from torch import nn
            with torch.random.fork_rng():
                torch.manual_seed(42)
                self.net = CandidateNetwork(self.name == "transformer")
                optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
                self.net.train()
                x = torch.tensor(np.stack(sequences))
                y = torch.tensor(labels)
                for _ in range(15):
                    optimizer.zero_grad()
                    loss = nn.functional.cross_entropy(self.net(x), y)
                    loss.backward()
                    optimizer.step()
                self.net.eval()
        return self

    def scores(self, board):
        x = np.stack([candidate_features(board, number) for number in board.missing])
        if self.name in ("lightgbm", "catboost"):
            a = self.estimator.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(x.shape[:2])
        else:
            import torch
            with torch.no_grad():
                a = torch.softmax(self.net(torch.tensor(x)), dim=1).numpy()
        return normalize(a)


def CandidateNetwork(attention):
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(7, 16)
            self.attention = nn.MultiheadAttention(16, 2, dropout=0, batch_first=True) if attention else None
            self.output = nn.Linear(16, 1)

        def forward(self, x):
            x = self.input(x).relu()
            if self.attention is not None:
                x = x + self.attention(x, x, x, need_weights=False)[0]
            return self.output(x).squeeze(-1)
    return Network()


class LightGBM(Learned):
    name, package = "lightgbm", "lightgbm"


class CatBoost(Learned):
    name, package = "catboost", "catboost"


class Neural(Learned):
    name, package = "neural", "torch"


class Transformer(Learned):
    name, package = "transformer", "torch"
