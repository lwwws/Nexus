from typing import Protocol, List, Optional, Sequence, Tuple
import numpy as np
from .models import Message, ThreadId, MessageId

class Formatter(Protocol):
    def format(self, idx: int, messages: List[Message]) -> str:
        pass

    def format_all(self, messages: List[Message]) -> List[str]:
        """Override if needed"""
        return [self.format(i, messages) for i in range(len(messages))]

class Embedder(Protocol):
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """(n texts) -> (n, d)"""
        pass

class Reducer(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

class Clusterer(Protocol):
    def cluster(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(n, k) -> labels shape (n,), scores shape (n,)
        scores can be all 1.0 if method doesn't provide."""
        pass

class ThreadRepComputer(Protocol):
    def compute(self, thread_id: ThreadId) -> np.ndarray:
        """Compute thread representation (e.g. centroid)."""
        pass

    def compute_all(self, thread_ids: List[ThreadId]) -> np.ndarray:
        """Override if needed"""
        return np.stack([self.compute(id) for id in thread_ids], axis=0)

class Assigner(Protocol):
    def assign(self, message_id: MessageId) -> List[Tuple[ThreadId, float]]:
        """Return list of (thread_id, score) for multi-membership."""
        pass

class UpdateStrategy(Protocol):
    def on_new_message(self, message_id: MessageId) -> None:
        """What to do when a new message arrives (e.g. immediate vs buffer)."""
        pass

    def flush(self) -> None:
        """Process pending messages if needed."""
        pass