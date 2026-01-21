"""
Core Interfaces (Protocols)
===========================

This module defines the abstract contracts (Protocols) for the interchangeable
strategies in the processing pipeline. Implementing these interfaces allows
hot-swapping of components (e.g., changing from TF-IDF to BERT or switching
clustering algorithms) without modifying the core logic.

Protocols:
    Formatter: Prepares message text for embedding (e.g. adding context).
    Embedder: Converts text strings into vector representations.
    Reducer: Reduces vector dimensionality (e.g., UMAP, PCA).
    Clusterer: Identifies semantic groups in vector space (e.g., HDBSCAN).
    ThreadRepComputer: Calculates the representative vector for a thread.
    Assigner: Matches new messages to existing threads.
    UpdateStrategy: Manages the lifecycle of new messages (buffering vs. immediate).
"""
from typing import Protocol, List, Optional, Sequence, Tuple
import numpy as np
from .models import Message, ThreadId, MessageId, UpdateResult


class Formatter(Protocol):
    """
    Strategy for converting Message objects into a string format suitable for embedding.
    Useful for adding context (e.g. "User: [Text]") or combining multiple messages.
    """
    def format(self, idx: int, messages: List[Message]) -> str:
        """Format a single message (potentially looking at neighbors)."""
        pass

    def format_all(self, messages: List[Message]) -> List[str]:
        """Batch format messages. Override for efficiency if needed."""
        return [self.format(i, messages) for i in range(len(messages))]


class Embedder(Protocol):
    """
    Strategy for converting text into high-dimensional vectors.
    """
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        Returns: Matrix of shape (n_samples, n_features).
        """
        pass


class Reducer(Protocol):
    """
    Strategy for dimensionality reduction (optional step before clustering).
    """
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform the data."""
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted reducer."""
        pass


class Clusterer(Protocol):
    """
    Strategy for discovering structure in data.
    """

    def cluster(self, X: np.ndarray) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Cluster the input data allowing for multiple labels per item.

        Returns:
            labels: A list where index `i` contains a list of cluster IDs for message `i`.
                    Example: [[1, 5], [2], [], [1]]
            scores: A list where index `i` contains a list of confidence scores matching the labels.
                    Example: [[0.9, 0.4], [1.0], [], [0.8]]
        """
        pass


class ThreadRepComputer(Protocol):
    """
    Strategy for calculating the mathematical representation of a thread
    (e.g., computing the Centroid or Medoid).
    """
    def compute(self, thread_id: ThreadId) -> np.ndarray:
        """Compute the representation vector for a single thread."""
        pass

    def compute_all(self, thread_ids: List[ThreadId]) -> np.ndarray:
        """Compute representations for multiple threads. Override for batch efficiency."""
        return np.stack([self.compute(id) for id in thread_ids], axis=0)


class UpdateStrategy(Protocol):
    """
    Policy for handling real-time updates.
    Decides whether to process a message immediately or buffer it.
    """
    def on_new_message(self, message_id: MessageId) -> UpdateResult:
        """Process a new message and return a decision."""
        pass

    def flush(self)-> UpdateResult:
        """Force process pending buffer."""
        pass

class ThreadLabeler(Protocol):
    """
    Strategy for generating human-readable metadata (title, summary) for a thread
    based on its content.
    """
    def label(self, messages: List[Message]) -> Tuple[str, str]:
        """
        Analyze a list of messages and return a tuple: (Title, Summary).
        """
        pass