"""
Data Stores
===========

This module implements the in-memory storage layer. It abstracts the
underlying data structures (dicts/lists) and provides optimized access
patterns for the processing pipeline.

Stores:
    MessageStore: Retrieving messages by ID and maintaining order.
    ThreadStore: Managing thread metadata.
    EmbeddingStore: Storing and retrieving vector representations efficiently.
    MembershipStore: Tracking relationships (Message <-> Thread).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Tuple, Union

import numpy as np

from .models import Message, Thread, Membership, MessageId, ThreadId, SpaceId

@dataclass
class MessageStore:
    """
    In-memory storage for raw messages.
    """
    by_id: Dict[MessageId, Message] = field(default_factory=dict, repr=False) # Maps ID -> Message object
    order: List[MessageId] = field(default_factory=list) # Maintains insertion order of messages

    def add(self, msgs: Iterable[Message]) -> None:
        """Bulk insert messages."""
        for m in msgs:
            if m.id not in self.by_id:
                self.order.append(m.id)
            self.by_id[m.id] = m

    def get(self, mid: MessageId) -> Message:
        """Retrieve a single message by ID."""
        return self.by_id[mid]

    def has(self, mid: MessageId) -> bool:
        """Check if a message exists."""
        return mid in self.by_id

    def all(self) -> List[Message]:
        """Retrieve all messages in insertion order."""
        return [self.by_id[mid] for mid in self.order]

    def ids(self) -> List[ThreadId]:
        """Retrieve all message IDs in insertion order."""
        return list(self.order)

    def __repr__(self) -> str:
        return f"<MessageStore with {len(self.by_id)} messages>"

@dataclass
class ThreadStore:
    """
    In-memory storage for Thread clusters.
    """
    by_id: Dict[ThreadId, Thread] = field(default_factory=dict, repr=False) # Maps ID -> Thread object

    def add(self, threads: Iterable[Thread]) -> None:
        """Insert or update threads."""
        for t in threads:
            self.by_id[t.id] = t

    def get(self, tid: ThreadId) -> Thread:
        """Retrieve a thread by ID."""
        return self.by_id[tid]

    def has(self, tid: ThreadId) -> bool:
        """Check if a thread exists."""
        return tid in self.by_id

    def all(self) -> List[Thread]:
        """Retrieve all threads."""
        return list(self.by_id.values())

    def ids(self) -> List[ThreadId]:
        """Retrieve all thread IDs."""
        return list(self.by_id.keys())

    def __repr__(self) -> str:
        return f"<ThreadStore with {len(self.by_id)} threads>"

@dataclass
class EmbeddingStore:
    """
    Storage for high-dimensional vector representations.
    Supports efficient matrix retrieval for clustering.
    """
    data: Dict[Tuple[SpaceId, str], np.ndarray] = field(default_factory=dict, repr=False) # Maps (Space, ID) -> Vector
    order: Dict[SpaceId, List[str]] = field(default_factory=dict) # Index ensuring stable matrix row order

    def add(self, space: SpaceId, ids: List[str], X: np.ndarray) -> None:
        """
        Store multiple vectors into a specific namespace.
        Args:
            space: Namespace (e.g. 'raw', 'umap')
            ids: List of IDs corresponding to the rows in X
            X: Matrix of shape (N, Dim)
        """
        self.order.setdefault(space, [])
        for i, _id in enumerate(ids):
            if (space, _id) not in self.data:
                self.order[space].append(_id)
            self.data[(space, _id)] = X[i].astype(np.float32)

    def get(self, space: SpaceId, _id: str) -> np.ndarray:
        """Get a single vector."""
        return self.data[(space, _id)]

    def has(self, space: SpaceId, _id: str) -> bool:
        """Check if vector exists."""
        return (space, _id) in self.data

    def get_matrix(self, space: SpaceId) -> tuple[List[str], np.ndarray]:
        """
        Retrieve all vectors in a space as a contiguous matrix.
        Returns: (List of IDs, Matrix)
        """
        ids = self.order.get(space, [])
        X = np.stack([self.data[(space, _id)] for _id in ids], axis=0) if ids else np.zeros((0,0), dtype=np.float32)
        return ids, X

    def __repr__(self) -> str:
        return f"<EmbeddingStore with {len(self.data)} vectors>"

@dataclass
class MembershipStore:
    """
    Relational store for mapping Messages to Threads (Many-to-Many).
    """
    by_message: Dict[MessageId, List[Membership]] = field(default_factory=dict, repr=False) # Index for fast message lookup
    by_thread: Dict[ThreadId, List[Membership]] = field(default_factory=dict, repr=False) # Index for fast thread lookup
    _all: List[Membership] = field(default_factory=list, repr=False) # Flat list of all memberships

    def add(self, memberships: Iterable[Membership]) -> None:
        """Bulk insert memberships and update indices."""
        for m in memberships:
            self._all.append(m)
            self.by_message.setdefault(m.message_id, []).append(m)
            self.by_thread.setdefault(m.thread_id, []).append(m)

    def for_message(self, mid: MessageId, status: str = "active") -> List[Membership]:
        """Get all threads a message belongs to."""
        members = self.by_message.get(mid, [])
        return [m for m in members if m.status == status]

    def for_thread(self, tid: ThreadId, status: str = "active") -> List[Membership]:
        """Get all messages in a thread."""
        members = self.by_thread.get(tid, [])
        return [m for m in members if m.status == status]

    def reject(self, mid: MessageId, tid: ThreadId) -> None:
        """Soft-delete a specific assignment."""
        for m in self.by_message.get(mid, []):
            if m.thread_id == tid and m.status == "active":
                m.status = "rejected"

    def __repr__(self) -> str:
        return f"<MembershipStore with {len(self._all)} memberships>"