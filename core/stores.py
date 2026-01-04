from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Tuple, Union

import numpy as np

from .models import Message, Thread, Membership, MessageId, ThreadId, SpaceId

@dataclass
class MessageStore:
    by_id: Dict[MessageId, Message] = field(default_factory=dict, repr=False)
    order: List[MessageId] = field(default_factory=list)

    def add(self, msgs: Iterable[Message]) -> None:
        for m in msgs:
            if m.id not in self.by_id:
                self.order.append(m.id)
            self.by_id[m.id] = m

    def get(self, mid: MessageId) -> Message:
        return self.by_id[mid]

    def has(self, mid: MessageId) -> bool:
        return mid in self.by_id

    def all(self) -> List[Message]:
        return [self.by_id[mid] for mid in self.order]

    def ids(self) -> List[ThreadId]:
        return list(self.order)

    def __repr__(self) -> str:
        return f"<MessageStore with {len(self.by_id)} messages>"

@dataclass
class ThreadStore:
    by_id: Dict[ThreadId, Thread] = field(default_factory=dict, repr=False)

    def add(self, threads: Iterable[Thread]) -> None:
        for t in threads:
            self.by_id[t.id] = t

    def get(self, tid: ThreadId) -> Thread:
        return self.by_id[tid]

    def has(self, tid: ThreadId) -> bool:
        return tid in self.by_id

    def all(self) -> List[Thread]:
        return list(self.by_id.values())

    def ids(self) -> List[ThreadId]:
        return list(self.by_id.keys())

    def __repr__(self) -> str:
        return f"<ThreadStore with {len(self.by_id)} threads>"

@dataclass
class EmbeddingStore:
    data: Dict[Tuple[SpaceId, str], np.ndarray] = field(default_factory=dict, repr=False)
    order: Dict[SpaceId, List[str]] = field(default_factory=dict)

    def add(self, space: SpaceId, ids: List[str], X: np.ndarray) -> None:
        self.order.setdefault(space, [])
        for i, _id in enumerate(ids):
            if (space, _id) not in self.data:
                self.order[space].append(_id)
            self.data[(space, _id)] = X[i].astype(np.float32)

    def get(self, space: SpaceId, _id: str) -> np.ndarray:
        return self.data[(space, _id)]

    def has(self, space: SpaceId, _id: str) -> bool:
        return (space, _id) in self.data

    def get_matrix(self, space: SpaceId) -> tuple[List[str], np.ndarray]:
        ids = self.order.get(space, [])
        X = np.stack([self.data[(space, _id)] for _id in ids], axis=0) if ids else np.zeros((0,0), dtype=np.float32)
        return ids, X

    def __repr__(self) -> str:
        return f"<EmbeddingStore with {len(self.data)} vectors>"

@dataclass
class MembershipStore:
    by_message: Dict[MessageId, List[Membership]] = field(default_factory=dict, repr=False)
    by_thread: Dict[ThreadId, List[Membership]] = field(default_factory=dict, repr=False)
    _all: List[Membership] = field(default_factory=list, repr=False)

    def add(self, memberships: Iterable[Membership]) -> None:
        for m in memberships:
            self._all.append(m)
            self.by_message.setdefault(m.message_id, []).append(m)
            self.by_thread.setdefault(m.thread_id, []).append(m)

    def for_message(self, mid: MessageId, status: str = "active") -> List[Membership]:
        members = self.by_message.get(mid, [])
        return [m for m in members if m.status == status]

    def for_thread(self, tid: ThreadId, status: str = "active") -> List[Membership]:
        members = self.by_thread.get(tid, [])
        return [m for m in members if m.status == status]

    def reject(self, mid: MessageId, tid: ThreadId) -> None:
        for m in self.by_message.get(mid, []):
            if m.thread_id == tid and m.status == "active":
                m.status = "rejected"

    def __repr__(self) -> str:
        return f"<MembershipStore with {len(self._all)} memberships>"
