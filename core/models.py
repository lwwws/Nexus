from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, Literal
import datetime as dt
import numpy as np

MessageId = str
ThreadId = str
SpaceId = str

@dataclass
class Message:
    id: MessageId
    timestamp: dt.datetime
    user: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Thread:
    id: ThreadId
    title: str = ""
    summary: str = ""

    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now())
    updated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Membership:
    message_id: MessageId
    thread_id: ThreadId
    score: float
    origin: str = "auto"     # "auto", "user"
    status: str = "active"   # "active", "rejected"
    reason: str = "unknown"  # "centroid_sim", "hdbscan", etc.
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now())