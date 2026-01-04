"""
Core Data Models
================

This module defines the fundamental data structures used throughout the chat
disentanglement system. It separates the data shape (Models) from the
storage logic (Stores).

Types:
    Message: An atomic unit of text from a chat platform.
    Thread: A semantic cluster of messages (a topic).
    Membership: The link between a Message and a Thread.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, Literal
import datetime as dt
import numpy as np

# --- Type Aliases ---

MessageId = str # Unique identifier for a message
ThreadId = str # Unique identifier for a thread cluster
SpaceId = str # Namespace identifier for vector embeddings

# --- Data Models ---

@dataclass
class Message:
    """
    Represents a single message unit of communication from the source platform.
    """
    id: MessageId # Unique identifier for the message
    timestamp: dt.datetime # The time the message was sent
    user: str # Name or ID of the sender
    text: str # The raw content of the message

    metadata: Dict[str, Any] = field(default_factory=dict) # Optional source-specific data (e.g., reply_to_id, attachments)


@dataclass
class Thread:
    """
    Represents a semantic cluster or topic of conversation.
    """
    id: ThreadId # System-generated unique ID for the thread
    title: str = "" # Short label for the topic
    summary: str = "" # Summary of the thread's content

    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now()) # Timestamp when the thread was first identified
    updated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now()) # Timestamp of the last modification (message added or summary updated)

    metadata: Dict[str, Any] = field(default_factory=dict) # Extra state (e.g., message_count, centroid_version)


@dataclass
class Membership:
    """
    Represents the association between a Message and a Thread.
    """
    message_id: MessageId # Reference to the message

    thread_id: ThreadId # Reference to the assigned thread

    score: float # Confidence score of the assignment

    origin: str = "auto" # Source of assignment (e.g. algorithm or human correction)
    status: str = "active" # Current state: 'active' (visible) or 'rejected' (soft-deleted)
    reason: str = "unknown" # The specific logic used (e.g., 'centroid_sim', 'hdbscan')

    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now()) # Timestamp when this assignment was created
