from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from core.models import Message, MessageId, ThreadId, SpaceId
from core.stores import EmbeddingStore, MembershipStore, MessageStore, ThreadStore
from core.interfaces import Formatter, Embedder, Reducer, Clusterer, ThreadRepComputer, Assigner, UpdateStrategy, ThreadLabeler

import re
import datetime
import numpy as np
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Formatter ----------

@dataclass
class ContextWindowFormatterOld(Formatter):
    window_back: int = 2
    window_fwd: int = 1
    time_threshold_minutes: int = 10
    repeat_center: int = 2

    def format(self, idx: int, messages: List[Message]) -> str:
        cur = messages[idx]
        cur_time = cur.timestamp

        back_parts: List[str] = []
        for k in range(1, self.window_back + 1):
            j = idx - k
            if j < 0:
                break
            prev = messages[j]
            dt_min = (cur_time - prev.timestamp).total_seconds() / 60.0
            if dt_min > self.time_threshold_minutes:
                break
            back_parts.insert(0, f"{prev.user}: {prev.text}")

        fwd_parts: List[str] = []
        for k in range(1, self.window_fwd + 1):
            j = idx + k
            if j >= len(messages):
                break
            nxt = messages[j]
            dt_min = (nxt.timestamp - cur_time).total_seconds() / 60.0
            if dt_min > self.time_threshold_minutes:
                break
            fwd_parts.append(f"{nxt.user}: {nxt.text}")

        center = f"{cur.user}: {cur.text}"
        full = back_parts + [center] * self.repeat_center + fwd_parts
        return " \n ".join(full)


@dataclass
class ContextWindowFormatter(Formatter):
    """
    Formats messages by clearly separating context from the target message.
    It boosts the signal of the target message by placing it at the end
    and optionally repeating it (weighted focus).
    """
    window_back: int = 3  # Look back 3 messages for context
    window_fwd: int = 1  # Look forward 1 message (if available)
    time_threshold_minutes: int = 10
    repeat_center: int = 2  # How many times to repeat the target message

    def format(self, idx: int, messages: List[Message]) -> str:
        cur = messages[idx]
        cur_time = cur.timestamp

        # 1. Collect Context (Backwards)
        back_parts: List[str] = []
        for k in range(1, self.window_back + 1):
            j = idx - k
            if j < 0:
                break
            prev = messages[j]

            # Stop if time gap is too large (context is stale)
            dt_min = (cur_time - prev.timestamp).total_seconds() / 60.0
            if dt_min > self.time_threshold_minutes:
                break

            # Insert at beginning to maintain chronological order
            back_parts.insert(0, f"{prev.user}: {prev.text}")

        # 2. Collect Context (Forwards) - Optional but helps disambiguate
        fwd_parts: List[str] = []
        for k in range(1, self.window_fwd + 1):
            j = idx + k
            if j >= len(messages):
                break
            nxt = messages[j]

            dt_min = (nxt.timestamp - cur_time).total_seconds() / 60.0
            if dt_min > self.time_threshold_minutes:
                break
            fwd_parts.append(f"{nxt.user}: {nxt.text}")

        # 3. Construct the String
        # Strategy: "Context >>> Focus"

        # Combine back and forward parts into a single context string
        context_str = " | ".join(back_parts + fwd_parts)

        # The target message
        target_str = f"{cur.user}: {cur.text}"

        # Apply weighting (repetition)
        # We repeat the target 'focus_weight' times to increase its vector influence
        weighted_target = " ; ".join([target_str] * self.repeat_center)

        if context_str:
            # We use ">>>" as a strong visual separator for the model
            return f"Context: {context_str} >>> Focus: {weighted_target}"
        else:
            # If no context, just return the weighted target
            return f"Focus: {weighted_target}"

# ---------- Embedder ----------

class MiniLMEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        X = self.model.encode(texts, show_progress_bar=True)
        return np.asarray(X, dtype=np.float32)


# ---------- Reducer ----------

class UMAPReducer(Reducer):
    def __init__(
        self,
        n_neighbors: int = 30,
        n_components: int = 5,
        min_dist: float = 0.0,
        metric: str = "cosine",
        random_state: int = 42,
    ):
        import umap
        self._umap = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        self._is_fit = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        Y = self._umap.fit_transform(X)
        self._is_fit = True
        return np.asarray(Y, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("UMAPReducer.transform() called before fit_transform().")
        Y = self._umap.transform(X)
        return np.asarray(Y, dtype=np.float32)


# ---------- Clusterer ----------

class HDBSCANClusterer(Clusterer):
    def __init__(
        self,
        min_cluster_size: int = 30,
        min_samples: int = 3,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ):
        import hdbscan
        self._hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )

    def cluster(self, X: np.ndarray) -> Tuple[List[List[int]], List[List[float]]]:
        # Fit the model
        self._hdbscan.fit(X)

        # Get flat arrays (standard HDBSCAN output)
        flat_labels = self._hdbscan.labels_.astype(int)

        # Handle probabilities safely
        probs = getattr(self._hdbscan, "probabilities_", None)
        if probs is None:
            flat_scores = np.ones((len(flat_labels),), dtype=np.float32)
        else:
            flat_scores = np.asarray(probs, dtype=np.float32)

        # Transform flat arrays -> List of Lists (Adapter Logic)
        labels_out = []
        scores_out = []

        for label, score in zip(flat_labels, flat_scores):
            if label == -1:
                # Noise (-1) becomes an empty list -> belongs to no threads
                labels_out.append([])
                scores_out.append([])
            else:
                # Valid label becomes a single-item list
                labels_out.append([int(label)])
                scores_out.append([float(score)])

        return labels_out, scores_out


# ---------- ThreadRepComputer ----------

@dataclass
class CentroidThreadRepComputer(ThreadRepComputer):
    memberships: MembershipStore
    embeddings: EmbeddingStore
    msg_space: SpaceId = "msg:full"

    def _infer_dim(self) -> int:
        # find any message embedding to determine d
        for (space, _id), v in self.embeddings.data.items():
            if space == self.msg_space:
                return int(v.shape[0])
        return 0

    def compute(self, thread_id: ThreadId) -> np.ndarray:
        ms = self.memberships.for_thread(thread_id, status="active")
        mids = [m.message_id for m in ms]
        if not mids:
            d = self._infer_dim()
            return np.zeros((d,), dtype=np.float32) if d else np.zeros((0,), dtype=np.float32)

        vecs = []
        for mid in mids:
            if self.embeddings.has(self.msg_space, mid):
                vecs.append(self.embeddings.get(self.msg_space, mid))
        if not vecs:
            d = self._infer_dim()
            return np.zeros((d,), dtype=np.float32) if d else np.zeros((0,), dtype=np.float32)

        V = np.stack(vecs, axis=0).astype(np.float32)
        return V.mean(axis=0)


# ---------- Stubs for batch mode (run_batch doesn't need streaming assignment) ----------

class NoOpAssigner(Assigner):
    def assign(self, message_id: MessageId):
        return []

class NoOpUpdateStrategy(UpdateStrategy):
    def on_new_message(self, message_id: MessageId) -> None:
        return
    def flush(self) -> None:
        return

# ---------- ThreadLabeler ----------

class LlamaThreadLabelerOld(ThreadLabeler):
    def __init__(self,
                 model_path: str,
                 n_ctx: int = 2048,
                 n_gpu_layers: int = -1,
                 max_msg_chars: int = 500): # Increased char limit slightly
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("Please `pip install llama-cpp-python`")

        self.max_msg_chars = max_msg_chars

        # verbose=False keeps logs clean
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

    def label(self, messages: List[Message]) -> Tuple[str, str]:
        # Build Context
        lines = []
        for m in messages:
            text = m.text.replace("\n", " ").strip()
            if not text: continue # Skip empty messages

            if len(text) > self.max_msg_chars:
                text = text[:self.max_msg_chars] + "..."

            lines.append(f"- {text}")

        # If no valid text, return generic
        if not lines:
            return "Empty Thread", "No content."

        chat_block = "\n".join(lines)

        prompt = f"""Conversation:
{chat_block}

Instructions:
1. Provide a short label (3-5 words) for this conversation.
2. Provide a summary, made up of 2-4 sentences at most.

Output format:
Title: [Label]
Summary: [Summary]

Response:
"""

        # Inference
        output = self.llm(
            prompt,
            max_tokens=128,
            temperature=0.3,
            stop=["\n\n", "Conversation:"],
            echo=False
        )

        raw_text = output['choices'][0]['text'].strip()

        # Robust Parsing
        title = None
        summary = None

        for line in raw_text.split('\n'):
            line = line.strip()
            if not line: continue

            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()

        # Fallback: If "Title:" keyword is missing, just assume the first line is the title
        if not title:
            # Clean up common chatty prefixes if the model ignores instructions
            first_line = raw_text.split('\n')[0].strip()
            if first_line:
                title = first_line.replace("Label:", "").replace("Topic:", "").strip()
            else:
                title = "Untitled Topic"

        if not summary:
            summary = "Summary not generated."

        logging.info(f"DEBUG: Raw LLM Output: {raw_text[:100]}... -> Parsed: {title}")

        return title, summary


class LlamaThreadLabeler(ThreadLabeler):
    def __init__(self,
                 model_path: str,
                 n_ctx: int = 2048,
                 n_gpu_layers: int = -1,
                 max_msg_chars: int = 500):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("Please `pip install llama-cpp-python`")

        self.max_msg_chars = max_msg_chars
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

    def label(self, messages: List[Message]) -> Tuple[str, str]:
        # BUILD CONTEXT (Same as before)
        lines = []
        prev_time = None

        for m in messages:
            text = m.text.replace("\n", " ").strip()
            if not text: continue

            if len(text) > self.max_msg_chars:
                text = text[:self.max_msg_chars] + "..."

            if prev_time:
                delta = m.timestamp - prev_time
                if delta.days > 0:
                    lines.append(f"\n... ({delta.days} days later) ...\n")
                elif delta.total_seconds() > 7200:
                    hours = int(delta.total_seconds() / 3600)
                    lines.append(f"\n... ({hours} hours later) ...\n")

            lines.append(f"- {m.user}: {text}")
            prev_time = m.timestamp

        if not lines:
            return "Empty Thread", "No content."

        chat_block = "\n".join(lines)

        # PROMPT
        prompt = f"""Review these highlights from a conversation.
Conversation:
{chat_block}

Instructions:
1. Write a Title: 3-5 words starting with an Emoji.
2. Write a Summary: 1-3 sentences.

Output Format:
Title: [Emoji] [Short Label]
Summary: [Text]

Response:
"""

        # INFERENCE
        output = self.llm(
            prompt,
            max_tokens=150,
            temperature=0.2,
            stop=["Conversation:", "Instructions:", "User:"],
            echo=False
        )

        raw_text = output['choices'][0]['text'].strip()

        # DETERMINISTIC PARSING
        # No more magic numbers. We rely on the order of lines.

        # Clean up markdown bolding just in case (**text** -> text)
        clean_text = raw_text.replace('**', '').replace('##', '')
        raw_lines = [l.strip() for l in clean_text.split('\n') if l.strip()]

        title_part = "Untitled Topic"
        summary_parts = []

        # Flag to know if we've finished finding any parts
        found_title = False
        found_summary = False

        for i, line in enumerate(raw_lines):
            lower = line.lower()

            # If we see "Title:" AGAIN after already finding it, stop immediately.
            if found_title and lower.startswith("title:"):
                break

            # If we see the exact same title string appearing in the summary body, stop.
            if found_title and line.strip() == title_part:
                break

            # EXPLICIT PARSING
            if lower.startswith("title:"):
                title_part = line.split(":", 1)[1].strip()
                found_title = True
                continue

            if lower.startswith("summary:"):
                # If we see "Summary:" a second time, break
                if found_summary:
                    break
                summary_parts.append(line.split(":", 1)[1].strip())
                found_title = True  # We found summary, so title phase is definitely over
                found_summary = True
                continue

            # IMPLICIT PARSING
            if not found_title:
                # First line logic (Dangling Emoji Handling)
                if len(line) < 5 and i + 1 < len(raw_lines):
                    title_part = line
                else:
                    if title_part != "Untitled Topic" and len(title_part) < 5:
                        title_part = f"{title_part} {line}"
                    else:
                        title_part = line
                    found_title = True
            else:
                # We are in the summary section
                summary_parts.append(line)

        # Final Cleanup
        final_summary = " ".join(summary_parts).strip()
        if not final_summary: final_summary = "Summary not provided."

        # Strip potential lingering prefixes inside the string
        title_part = title_part.strip('"').strip("'")

        logging.info(f"Labeler: {title_part} | Sum: {final_summary[:30]}...")
        return title_part, final_summary