from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from core.models import Message, MessageId, ThreadId, SpaceId, UpdateResult
from core.stores import EmbeddingStore, MembershipStore, MessageStore, ThreadStore
from core.interfaces import Formatter, Embedder, Reducer, Clusterer, ThreadRepComputer, UpdateStrategy, ThreadLabeler

import os
import re
import datetime
import pandas as pd
from huggingface_hub import hf_hub_download

import logging

from utils import UserMapper

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

    def __init__(self,
                 window_back: int = 3,
                 window_fwd: int = 1,
                 time_threshold_minutes: int = 10,
                 repeat_center: int = 2):

        self.window_back = window_back
        self.window_fwd = window_fwd
        self.time_threshold_minutes = time_threshold_minutes
        self.repeat_center = repeat_center

        # Initialize the mapper here
        self.mapper = UserMapper()

    def format(self, idx: int, messages: List[Message]) -> str:
        cur = messages[idx]
        cur_time = cur.timestamp

        # Transform current user
        user_alias = self.mapper.get_alias(cur.user)

        # Collect Context (Backwards)
        back_parts: List[str] = []
        for k in range(1, self.window_back + 1):
            j = idx - k
            if j < 0:
                break
            prev = messages[j]

            # Stop if context is too old
            dt_min = (cur_time - prev.timestamp).total_seconds() / 60.0
            if dt_min > self.time_threshold_minutes:
                break

            # Map context user
            prev_alias = self.mapper.get_alias(prev.user)
            back_parts.insert(0, f"{prev_alias}: {prev.text}")

        # Collect Context (Forwards)
        fwd_parts: List[str] = []
        for k in range(1, self.window_fwd + 1):
            j = idx + k
            if j >= len(messages):
                break
            nxt = messages[j]

            dt_min = (nxt.timestamp - cur_time).total_seconds() / 60.0
            if dt_min > self.time_threshold_minutes:
                break

            # Map context user
            nxt_alias = self.mapper.get_alias(nxt.user)
            fwd_parts.append(f"{nxt_alias}: {nxt.text}")

        # Construct Target String
        target_str = f"{user_alias}: {cur.text}"

        # Apply weighting (repetition)
        # We repeat the target 'focus_weight' times to increase its vector influence
        weighted_target = " ; ".join([target_str] * self.repeat_center)

        context_str = " | ".join(back_parts + fwd_parts)

        out = f"Context: {context_str} >>> Focus: {weighted_target}" if context_str else f"Focus: {weighted_target}"
        logging.info(f"Formatted text: {out}")
        return out

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


# ---------- Buffer Strategy ----------
class BufferedUpdateStrategy(UpdateStrategy):
    """
    Hybrid Strategy:
    1. Fast Path: Compare new message to existing Thread Centroids.
       If sim >= Dynamic Threshold (percentile of thread), assign immediately.
    2. Slow Path: If no match, add to Buffer.
    3. Flush: When buffer is full, run clustering on buffered messages.
       - Any noise/unclustered messages are kept in a separate pending buffer for retry.
    """

    def __init__(self,
                 embeddings: EmbeddingStore,
                 threads: ThreadStore,
                 memberships: MembershipStore,
                 clusterer: Clusterer,
                 msg_sim_space: SpaceId = "msg:full",
                 msg_cluster_space: SpaceId = "msg:cluster",
                 thread_centroid_space: SpaceId = "thread:centroid",
                 global_min_threshold: float = 0.65,
                 percentile_threshold: int = 10,
                 buffer_size_limit: int = 15,
                 pending_size_limit: int = 50,
                 min_delta: float = 0.05
                 ):

        self.embeddings = embeddings
        self.threads = threads
        self.memberships = memberships
        self.clusterer = clusterer

        self.msg_sim_space = msg_sim_space
        self.msg_cluster_space = msg_cluster_space
        self.thread_centroid_space = thread_centroid_space

        self.global_min_threshold = global_min_threshold
        self.percentile_threshold = percentile_threshold

        self.buffer_limit = buffer_size_limit
        self.pending_limit = pending_size_limit

        self.min_delta = min_delta

        self._buffer: List[MessageId] = []
        self._pending: List[MessageId] = []

    def on_new_message(self, message_id: MessageId) -> UpdateResult:
        best_tid, best_score, second_tid, second_score = self._find_top2_threads(message_id)

        if best_tid:
            threshold = self._compute_thread_threshold(best_tid)
            delta = best_score - second_score

            if best_score >= threshold and delta >= self.min_delta:
                logger.info(
                    f"(---) Fast Assign: {message_id} -> {best_tid} "
                    f"(score={best_score:.3f} thr={threshold:.3f} delta={delta:.3f})"
                )
                return UpdateResult(action="assigned", assigned_thread_id=best_tid, assigned_score=best_score)

            logger.info(
                f"(XXX) Fast Reject (buffered): {message_id} best={best_tid} "
                f"(score={best_score:.3f} thr={threshold:.3f} delta={delta:.3f} "
                f"second={second_tid}:{second_score:.3f})"
            )

        self._buffer.append(message_id)
        logger.info(f"Buffer length: {len(self._buffer)}")

        # Optional: if pending is already large, try to flush earlier
        if len(self._buffer) >= self.buffer_limit:
            return self.flush()

        return UpdateResult(action="buffered")

    def flush(self) -> UpdateResult:
        """
        Flush strategy:
        - Cluster (pending + buffer) to give HDBSCAN enough points.
        - Keep noise (unassigned) in pending for later retry.
        - Not dropping messages missing embeddings; keep them pending.
        """
        if not self._buffer and not self._pending:
            return UpdateResult(action="buffered")

        # Combine to form a larger batch (helps "single-topic but small buffer" cases)
        batch_ids = self._pending + self._buffer
        self._buffer = []  # we'll rebuild pending below

        logger.info(f"Strategy flushing batch={len(batch_ids)} (pending={len(self._pending)})")

        vecs: List[np.ndarray] = []
        valid_ids: List[MessageId] = []
        missing_ids: List[MessageId] = []

        for mid in batch_ids:
            if self.embeddings.has(self.msg_cluster_space, mid):
                vecs.append(self.embeddings.get(self.msg_cluster_space, mid))
                valid_ids.append(mid)
            else:
                missing_ids.append(mid)

        X = np.stack(vecs, axis=0)

        try:
            labels, scores = self.clusterer.cluster(X)
        except ValueError as e:
            logger.warning(f"Cluster failed on n={len(valid_ids)}: {e}")
            self._pending = (missing_ids + valid_ids)[-self.pending_limit:]
            return UpdateResult(action="buffered")

        # Identify which messages are "noise" (label_list empty in your adapter)
        noise_ids: List[MessageId] = [mid for mid, lab in zip(valid_ids, labels) if not lab]

        # Keep missing + noise for retry, cap size
        self._pending = (missing_ids + noise_ids)[-self.pending_limit:]

        # Return the clustering result for processor to create threads/memberships
        return UpdateResult(
            action="flushed",
            flush_ids=valid_ids,
            flush_labels=labels,
            flush_scores=scores
        )

    def _find_top2_threads(self, mid: MessageId) -> Tuple[Optional[str], float, Optional[str], float]:
        if not self.embeddings.has(self.msg_sim_space, mid):
            return None, 0.0, None, 0.0
        msg_vec = self.embeddings.get(self.msg_sim_space, mid)

        best_tid, best_score = None, -1.0
        second_tid, second_score = None, -1.0

        for t in self.threads.all():
            if not self.embeddings.has(self.thread_centroid_space, t.id):
                continue
            c = self.embeddings.get(self.thread_centroid_space, t.id)
            sim = self._cosine(msg_vec, c)

            if sim > best_score:
                second_tid, second_score = best_tid, best_score
                best_tid, best_score = t.id, sim
            elif sim > second_score:
                second_tid, second_score = t.id, sim

        if best_score < 0:
            return None, 0.0, None, 0.0
        if second_score < 0:
            return best_tid, best_score, None, 0.0

        return best_tid, best_score, second_tid, second_score
    def _compute_thread_threshold(self, tid: str) -> float:
        """
        Dynamically calculates the acceptance threshold (5th percentile).
        """
        # Get Members
        members = self.memberships.for_thread(tid, status="active")

        if len(members) < 10:
            return self.global_min_threshold

        # Use ALL cached similarities (Fastest & Most Accurate)
        sims = [m.centroid_similarity for m in members if m.centroid_similarity >= 0]

        # Fallback to Vector Calculation (Slow)
        # Only do this if we don't have enough cached data.
        if len(sims) < 10:
            logger.info(f"Not enough similarities... len(sims)={len(sims)}")
            if not self.embeddings.has(self.thread_centroid_space, tid):
                return self.global_min_threshold

            centroid = self.embeddings.get(self.thread_centroid_space, tid)

            # SAFETY LIMIT: Only fetch vectors for the last messages
            target_members = members[-100:]

            for m in target_members:
                if self.embeddings.has(self.msg_sim_space, m.message_id):
                    v = self.embeddings.get(self.msg_sim_space, m.message_id)
                    sims.append(self._cosine(v, centroid))

        if not sims:
            return self.global_min_threshold

        # Compute Percentile
        perc = float(np.percentile(sims, self.percentile_threshold))

        final = max(self.global_min_threshold, perc)

        # Log if we are being adaptive
        if perc > self.global_min_threshold:
            logger.info(f"Adaptive ({tid}): Using percentile={perc:.3f} (N={len(sims)})")
        else:
            logger.info(f"Using global min of {self.global_min_threshold} for tid=({tid}) ")

        return final

    def _cosine(self, a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


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

        # Download Model
        if not os.path.exists(model_path):
            model_path = hf_hub_download(
                repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
                filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                local_dir="./models"  # Downloads to a 'models' folder in your current directory
            )
            logger.info("Downloaded Llama model to %s", model_path)

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