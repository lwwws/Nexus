"""
Chat Processor
==============

This module acts as the central engine (Controller) of the system.
It coordinates the interaction between:
1. Data Stores (holding the state)
2. Strategy Interfaces (executing the logic)

It is responsible for the full lifecycle of a message:
Ingestion -> Embedding -> Reduction -> Clustering -> Assignment -> Retrieval.
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Callable, Set, Dict, Any, Tuple
import numpy as np
import uuid

from .models import Message, Thread, Membership, SpaceId, MessageId, ThreadId
from .stores import MessageStore, ThreadStore, MembershipStore, EmbeddingStore
from .interfaces import Formatter, Embedder, Reducer, Clusterer, ThreadRepComputer, Assigner, UpdateStrategy

logger = logging.getLogger(__name__)


def new_thread_id() -> str:
    """Generates a unique, URL-safe thread identifier."""
    return f"thread_{uuid.uuid4().hex[:10]}"


@dataclass
class ChatProcessor:
    """
    The main coordinator class.

    Attributes:
        messages (MessageStore): Storage for raw message objects.
        threads (ThreadStore): Storage for thread clusters.
        memberships (MembershipStore): Storage for message-thread links.
        embeddings (EmbeddingStore): Storage for vectors.

        embedder (Embedder): Strategy to convert text to vectors.
        reducer (Reducer): Optional strategy to lower vector dimensionality.
        clusterer (Clusterer): Optional strategy to group vectors (HDBSCAN, etc.).

        thread_rep_computer (ThreadRepComputer): Strategy to calculate centroids.
        assigner (Assigner): Strategy to match new messages to threads.
        update_strategy (UpdateStrategy): Policy for real-time updates (buffer vs immediate).

        formatter (Formatter): Strategy to prepare text for embedding.
    """
    messages: MessageStore
    threads: ThreadStore
    memberships: MembershipStore
    embeddings: EmbeddingStore

    embedder: Embedder
    reducer: Optional[Reducer]
    clusterer: Optional[Clusterer]

    thread_rep_computer: ThreadRepComputer
    assigner: Assigner
    update_strategy: UpdateStrategy

    formatter: Formatter

    # Configuration for vector spaces
    msg_space: SpaceId = "msg:full"  # Raw high-dim embeddings
    msg_cluster_space: SpaceId = "msg:cluster"  # Reduced/Projected embeddings
    thread_centroid_space: SpaceId = "thread:centroid"  # Thread representation vectors

    def run_batch(self) -> None:
        """
        Executes the 'Cold Start' pipeline on all currently stored messages.

        Steps:
            Format all messages.
            Embed texts (Bulk).
            Reduce dimensions (optional).
            Cluster (e.g. HDBSCAN).
            Generate Thread objects and Memberships.
            Compute Thread Centroids.
        """
        logger.info("run_batch: start")
        msgs = self.messages.all()
        mids = self.messages.ids()

        logger.info("run_batch: messages=%d", len(msgs))
        if not msgs:
            logger.info("run_batch: no messages, returning")
            return

        logger.info("run_batch: formatting messages")
        texts = self.formatter.format_all(msgs)
        logger.info("run_batch: formatted texts=%d", len(texts))

        logger.info("run_batch: embedding texts")
        X = self.embedder.embed_texts(texts)
        logger.info("run_batch: embeddings shape=%s dtype=%s", getattr(X, "shape", None), getattr(X, "dtype", None))
        self.embeddings.add(self.msg_space, mids, X)
        logger.info("run_batch: stored msg embeddings space=%s count=%d", self.msg_space, len(mids))

        if self.reducer:
            logger.info("run_batch: reducing embeddings")
            Xc = self.reducer.fit_transform(X)
            logger.info("run_batch: reduced shape=%s dtype=%s", getattr(Xc, "shape", None), getattr(Xc, "dtype", None))
        else:
            logger.info("run_batch: reducer=None, skipping reduction")
            Xc = X

        self.embeddings.add(self.msg_cluster_space, mids, Xc)
        logger.info("run_batch: stored cluster embeddings space=%s count=%d", self.msg_cluster_space, len(mids))

        if self.clusterer:
            logger.info("run_batch: clustering")
            labels, scores = self.clusterer.cluster(Xc)

            # Logging stats
            unique_labels = {lbl for sublist in labels for lbl in sublist}
            n_noise = sum(1 for sublist in labels if not sublist)
            n_clusters = len(unique_labels - {-1})

            logger.info("run_batch: clustering done clusters=%d noise_msgs=%d", n_clusters, n_noise)

            self._labels_to_threads(mids, labels, scores)
            logger.info("run_batch: threads after labels=%d memberships_total=%d",
                        len(self.threads.ids()), len(getattr(self.memberships, "_all", [])))
        else:
            logger.info("run_batch: clusterer=None, skipping clustering")

        # Compute centroids for all threads found
        tids = self.threads.ids()
        logger.info("run_batch: computing thread reps for tids=%d", len(tids))
        if tids:
            R = self.thread_rep_computer.compute_all(tids)
            logger.info("run_batch: thread reps shape=%s dtype=%s", getattr(R, "shape", None),
                        getattr(R, "dtype", None))
            self.embeddings.add(self.thread_centroid_space, tids, R)
            logger.info("run_batch: stored thread reps space=%s count=%d", self.thread_centroid_space, len(tids))
        else:
            logger.info("run_batch: no threads to compute reps for")

        logger.info("run_batch: done")

    def ingest_new_message(self, msg: Message) -> None:
        """
        Handles the arrival of a single real-time message.

        Steps:
        1. Persist message to store.
        2. Embed and Reduce (immediately).
        3. Delegate to UpdateStrategy (decides whether to assign now or buffer).
        """
        logger.info("ingest_new_message: id=%s user=%s ts=%s", msg.id, msg.user, msg.timestamp)
        self.messages.add([msg])
        logger.info("ingest_new_message: messages_total=%d", len(self.messages.all()))

        # embed immediately so it's available for the strategy
        msgs = self.messages.all()
        # Note: formatting entire history is O(N), consider optimizing to context window only
        text = self.formatter.format(len(msgs) - 1, msgs)
        v = self.embedder.embed_texts([text])[0]
        self.embeddings.add(self.msg_space, [msg.id], v[None, :])
        logger.info("ingest_new_message: stored msg embedding space=%s id=%s shape=%s",
                    self.msg_space, msg.id, getattr(v, "shape", None))

        if self.reducer:
            vc = self.reducer.transform(v[None, :])[0]
            self.embeddings.add(self.msg_cluster_space, [msg.id], vc[None, :])
            logger.info("ingest_new_message: stored reduced embedding space=%s id=%s shape=%s",
                        self.msg_cluster_space, msg.id, getattr(vc, "shape", None))
        else:
            logger.info("ingest_new_message: reducer=None, skipping reduction")

        logger.info("ingest_new_message: delegating to update_strategy.on_new_message(%s)", msg.id)
        self.update_strategy.on_new_message(msg.id)
        logger.info("ingest_new_message: done id=%s", msg.id)

    def _labels_to_threads(self, message_ids: List[str],
                           labels: List[List[int]],
                           scores: List[List[float]]) -> None:
        """
        Internal helper: Converts multi-label clustering output into
        Thread objects and Membership records.
        """
        logger.info("_labels_to_threads: start messages=%d", len(message_ids))

        # Flatten all labels to find unique clusters to create
        # We use a set comprehension over the nested lists
        all_labels_flat = {lab for sublist in labels for lab in sublist}

        label_to_tid = {}
        created_threads = 0

        for lab in all_labels_flat:
            lab = int(lab)
            if lab == -1:
                continue  # Skip noise if present

            tid = new_thread_id()
            self.threads.add([Thread(id=tid, title=f"Topic {lab}")])
            label_to_tid[lab] = tid
            created_threads += 1

        logger.info("_labels_to_threads: created_threads=%d", created_threads)

        ms = []
        kept = 0

        # Iterate through the triplet (MessageID, LabelList, ScoreList)
        for mid, label_list, score_list in zip(message_ids, labels, scores):

            # Handle multiple clusters for this single message
            for lab, score in zip(label_list, score_list):
                lab = int(lab)
                if lab == -1: continue

                ms.append(Membership(
                    message_id=mid,
                    thread_id=label_to_tid[lab],
                    score=score,
                    reason="cluster"
                ))
                kept += 1

        self.memberships.add(ms)
        logger.info("_labels_to_threads: done total_memberships=%d", kept)

    def apply_user_fix(
            self,
            message_id: MessageId,
            add_to: List[ThreadId],
            remove_from: List[ThreadId],
    ) -> None:
        """
        Applies a 'Human-in-the-Loop' correction.
            Soft-deletes (rejects) old memberships.
            Adds new 'user' origin memberships.
            Recalculates centroids for all affected threads.
        """
        logger.info("apply_user_fix: message_id=%s add_to=%d remove_from=%d",
                    message_id, len(add_to), len(remove_from))
        touched: Set[ThreadId] = set(add_to) | set(remove_from)
        logger.info("apply_user_fix: touched_threads=%d", len(touched))

        # reject memberships
        for tid in remove_from:
            logger.info("apply_user_fix: rejecting membership message=%s thread=%s", message_id, tid)
            self.memberships.reject(message_id, tid)

        # add new memberships (user-origin)
        new_ms = [
            Membership(
                message_id=message_id,
                thread_id=tid,
                score=1.0,
                origin="user",
                status="active",
                reason="user_fix",
            )
            for tid in add_to
        ]
        if new_ms:
            self.memberships.add(new_ms)
            logger.info("apply_user_fix: memberships_added=%d", len(new_ms))
        else:
            logger.info("apply_user_fix: no memberships added")

        # Update thread representations
        updated = 0
        for tid in touched:
            rep = self.thread_rep_computer.compute(tid)
            if rep.size != 0:
                self.embeddings.add(self.thread_centroid_space, [tid], rep[None, :])
                updated += 1
            else:
                logger.info("apply_user_fix: thread=%s rep empty, skipped centroid store", tid)

        logger.info("apply_user_fix: updated_centroids=%d done", updated)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Computes Cosine Similarity between two vectors."""
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    def semantic_search(
            self,
            query: str,
            top_threads: int = 5,
            top_messages_per_thread: int = 5,
            min_thread_sim: float = 0.25,
            min_msg_sim: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Performs a two-stage semantic search.
            Find threads similar to query (using Thread Centroids).
            Find messages within those threads similar to query.
        """
        logger.info("semantic_search: query_len=%d top_threads=%d top_msgs=%d min_thread_sim=%.3f min_msg_sim=%.3f",
                    len(query), top_threads, top_messages_per_thread, min_thread_sim, min_msg_sim)

        # embed query in SAME msg embedding space
        q_vec = self.embedder.embed_texts([query])[0]
        logger.debug("semantic_search: embedded query shape=%s", getattr(q_vec, "shape", None))

        # rank threads by similarity to centroid
        thread_scores: List[Tuple[ThreadId, float]] = []
        scanned_threads = 0
        missing_centroids = 0

        for t in self.threads.all():
            scanned_threads += 1
            if not self.embeddings.has(self.thread_centroid_space, t.id):
                missing_centroids += 1
                continue
            c = self.embeddings.get(self.thread_centroid_space, t.id)
            s = self._cosine(q_vec, c)
            if s >= min_thread_sim:
                thread_scores.append((t.id, s))

        logger.info("semantic_search: scanned_threads=%d missing_centroids=%d candidates=%d",
                    scanned_threads, missing_centroids, len(thread_scores))

        thread_scores.sort(key=lambda x: x[1], reverse=True)
        thread_scores = thread_scores[:top_threads]

        # within each thread, rank messages by similarity to query
        results: List[Dict[str, Any]] = []
        for tid, tscore in thread_scores:
            ms = self.memberships.for_thread(tid, status="active")
            mids = [m.message_id for m in ms]

            msg_hits: List[Tuple[MessageId, float]] = []
            missing_msg_vecs = 0
            for mid in mids:
                if not self.embeddings.has(self.msg_space, mid):
                    missing_msg_vecs += 1
                    continue
                v = self.embeddings.get(self.msg_space, mid)
                s = self._cosine(q_vec, v)
                if s >= min_msg_sim:
                    msg_hits.append((mid, s))

            msg_hits.sort(key=lambda x: x[1], reverse=True)
            msg_hits = msg_hits[:top_messages_per_thread]

            logger.info("semantic_search: thread=%s tscore=%.3f members=%d missing_msg_vecs=%d hits=%d",
                        tid, tscore, len(mids), missing_msg_vecs, len(msg_hits))

            # materialize message objects
            msg_payload = []
            for mid, s in msg_hits:
                m = self.messages.get(mid)
                msg_payload.append({
                    "message_id": mid,
                    "score": s,
                    "user": m.user,
                    "text": m.text,
                    "timestamp": m.timestamp,
                })

            t = self.threads.get(tid)
            results.append({
                "thread_id": tid,
                "thread_title": t.title,
                "thread_score": tscore,
                "messages": msg_payload
            })

        logger.info("semantic_search: returning_threads=%d", len(results))
        return results