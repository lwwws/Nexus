import os
import uuid
import json
import pickle
import tempfile
from queue import Queue, Empty
import threading
import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download

from core.models import Message, Thread, Membership
from core.stores import MessageStore, ThreadStore, MembershipStore, EmbeddingStore
from core.processor import ChatProcessor, new_thread_id
from core.strategies import (
    ContextWindowFormatter,
    MiniLMEmbedder,
    UMAPReducer,
    HDBSCANClusterer,
    CentroidThreadRepComputer,
    LlamaThreadLabeler,
    BufferedUpdateStrategy,
)
from utils import raw2df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus")

app = Flask(__name__)
app.config["MODEL_PATH"] = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["PERSIST_DIR"] = "./persisted_chats"
os.makedirs(app.config["PERSIST_DIR"], exist_ok=True)

# Download Model
if not os.path.exists(app.config["MODEL_PATH"]):
    model_path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        local_dir="./models"  # Downloads to a 'models' folder in your current directory
    )
    logger.info("Downloaded Llama model to %s", model_path)

LABELER = None
if os.path.exists(app.config["MODEL_PATH"]):
    try:
        LABELER = LlamaThreadLabeler(
            model_path=app.config["MODEL_PATH"],
            n_ctx=2048,
            max_msg_chars=300
        )
    except Exception as e:
        logger.warning("Failed to init LlamaThreadLabeler: %s", e)
        LABELER = None
else:
    logger.warning("Llama model not found at %s", app.config["MODEL_PATH"])

LOCK = threading.Lock()

# Defining embedder
EMBEDDER = MiniLMEmbedder("all-MiniLM-L6-v2")

# Max/Min limits for buffers
BUFFER_SIZE_MIN = 5
PENDING_SIZE_MIN = 5
BUFFER_SIZE_MAX = 100
PENDING_SIZE_MAX = 200

@dataclass
class ChatSession:
    chat_id: str
    name: str
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now())
    updated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now())

    messages: MessageStore = field(default_factory=MessageStore)
    threads: ThreadStore = field(default_factory=ThreadStore)
    memberships: MembershipStore = field(default_factory=MembershipStore)
    embeddings: EmbeddingStore = field(default_factory=EmbeddingStore)

    processor: Optional[ChatProcessor] = None
    is_ready: bool = False
    last_job_id: Optional[str] = None

    # Buffer settings (user-configurable)
    buffer_size_limit: int = 10
    pending_size_limit: int = 50

    # The Queue Infrastructure for new messages
    ingest_queue: Queue = field(default_factory=Queue)
    worker_running: bool = False
    worker_lock: threading.Lock = field(default_factory=threading.Lock)  # Protects the startup check


@dataclass
class JobState:
    job_id: str
    chat_id: str
    status: str = "queued"     # queued|running|done|error
    stage: str = "Queued"
    progress: int = 0          # 0..100
    detail: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


CHATS: Dict[str, ChatSession] = {}
JOBS: Dict[str, JobState] = {}

def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _require_chat(chat_id: str) -> ChatSession:
    with LOCK:
        c = CHATS.get(chat_id)
    if not c:
        raise KeyError("Chat not found")
    return c


def _try_parse_whatsapp(filepath: str):
    """
    Parse WhatsApp chat export file with auto-detection of datetime format.
    """
    return raw2df(filepath, "auto")


def _fallback_summary_for_thread(chat: ChatSession, tid: str) -> str:
    msgs = []
    for mem in chat.memberships.for_thread(tid, status="active"):
        try:
            m = chat.messages.get(mem.message_id)
            if m.user == "group_notification":
                continue
            msgs.append(m)
        except Exception:
            continue
    msgs.sort(key=lambda m: m.timestamp)
    snippets = []
    for m in msgs[:3]:
        txt = (m.text or "").strip().replace("\n", " ")
        if len(txt) > 140:
            txt = txt[:140].rstrip() + "…"
        if txt:
            snippets.append(txt)
    return " • ".join(snippets) if snippets else "Summary not available."


class ProgressEmbedder:
    """Wraps the embedder to update Job UI during the embedding phase."""
    def __init__(self, base_embedder: MiniLMEmbedder, job: JobState,
                 start_pct: int, end_pct: int, batch_size: int = 64):
        self.model = base_embedder.model
        self.job = job
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, 384), dtype=np.float32)

        X_parts = []
        done = 0
        for i in range(0, n, self.batch_size):
            batch = texts[i:i + self.batch_size]
            xb = self.model.encode(batch, show_progress_bar=False)
            xb = np.asarray(xb, dtype=np.float32)
            X_parts.append(xb)

            done += len(batch)
            pct = self.start_pct + int((self.end_pct - self.start_pct) * (done / max(1, n)))
            with LOCK:
                self.job.progress = max(self.job.progress, pct)
                self.job.detail = f"Embedding messages: {done}/{n}"

        return np.vstack(X_parts)



def _build_processor(chat: ChatSession, job: Optional[JobState]) -> ChatProcessor:
    if job is not None:
        # Wrap embedder to provide visual progress
        embedder = ProgressEmbedder(EMBEDDER, job, start_pct=28, end_pct=68, batch_size=64)
    else:
        embedder = EMBEDDER

    reducer = UMAPReducer(n_neighbors=15, n_components=10, min_dist=0.0)
    clusterer = HDBSCANClusterer(min_cluster_size=15, min_samples=5)

    # Incremental Clusterer
    incremental_clusterer = HDBSCANClusterer(min_cluster_size=10, min_samples=4)

    thread_rep = CentroidThreadRepComputer(memberships=chat.memberships, embeddings=chat.embeddings)
    update_strategy = BufferedUpdateStrategy(
        embeddings=chat.embeddings,
        threads=chat.threads,
        memberships=chat.memberships,
        clusterer=incremental_clusterer,
        global_min_threshold=0.65,
        percentile_threshold=25,
        buffer_size_limit=chat.buffer_size_limit,
        pending_size_limit=chat.pending_size_limit,
        min_delta=0.08,
        anchor_size=50,
    )

    formatter = ContextWindowFormatter(window_back=2, window_fwd=1, time_threshold_minutes=10, repeat_center=2)

    processor = ChatProcessor(
        messages=chat.messages,
        threads=chat.threads,
        memberships=chat.memberships,
        embeddings=chat.embeddings,
        embedder=embedder,
        reducer=reducer,
        clusterer=clusterer,
        labeler=LABELER,
        thread_rep_computer=thread_rep,
        update_strategy=update_strategy,
        formatter=formatter,
    )
    return processor

def _chat_dir(chat_id: str) -> str:
    return os.path.join(app.config["PERSIST_DIR"], chat_id)

def _save_chat(chat: ChatSession) -> None:
    """
    Persist the minimal state needed to restore the UI instantly:
    - messages (ordered)
    - threads
    - memberships (including rejected)
    - embeddings for spaces used at runtime (msg:full, msg:cluster, thread:centroid)
    """
    d = _chat_dir(chat.chat_id)
    os.makedirs(d, exist_ok=True)

    meta = {
        "chat_id": chat.chat_id,
        "name": chat.name,
        "created_at": chat.created_at.isoformat(timespec="seconds"),
        "updated_at": chat.updated_at.isoformat(timespec="seconds"),
        "is_ready": bool(chat.is_ready),
        "last_job_id": chat.last_job_id,
    }

    # Messages in timeline order (MessageStore.order is your ground truth) :contentReference[oaicite:2]{index=2}
    messages = []
    for mid in chat.messages.ids():
        m = chat.messages.get(mid)
        messages.append({
            "id": m.id,
            "timestamp": m.timestamp.isoformat(timespec="seconds"),
            "user": m.user,
            "text": m.text,
            "metadata": m.metadata,
        })

    # Threads (ThreadStore is dict-based) :contentReference[oaicite:3]{index=3}
    threads = []
    for t in chat.threads.all():
        threads.append({
            "id": t.id,
            "title": t.title,
            "summary": t.summary,
            "created_at": t.created_at.isoformat(timespec="seconds"),
            "updated_at": t.updated_at.isoformat(timespec="seconds"),
            "metadata": t.metadata,
        })

    # Memberships: include rejected too (store has _all internally) :contentReference[oaicite:4]{index=4}
    memberships = []
    for m in getattr(chat.memberships, "_all", []):
        memberships.append({
            "message_id": m.message_id,
            "thread_id": m.thread_id,
            "score": float(m.score),
            "centroid_similarity": float(m.centroid_similarity),
            "origin": m.origin,
            "status": m.status,
            "reason": m.reason,
            "created_at": m.created_at.isoformat(timespec="seconds"),
        })

    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(d, "messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)

    with open(os.path.join(d, "threads.json"), "w", encoding="utf-8") as f:
        json.dump(threads, f, ensure_ascii=False)

    with open(os.path.join(d, "memberships.json"), "w", encoding="utf-8") as f:
        json.dump(memberships, f, ensure_ascii=False)

    # Embeddings: persist per space as (ids, matrix) and restore via EmbeddingStore.add :contentReference[oaicite:5]{index=5}
    spaces = ["msg:full", "msg:cluster", "thread:centroid"]
    npz_path = os.path.join(d, "embeddings.npz")
    arrays = {}
    for space in spaces:
        ids, X = chat.embeddings.get_matrix(space)
        arrays[f"{space}__ids"] = np.array(ids, dtype=object)
        arrays[f"{space}__X"] = X.astype(np.float32)
    np.savez_compressed(npz_path, **arrays)

    # Save the fitted UMAP reducer if it exists
    if chat.processor and chat.processor.reducer:
        reducer_path = os.path.join(d, "reducer.pkl")
        try:
            with open(reducer_path, "wb") as f:
                pickle.dump(chat.processor.reducer, f)
            logger.info(f"Saved reducer for chat {chat.chat_id}")
        except Exception as e:
            logger.warning(f"Failed to save reducer for chat {chat.chat_id}: {e}")


def _load_all_chats_from_disk() -> None:
    """
    Scan PERSIST_DIR, rebuild ChatSession objects into CHATS,
    and rebuild chat.processor so semantic_search/apply_user_fix work without re-running run_batch.
    """
    root = app.config["PERSIST_DIR"]
    if not os.path.isdir(root):
        return

    for chat_id in os.listdir(root):
        d = os.path.join(root, chat_id)
        if not os.path.isdir(d):
            continue

        meta_path = os.path.join(d, "meta.json")
        msgs_path = os.path.join(d, "messages.json")
        threads_path = os.path.join(d, "threads.json")
        mems_path = os.path.join(d, "memberships.json")
        emb_path = os.path.join(d, "embeddings.npz")

        if not (os.path.exists(meta_path) and os.path.exists(msgs_path) and os.path.exists(threads_path) and os.path.exists(mems_path)):
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        chat = ChatSession(
            chat_id=meta["chat_id"],
            name=meta.get("name", meta["chat_id"]),
        )
        chat.created_at = dt.datetime.fromisoformat(meta["created_at"])
        chat.updated_at = dt.datetime.fromisoformat(meta["updated_at"])
        chat.is_ready = bool(meta.get("is_ready", True))
        chat.last_job_id = meta.get("last_job_id")

        # Restore messages in order
        with open(msgs_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        msgs = []
        for x in arr:
            msgs.append(Message(
                id=x["id"],
                timestamp=dt.datetime.fromisoformat(x["timestamp"]),
                user=x["user"],
                text=x["text"],
                metadata=x.get("metadata", {}),
            ))
        chat.messages.add(msgs)

        # Restore threads
        with open(threads_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        ths = []
        for x in arr:
            t = Thread(
                id=x["id"],
                title=x.get("title", ""),
                summary=x.get("summary", ""),
                metadata=x.get("metadata", {}),
            )
            t.created_at = dt.datetime.fromisoformat(x["created_at"])
            t.updated_at = dt.datetime.fromisoformat(x["updated_at"])
            ths.append(t)
        chat.threads.add(ths)

        # Restore memberships (active + rejected)
        with open(mems_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        ms = []
        for x in arr:
            m = Membership(
                message_id=x["message_id"],
                thread_id=x["thread_id"],
                score=float(x["score"]),
                centroid_similarity=float(x.get("centroid_similarity", -1.0)),
                origin=x.get("origin", "auto"),
                status=x.get("status", "active"),
                reason=x.get("reason", "unknown"),
            )
            # created_at exists in the dataclass, set it if present :contentReference[oaicite:6]{index=6}
            if x.get("created_at"):
                m.created_at = dt.datetime.fromisoformat(x["created_at"])
            ms.append(m)
        chat.memberships.add(ms)

        # Restore embeddings if present
        if os.path.exists(emb_path):
            npz = np.load(emb_path, allow_pickle=True)
            for space in ["msg:full", "msg:cluster", "thread:centroid"]:
                ids = list(npz[f"{space}__ids"])
                X = npz[f"{space}__X"]
                if len(ids) and X.size:
                    chat.embeddings.add(space, ids, X)

        # Rebuild processor (no job -> no progress wrapper) :contentReference[oaicite:7]{index=7}
        chat.processor = _build_processor(chat, job=None)

        # Restore the fitted UMAP reducer if it was saved
        reducer_path = os.path.join(d, "reducer.pkl")
        if os.path.exists(reducer_path) and chat.processor.reducer:
            import pickle
            try:
                with open(reducer_path, "rb") as f:
                    chat.processor.reducer = pickle.load(f)
                logger.info(f"Loaded saved reducer for chat {chat_id}")
            except Exception as e:
                logger.warning(f"Failed to load reducer for chat {chat_id}, will refit: {e}")
                # Fallback: refit if loading fails
                try:
                    ids, X = chat.embeddings.get_matrix("msg:full")
                    if len(ids) > 0 and X.size > 0:
                        logger.info(f"Refitting reducer for chat {chat_id} with {len(ids)} existing embeddings")
                        chat.processor.reducer.fit_transform(X)
                        logger.info(f"Reducer fitted successfully for chat {chat_id}")
                except Exception as e2:
                    logger.warning(f"Failed to refit reducer for chat {chat_id}: {e2}")
        elif chat.processor.reducer:
            # No saved reducer found, refit on existing embeddings
            try:
                ids, X = chat.embeddings.get_matrix("msg:full")
                if len(ids) > 0 and X.size > 0:
                    logger.info(f"No saved reducer found. Refitting reducer for chat {chat_id} with {len(ids)} existing embeddings")
                    chat.processor.reducer.fit_transform(X)
                    logger.info(f"Reducer fitted successfully for chat {chat_id}")
            except Exception as e:
                logger.warning(f"Failed to refit reducer for chat {chat_id}: {e}")

        with LOCK:
            CHATS[chat.chat_id] = chat

_load_all_chats_from_disk()

def _run_batch_with_progress(chat: ChatSession, job: JobState) -> None:
    """
    Delegates pipeline execution to the Processor, bridging updates to the JobState.
    """
    processor = chat.processor
    assert processor is not None

    def on_progress(stage: str, percent: int):
        with LOCK:
            job.stage = stage
            job.progress = max(job.progress, percent)
            job.detail = f"({percent}%)"

    # 4. Run the Processor (This handles Formatting -> Embedding -> Clustering -> Labeling)
    processor.run_batch(progress_callback=on_progress)

    with LOCK:
        job.progress = 100
        job.stage = "Done"
        job.detail = f"Processed {len(chat.threads.all())} topics."


def _run_job_process_chat_with_cleanup(job: JobState, filepath: str) -> None:
    """Wrapper that ensures temporary file cleanup after processing."""
    try:
        _run_job_process_chat(job, filepath)
    finally:
        # Clean up temporary file
        try:
            os.unlink(filepath)
        except Exception:
            pass


def _run_job_process_chat(job: JobState, filepath: str) -> None:
    try:
        with LOCK:
            job.status = "running"
            job.started_at = _now_iso()
            job.stage = "Parsing WhatsApp export"
            job.progress = 5

        df = _try_parse_whatsapp(filepath)

        with LOCK:
            job.stage = "Building chat session"
            job.progress = 12
            job.detail = ""

        chat = _require_chat(job.chat_id)
        chat.processor = _build_processor(chat, job)

        msgs: List[Message] = []
        for i, row in df.iterrows():
            mid = f"{chat.chat_id}_m_{i}"
            msgs.append(Message(
                id=mid,
                timestamp=row["date_time"].to_pydatetime(),
                user=str(row["user"]),
                text=str(row["message"]),
            ))
        chat.messages.add(msgs)

        with LOCK:
            job.progress = 20
            job.stage = "Embedding + clustering + labeling topics"
            job.detail = f"Loaded {len(msgs)} messages"

        _run_batch_with_progress(chat, job)

        chat.is_ready = True
        chat.updated_at = dt.datetime.now()
        _save_chat(chat)

        with LOCK:
            job.status = "done"
            job.stage = "Done"
            job.progress = 100
            job.finished_at = _now_iso()

    except Exception as e:
        logger.exception("Job failed")
        with LOCK:
            job.status = "error"
            job.stage = "Error"
            job.error = str(e)
            job.finished_at = _now_iso()


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/jobs/<job_id>", methods=["GET"])
def api_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({
            "job_id": job.job_id,
            "chat_id": job.chat_id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "detail": job.detail,
            "error": job.error,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        })


@app.route("/api/chats", methods=["GET"])
def api_chats():
    with LOCK:
        chats = list(CHATS.values())
    out = []
    for c in chats:
        out.append({
            "chat_id": c.chat_id,
            "name": c.name,
            "is_ready": c.is_ready,
            "created_at": c.created_at.isoformat(timespec="seconds"),
            "updated_at": c.updated_at.isoformat(timespec="seconds"),
            "message_count": len(c.messages.all()),
            "topic_count": len(c.threads.all()),
            "last_job_id": c.last_job_id,
        })
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    return jsonify(out)


@app.route("/api/chats/upload", methods=["POST"])
def api_upload_chat():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(f.filename)
    chat_id = f"chat_{uuid.uuid4().hex[:10]}"
    job_id = f"job_{uuid.uuid4().hex[:10]}"

    chat_name = os.path.splitext(filename)[0] or chat_id

    with LOCK:
        CHATS[chat_id] = ChatSession(chat_id=chat_id, name=chat_name, last_job_id=job_id)
        JOBS[job_id] = JobState(job_id=job_id, chat_id=chat_id, stage="Queued", progress=0)

    # Create temp file; close handle so other libs can open by path on Windows too
    suffix = os.path.splitext(filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    f.save(tmp_path)

    t = threading.Thread(
        target=_run_job_process_chat_with_cleanup,
        args=(JOBS[job_id], tmp_path),
        daemon=True,
    )
    t.start()

    return jsonify({"chat_id": chat_id, "job_id": job_id})


@app.route("/api/chats/<chat_id>/users", methods=["GET"])
def api_chat_users(chat_id: str):
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    users = sorted({m.user for m in chat.messages.all() if m.user != "group_notification"})
    return jsonify(users)


@app.route("/api/chats/<chat_id>/config", methods=["GET"])
def api_chat_config(chat_id: str):
    """Get chat configuration including buffer constraints."""
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    return jsonify({
        "buffer_size_min": BUFFER_SIZE_MIN,
        "pending_size_min": PENDING_SIZE_MIN,
        "buffer_size_limit": chat.buffer_size_limit,
        "pending_size_limit": chat.pending_size_limit,
        "buffer_size_max": BUFFER_SIZE_MAX,
        "pending_size_max": PENDING_SIZE_MAX,
    })


@app.route("/api/chats/<chat_id>/config", methods=["POST"])
def api_update_chat_config(chat_id: str):
    """Update chat buffer configuration."""
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    data = request.json or {}

    updated_buffer = None
    updated_pending = None

    # Validate and update buffer_size_limit
    if "buffer_size_limit" in data:
        buffer_size = int(data["buffer_size_limit"])
        if buffer_size < BUFFER_SIZE_MIN:
            return jsonify({"error": f"buffer_size_limit must be >= {BUFFER_SIZE_MIN}"}), 400
        if buffer_size > BUFFER_SIZE_MAX:
            return jsonify({"error": f"buffer_size_limit must be <= {BUFFER_SIZE_MAX}"}), 400
        chat.buffer_size_limit = buffer_size
        updated_buffer = buffer_size

    # Validate and update pending_size_limit
    if "pending_size_limit" in data:
        pending_size = int(data["pending_size_limit"])
        if pending_size < PENDING_SIZE_MIN:
            return jsonify({"error": f"pending_size_limit must be >= {PENDING_SIZE_MIN}"}), 400
        if pending_size > PENDING_SIZE_MAX:
            return jsonify({"error": f"pending_size_limit must be <= {PENDING_SIZE_MAX}"}), 400
        chat.pending_size_limit = pending_size
        updated_pending = pending_size

    # Update the strategy directly (no need to rebuild entire processor)
    if chat.processor and chat.processor.update_strategy:
        if isinstance(chat.processor.update_strategy, BufferedUpdateStrategy):
            chat.processor.update_strategy.update_buffer_limits(
                buffer_size_limit=updated_buffer,
                pending_size_limit=updated_pending
            )
            logging.info(f"Chat processor updated buffer: buffer_limit={chat.processor.update_strategy.buffer_limit}, pending_limit={chat.processor.update_strategy.pending_limit}")

    chat.updated_at = dt.datetime.now()

    return jsonify({
        "ok": True,
        "buffer_size_limit": chat.buffer_size_limit,
        "pending_size_limit": chat.pending_size_limit,
    })


@app.route("/api/chats/<chat_id>/topics", methods=["GET"])
def api_chat_topics(chat_id: str):
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    if not chat.is_ready:
        return jsonify([])

    topics = []
    for t in chat.threads.all():
        cnt = len(chat.memberships.for_thread(t.id, status="active"))
        # IMPORTANT: now we only fallback if it's actually empty
        summary = (t.summary or "").strip()
        if not summary:
            summary = _fallback_summary_for_thread(chat, t.id)

        topics.append({
            "id": t.id,
            "title": t.title or "Untitled topic",
            "summary": summary,
            "message_count": cnt,
        })

    topics.sort(key=lambda x: x["message_count"], reverse=True)
    return jsonify(topics)


@app.route("/api/chats/<chat_id>/history", methods=["GET"])
def api_chat_history(chat_id: str):
    """
    WhatsApp-like paging:
      offset=0, limit=80 => LAST 80 messages
      offset=80 => previous 80 messages before those, etc.
    """
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    limit = int(request.args.get("limit", 80))
    offset = int(request.args.get("offset", 0))

    all_ids = chat.messages.ids()
    total = len(all_ids)

    end = max(0, total - offset)
    start = max(0, end - limit)
    slice_ids = all_ids[start:end]

    # return chronological within slice
    out = []
    for mid in slice_ids:
        m = chat.messages.get(mid)

        # best thread tag (for timeline pills)
        best_tid = None
        best_title = None
        ms = chat.memberships.for_message(mid, status="active")
        if ms:
            best = max(ms, key=lambda x: x.score)
            best_tid = best.thread_id
            if chat.threads.has(best_tid):
                best_title = chat.threads.get(best_tid).title

        processed = chat.embeddings.has("msg:full", mid)

        out.append({
            "id": m.id,
            "user": m.user,
            "text": m.text,
            "timestamp": m.timestamp.isoformat(timespec="seconds"),
            "is_system": (m.user == "group_notification"),
            "thread_id": best_tid,
            "thread_title": best_title,
            "processed": processed,
        })

    has_more = start > 0
    return jsonify({
        "messages": out,
        "total": total,
        "has_more": has_more,
        "next_offset": offset + len(slice_ids),
    })


@app.route("/api/chats/<chat_id>/focus/<topic_id>", methods=["GET"])
def api_focus_view(chat_id: str, topic_id: str):
    """
    Focus mode:
      - returns focus messages in TIMELINE order
      - includes each message's timeline index to allow gap detection client-side
    """
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    if not chat.is_ready or not chat.processor or not chat.threads.has(topic_id):
        return jsonify({"topic": None, "focus": [], "total_messages": len(chat.messages.ids()), "has_gaps": False})

    all_ids = chat.messages.ids()
    index_map = {mid: i for i, mid in enumerate(all_ids)}

    focus_msgs = chat.processor.get_messages_for_thread(topic_id)
    # sort in timeline order
    focus_msgs.sort(key=lambda m: index_map.get(m.id, 10**18))

    focus_payload = []
    for m in focus_msgs:
        focus_payload.append({
            "id": m.id,
            "user": m.user,
            "text": m.text,
            "timestamp": m.timestamp.isoformat(timespec="seconds"),
            "is_system": (m.user == "group_notification"),
            "idx": int(index_map.get(m.id, -1)),
        })

    has_gaps = False
    for a, b in zip(focus_payload, focus_payload[1:]):
        if a["idx"] != -1 and b["idx"] != -1 and b["idx"] - a["idx"] > 1:
            has_gaps = True
            break

    t = chat.threads.get(topic_id)
    summary = (t.summary or "").strip() or _fallback_summary_for_thread(chat, topic_id)

    return jsonify({
        "topic": {
            "id": topic_id,
            "title": t.title or "Untitled topic",
            "summary": summary,
        },
        "focus": focus_payload,
        "total_messages": len(all_ids),
        "has_gaps": has_gaps,
    })


@app.route("/api/chats/<chat_id>/between", methods=["GET"])
def api_between(chat_id: str):
    """
    Return messages strictly between from_id and to_id in TIMELINE order.
    Used by Focus Mode to "uncollapse messages in between two focus messages".
    """
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    from_id = request.args.get("from", "")
    to_id = request.args.get("to", "")
    if not from_id or not to_id:
        return jsonify({"error": "from and to required"}), 400

    all_ids = chat.messages.ids()
    try:
        i1 = all_ids.index(from_id)
        i2 = all_ids.index(to_id)
    except ValueError:
        return jsonify({"error": "Message not found"}), 404

    if i2 <= i1 + 1:
        return jsonify({"messages": []})

    mids = all_ids[i1 + 1:i2]
    out = []
    for mid in mids:
        m = chat.messages.get(mid)
        out.append({
            "id": m.id,
            "user": m.user,
            "text": m.text,
            "timestamp": m.timestamp.isoformat(timespec="seconds"),
            "is_system": (m.user == "group_notification"),
        })
    return jsonify({"messages": out, "count": len(out)})

def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _clampint(x, lo, hi, default) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


@app.route("/api/chats/<chat_id>/search", methods=["GET"])
def api_search_chat(chat_id: str):
    """
    Performs semantic search using the Processor.
    Query param: ?q=search_term
    """
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    if not chat.is_ready or not chat.processor:
        return jsonify({"error": "Chat not processed yet"}), 400

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query param 'q'"}), 400

    try:
        # Delegate to the processor's robust logic
        min_thread_sim = _clamp01(request.args.get("min_thread_sim", 0.10))
        min_msg_sim = _clamp01(request.args.get("min_msg_sim", 0.25))
        top_threads = _clampint(request.args.get("top_threads", 5), 1, 50, 5)
        top_msgs = _clampint(request.args.get("top_messages_per_thread", 5), 1, 50, 5)


        results = chat.processor.semantic_search(
            query,
            top_threads=top_threads,
            top_messages_per_thread=top_msgs,
            min_thread_sim=min_thread_sim,
            min_msg_sim=min_msg_sim,
        )
        return jsonify(results)
    except Exception as e:
        logger.exception("Search failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chats/<chat_id>/user_fix", methods=["POST"])
def api_user_fix(chat_id: str):
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    if not chat.is_ready or not chat.processor:
        return jsonify({"error": "Chat not processed yet"}), 400

    data = request.json or {}
    message_id = (data.get("message_id") or "").strip()
    add_to = data.get("add_to") or []
    remove_from = data.get("remove_from") or []

    if not message_id:
        return jsonify({"error": "message_id required"}), 400
    if not isinstance(add_to, list) or not isinstance(remove_from, list):
        return jsonify({"error": "add_to/remove_from must be lists"}), 400

    # Optional: validate threads exist
    for tid in add_to + remove_from:
        if tid and not chat.threads.has(tid):
            return jsonify({"error": f"Unknown thread_id: {tid}"}), 400

    try:
        chat.processor.apply_user_fix(message_id=message_id, add_to=add_to, remove_from=remove_from)
        chat.updated_at = dt.datetime.now()
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("user_fix failed")
        return jsonify({"error": str(e)}), 500



def _ingest_worker(chat_id: str):
    """
    Dedicated worker thread for a specific chat.
    Processes messages one by one from the queue.
    """
    logger.info(f"Worker started for chat {chat_id}")

    try:
        while True:
            # Check if chat still exists (Graceful shutdown if chat deleted)
            with LOCK:
                chat = CHATS.get(chat_id)

            if not chat:
                logger.info(f"Chat {chat_id} not found, stopping worker.")
                return

            # Get next message (Blocking with timeout for liveness check)
            try:
                # Wait up to k seconds for a message
                msg = chat.ingest_queue.get(timeout=1.0)
            except Empty:
                # No messages? Loop back to check if chat still exists
                continue

            # Process the message
            try:
                # No locks needed here! The Queue implicitly serializes access.
                # Only this thread ever calls ingest_new_message for this chat.
                if chat.processor:
                    chat.processor.ingest_new_message(msg)
                    chat.updated_at = dt.datetime.now()
                    logger.info(f"Ingested message {msg.id}")
            except Exception:
                logger.exception(f"Worker failed to ingest {msg.id}")
            finally:
                chat.ingest_queue.task_done()
    finally:
        # allow restart if the thread ever exits
        with LOCK:
            chat = CHATS.get(chat_id)
        if chat:
            with chat.worker_lock:
                chat.worker_running = False

@app.route("/api/chats/<chat_id>/message", methods=["POST"])
def api_post_message(chat_id: str):
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    if not chat.is_ready or not chat.processor:
        return jsonify({"error": "Chat not ready yet"}), 409

    data = request.json or {}
    user = (data.get("user") or "").strip()
    text = (data.get("text") or "").strip()

    if not user or not text:
        return jsonify({"error": "user and text required"}), 400

    mid = f"{chat.chat_id}_new_{uuid.uuid4().hex[:10]}"
    msg = Message(
        id=mid,
        timestamp=dt.datetime.now(),
        user=user,
        text=text,
    )

    # Save to Store
    chat.messages.add([msg])
    chat.updated_at = dt.datetime.now()

    # Enqueue
    chat.ingest_queue.put(msg)

    # Ensure Worker is Running (Thread-Safe lazy start)
    if not chat.worker_running:
        with chat.worker_lock:
            # Double-check inside lock to prevent race condition
            if not chat.worker_running:
                t = threading.Thread(target=_ingest_worker, args=(chat_id,), daemon=True)
                chat.worker_running = True
                t.start()
                logger.info(f"Spawned new worker for chat {chat_id}")

    return jsonify({
        "status": "pending",
        "message_id": mid,
        "info": "Queued for processing."
    }), 202

@app.route("/api/chats/<chat_id>/flush", methods=["POST"])
def api_flush_chat(chat_id: str):
    """Force flush the buffered messages immediately."""
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    if not chat.is_ready or not chat.processor:
        return jsonify({"error": "Chat not ready yet"}), 409

    try:
        # Check if buffer has enough messages for clustering
        strategy = chat.processor.update_strategy
        if not isinstance(strategy, BufferedUpdateStrategy):
            return jsonify({"error": "Chat does not use buffered strategy"}), 400

        buffer_count = len(strategy._buffer)
        pending_count = len(strategy._pending)
        total_count = buffer_count + pending_count
        total_needed = BUFFER_SIZE_MIN + PENDING_SIZE_MIN

        # Minimum messages needed for HDBSCAN (min_cluster_size=7, min_samples=2)
        if total_count < total_needed:
            return jsonify({
                "error": f"Not enough messages to cluster. Need at least {total_needed}, have {total_count}.",
                "buffer": buffer_count,
                "pending": pending_count,
                "total": total_count
            }), 400

        # Count threads before flush
        threads_before = len(chat.threads.all())

        # Trigger flush
        result = strategy.flush()

        # Analyze the flush result
        if result.action == "flushed":
            # Count new threads created
            threads_after = len(chat.threads.all())
            new_threads = threads_after - threads_before

            # Analyze messages: count noise vs assigned
            messages_assigned = 0
            messages_noise = 0

            for labels_list in result.flush_labels:
                if labels_list:  # Has cluster assignment
                    messages_assigned += 1
                else:  # Noise (empty list)
                    messages_noise += 1

            # Get current pending count after flush
            pending_after = len(strategy._pending)

            return jsonify({
                "status": "success",
                "action": result.action,
                "buffer_count": buffer_count,
                "pending_count": pending_count,
                "total_processed": total_count,
                "new_threads": new_threads,
                "messages_assigned": messages_assigned,
                "messages_noise": messages_noise,
                "pending_after": pending_after
            }), 200
        else:
            # Flush didn't actually cluster (buffered action)
            return jsonify({
                "status": "success",
                "action": result.action,
                "buffer_count": buffer_count,
                "pending_count": pending_count,
                "total_processed": 0,
                "new_threads": 0,
                "messages_assigned": 0,
                "messages_noise": 0,
                "pending_after": pending_count
            }), 200

    except Exception as e:
        logger.error(f"Error flushing chat {chat_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats/<chat_id>/recluster", methods=["POST"])
def api_recluster_chat(chat_id: str):
    """Rerun batch clustering from scratch (run_batch)."""
    try:
        chat = _require_chat(chat_id)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404

    if not chat.processor:
        return jsonify({"error": "Chat processor not initialized"}), 409

    # Create a new job for this operation
    job_id = f"recluster_{uuid.uuid4().hex[:8]}"
    job = JobState(job_id=job_id, chat_id=chat_id)
    job.status = "running"
    job.stage = "Reclustering"
    job.started_at = dt.datetime.now().isoformat()

    with LOCK:
        JOBS[job_id] = job

    def _recluster_task():
        try:
            logger.info(f"Starting recluster for chat {chat_id}")

            # Save the old thread_centroid_space name before clearing
            old_space = chat.processor.thread_centroid_space if chat.processor else "thread:centroid"

            # Clear existing threads and memberships
            chat.threads.clear()
            chat.memberships.clear()

            # Clear thread embeddings
            chat.embeddings.clear_space(old_space)

            def progress_cb(stage: str, pct: int):
                job.stage = stage
                job.progress = pct
                logger.info(f"Recluster progress: {stage} {pct}%")

            # Rebuild processor with fresh job for progress tracking
            chat.processor = _build_processor(chat, job)

            # Clear buffer and pending from strategy after rebuild
            if isinstance(chat.processor.update_strategy, BufferedUpdateStrategy):
                chat.processor.update_strategy._buffer = []
                chat.processor.update_strategy._pending = []

            # Run batch clustering
            chat.processor.run_batch(progress_callback=progress_cb)

            job.status = "done"
            job.progress = 100
            job.stage = "Complete"

            # Save the reclustered chat
            _save_chat(chat)

            logger.info(f"Recluster complete for chat {chat_id}")

        except Exception as e:
            logger.error(f"Error reclustering chat {chat_id}: {e}", exc_info=True)
            job.status = "error"
            job.error = str(e)

    # Run in background thread
    threading.Thread(target=_recluster_task, daemon=True).start()

    return jsonify({
        "status": "started",
        "job_id": job_id,
        "message": "Reclustering started in background"
    }), 202

if __name__ == "__main__":
    app.run(debug=True, port=5000)
