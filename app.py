import os
import uuid
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
    NoOpUpdateStrategy, NoOpAssigner,
)
from utils import raw2df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"
app.config["MODEL_PATH"] = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

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

    thread_rep = CentroidThreadRepComputer(memberships=chat.memberships, embeddings=chat.embeddings)
    update_strategy = NoOpUpdateStrategy()
    assigner = NoOpAssigner()
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
        assigner=assigner,
        update_strategy=update_strategy,
        formatter=formatter,
    )
    return processor


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

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{chat_id}__{filename}")
    f.save(filepath)

    chat_name = os.path.splitext(filename)[0] or chat_id

    with LOCK:
        CHATS[chat_id] = ChatSession(chat_id=chat_id, name=chat_name, last_job_id=job_id)
        JOBS[job_id] = JobState(job_id=job_id, chat_id=chat_id, stage="Queued", progress=0)

    t = threading.Thread(target=_run_job_process_chat, args=(JOBS[job_id], filepath), daemon=True)
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

        out.append({
            "id": m.id,
            "user": m.user,
            "text": m.text,
            "timestamp": m.timestamp.isoformat(timespec="seconds"),
            "is_system": (m.user == "group_notification"),
            "thread_id": best_tid,
            "thread_title": best_title,
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
        results = chat.processor.semantic_search(
            query,
            top_threads=5,
            top_messages_per_thread=5,
            min_thread_sim = 0.25,
            min_msg_sim = 0.25
        )
        return jsonify(results)
    except Exception as e:
        logger.exception("Search failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chats/<chat_id>/message", methods=["POST"])
def api_post_message(chat_id: str):
    """
    Append message, embed+reduce immediately via processor.ingest_new_message,
    then assign to best topic by centroid similarity or create new topic.
    # TODO: connect properly to processor.ingest_new_message instead, which delegates it to UpdateStrategy
    """
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

    # ingest_new_message requires update_strategy; we set NoOpUpdateStrategy in processor build.
    chat.processor.ingest_new_message(msg)

    # Assign: cosine similarity to thread centroids
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        na = float(np.linalg.norm(a) + 1e-12)
        nb = float(np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b) / (na * nb))

    proc = chat.processor
    best_tid = None
    best_score = -1.0

    if chat.embeddings.has(proc.msg_space, mid):
        v = chat.embeddings.get(proc.msg_space, mid)

        for t in chat.threads.all():
            if chat.embeddings.has(proc.thread_centroid_space, t.id):
                c = chat.embeddings.get(proc.thread_centroid_space, t.id)
                s = cosine(v, c)
                if s > best_score:
                    best_score = s
                    best_tid = t.id

    created_new = False
    threshold = 0.35

    if best_tid is None or best_score < threshold:
        # Create new topic
        tid = new_thread_id()
        t = Thread(id=tid, title="New topic", summary="")
        chat.threads.add([t])
        created_new = True

        chat.memberships.add([Membership(
            message_id=mid,
            thread_id=tid,
            score=1.0,
            reason="new_topic",
        )])

        # Update centroid
        rep = proc.thread_rep_computer.compute(tid)
        if rep.size:
            chat.embeddings.add(proc.thread_centroid_space, [tid], rep[None, :])

        # Fallback summary so UI has something immediately
        t = chat.threads.get(tid)
        if not (t.summary or "").strip():
            t.summary = _fallback_summary_for_thread(chat, tid)
        t.updated_at = dt.datetime.now()
        chat.threads.add([t])

        assigned = {"thread_id": tid, "score": 1.0}
    else:
        tid = best_tid
        chat.memberships.add([Membership(
            message_id=mid,
            thread_id=tid,
            score=float(best_score),
            reason="centroid_sim",
        )])

        rep = proc.thread_rep_computer.compute(tid)
        if rep.size:
            chat.embeddings.add(proc.thread_centroid_space, [tid], rep[None, :])

        assigned = {"thread_id": tid, "score": float(best_score)}

    chat.updated_at = dt.datetime.now()

    return jsonify({
        "status": "ok",
        "message_id": mid,
        "assigned": assigned,
        "created_new_topic": created_new,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
