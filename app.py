import os
import logging
import uuid
import datetime as dt

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA

# Import core components
from core.models import Message
from core.stores import MessageStore, ThreadStore, MembershipStore, EmbeddingStore
from core.processor import ChatProcessor
from core.strategies import (
    ContextWindowFormatter, MiniLMEmbedder, UMAPReducer,
    HDBSCANClusterer, CentroidThreadRepComputer, LlamaThreadLabeler,
    NoOpAssigner, NoOpUpdateStrategy
)
from utils import raw2df

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_PATH'] = './models/Llama-3.2-3B-Instruct-Q4_K_M.gguf'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- GLOBAL SYSTEM INITIALIZATION ---
# We initialize the processor as a global object so it persists between requests.
print("Initializing AI System... (This may take a minute)")

messages = MessageStore()
threads = ThreadStore()
memberships = MembershipStore()
embeddings = EmbeddingStore()

# Initialize Strategies
formatter = ContextWindowFormatter(window_back=2, window_fwd=1)
embedder = MiniLMEmbedder("all-MiniLM-L6-v2")
reducer = UMAPReducer(n_neighbors=30, n_components=5, min_dist=0.0)
clusterer = HDBSCANClusterer(min_cluster_size=30, min_samples=3)

# Only load Llama if model exists, otherwise warn user
if os.path.exists(app.config['MODEL_PATH']):
    labeler = LlamaThreadLabeler(model_path=app.config['MODEL_PATH'], n_ctx=2048)
else:
    logger.error(f"WARNING: Llama model not found at {app.config['MODEL_PATH']}. Labeling will fail.")
    labeler = None

thread_rep = CentroidThreadRepComputer(memberships=memberships, embeddings=embeddings)

# The Processor
processor = ChatProcessor(
    messages=messages,
    threads=threads,
    memberships=memberships,
    embeddings=embeddings,
    embedder=embedder,
    reducer=reducer,
    clusterer=clusterer,
    thread_rep_computer=thread_rep,
    assigner=NoOpAssigner(),
    update_strategy=NoOpUpdateStrategy(),
    formatter=formatter,
    labeler=labeler
)
print("System Initialized.")


# --- API ENDPOINTS ---

@app.route('/')
def index():
    """Simple UI to interact with the API."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_chat():
    """Upload a .txt file, parse it, and run the batch pipeline."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Parse
        df = raw2df(filepath, '12hr')  # Assuming 12hr format, can be made dynamic

        # Populate Store
        msgs_objs = []
        for i, row in df.iterrows():
            msgs_objs.append(Message(
                id=f"m{i}",
                timestamp=row['date_time'].to_pydatetime(),
                user=str(row['user']),
                text=str(row['message'])
            ))
        processor.messages.add(msgs_objs)

        # Run Pipeline
        processor.run_batch()

        return jsonify({
            "status": "success",
            "message_count": len(processor.messages.all()),
            "thread_count": len(processor.threads.all())
        })

    except Exception as e:
        logger.exception("Upload failed")
        return jsonify({"error": str(e)}), 500


@app.route('/api/threads', methods=['GET'])
def get_threads():
    """Get list of all discovered threads."""
    result = []
    for t in processor.threads.all():
        # Get message count for this thread
        count = len(processor.memberships.for_thread(t.id, status="active"))
        result.append({
            "id": t.id,
            "title": t.title,
            "summary": t.summary,
            "message_count": count,
            "updated_at": t.updated_at.isoformat()
        })

    # Sort by message count desc
    result.sort(key=lambda x: x['message_count'], reverse=True)
    return jsonify(result)


@app.route('/api/chat/history', methods=['GET'])
def get_full_history():
    """Get chronological messages with pagination."""
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    # Get all message IDs in chronological order
    all_ids = processor.messages.ids()

    # Slice the list
    sliced_ids = all_ids[offset: offset + limit]

    # Build response
    results = []
    for mid in sliced_ids:
        m = processor.messages.get(mid)
        # Check if it belongs to a thread (to show a tag in the UI)
        memberships = processor.memberships.for_message(mid, status="active")
        tid = memberships[0].thread_id if memberships else None
        t_title = processor.threads.get(tid).title if tid else None

        results.append({
            "id": m.id,
            "user": m.user,
            "text": m.text,
            "timestamp": m.timestamp.isoformat(),
            "thread_id": tid,
            "thread_title": t_title
        })

    return jsonify({
        "messages": results,
        "total": len(all_ids)
    })

@app.route('/api/chat/context/<message_id>', methods=['GET'])
def get_message_context(message_id):
    """Get the surrounding neighbors of a message (Timeline context)."""
    window = int(request.args.get('window', 5))  # How many msgs before/after

    all_ids = processor.messages.ids()

    try:
        # Find index of the target message
        idx = all_ids.index(message_id)
    except ValueError:
        return jsonify({"error": "Message not found"}), 404

    # Calculate slice bounds safely
    start = max(0, idx - window)
    end = min(len(all_ids), idx + window + 1)

    context_ids = all_ids[start:end]

    results = []
    for mid in context_ids:
        m = processor.messages.get(mid)
        # Mark which one is the target so UI can highlight it
        is_target = (mid == message_id)

        results.append({
            "id": m.id,
            "user": m.user,
            "text": m.text,
            "timestamp": m.timestamp.isoformat(),
            "is_target": is_target
        })

    return jsonify(results)

@app.route('/api/threads/<thread_id>/messages', methods=['GET'])
def get_thread_messages(thread_id):
    """Get actual messages for a specific thread."""
    msgs = processor.get_messages_for_thread(thread_id)
    return jsonify([{
        "id": m.id,
        "user": m.user,
        "text": m.text,
        "timestamp": m.timestamp.isoformat()
    } for m in msgs])


@app.route('/api/search', methods=['GET'])
def search():
    """Semantic Search."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Query parameter 'q' required"}), 400

    results = processor.semantic_search(query)
    return jsonify(results)


@app.route('/api/fix', methods=['POST'])
def apply_fix():
    """Human-in-the-Loop Correction."""
    data = request.json
    mid = data.get('message_id')
    new_tid = data.get('thread_id')

    if not mid or not new_tid:
        return jsonify({"error": "Missing message_id or thread_id"}), 400

    # We need to find what thread it was in to remove it
    current_memberships = processor.memberships.for_message(mid, status="active")
    remove_tids = [m.thread_id for m in current_memberships]

    processor.apply_user_fix(
        message_id=mid,
        add_to=[new_tid],
        remove_from=remove_tids
    )

    return jsonify({"status": "fixed", "moved_to": new_tid})

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get unique list of users for the users dropdown."""
    users = set()
    for m in processor.messages.all():
        users.add(m.user)
    return jsonify(sorted(list(users)))


@app.route('/api/chat/message', methods=['POST'])
def post_message():
    """Simulate a new incoming message."""
    data = request.json
    user = data.get('user')
    text = data.get('text')

    if not user or not text:
        return jsonify({"error": "User and text required"}), 400

    # Create new Message object
    new_id = f"new_{uuid.uuid4().hex[:8]}"
    msg = Message(
        id=new_id,
        timestamp=dt.datetime.now(),
        user=user,
        text=text
    )

    # Trigger the pipeline
    processor.ingest_new_message(msg)

    return jsonify({
        "status": "pending",
        "id": new_id,
        "assigned_to": -1,
        "score": 0.0
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)