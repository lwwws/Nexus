# Nexus — Chat Topic Disentanglement

Nexus is a web app for **chat topic disentanglement**: you upload a WhatsApp chat export, and Nexus automatically splits the conversation into **topics** (threads). You can then browse topics, read summaries, and switch between a full timeline view and a focused topic view.

![Pipeline](media/Nexus%20Pipeline.png)  

---

## What you can do in Nexus

### 1) Upload WhatsApp chat exports
- Export your WhatsApp chat as a **.txt** file.
- Upload it in Nexus.
- Nexus parses the messages and starts the disentanglement pipeline.

### 2) Navigate between uploaded chats (left sidebar)
- The **left sidebar** lists the chats you uploaded.
- Click a chat to load it.

### 3) See processing progress
When you upload a chat, Nexus processes it and shows progress (e.g., similar to a tqdm progress indicator in logs) so you can tell:
- that the system is still working
- roughly how far along it is

### 4) Explore topics after processing (right sidebar)
After processing finishes, a **right sidebar** appears with:
- A list of **topics**
- Each topic has a **collapsible summary**

Clicking a topic activates **Focus Mode** (see below).

### 5) Focus Mode vs Timeline Mode
Nexus supports two ways to view a chat:

**Timeline Mode**
- Shows the full chat as it happened, in chronological order.

**Focus Mode**
- Shows only the messages that belong to the selected topic.
- Messages from other topics are **collapsed**, with an option to **unfold** them if needed.
- This helps you follow one conversation thread without losing context.

You can toggle between **Focus Mode** and **Timeline Mode**.

### 6) Append new messages to an ongoing chat
Nexus supports continued chatting after upload:
- Choose which “user” you want to write as (intentionally simple).
- Send a new message.
- The message is processed and assigned to a topic.
- The topic list and focus view update accordingly.

---

## Running Nexus locally

### 1) Create a virtual environment

**Windows (PowerShell)**
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Start the server
```bash
python app.py
```

Then open the URL printed in the terminal (commonly `http://127.0.0.1:5000`).

---

## Exporting a WhatsApp chat

On WhatsApp (mobile):
1. Open the chat
2. Tap the menu (⋮) / chat info
3. Choose **Export chat**
4. Pick **Without media** (recommended)
5. Save/share the resulting `.txt` file
6. Upload it in Nexus

> Note: steps may vary slightly between iOS/Android and WhatsApp versions.

---

## Tips for best results

- Export **without media** (faster, smaller files).
- Very large chats take longer to process—watch the progress indicator.
- If summaries/titles look generic, it usually means the current strategy is using a fallback labeler (depending on your configuration).

---

## Known limitations (normal for a prototype)

- Topic boundaries are heuristic/ML-based and may not be perfect.
- If names/timestamps are formatted differently than expected, parsing may need adjustment.
- LLM-based labeling (if enabled) can be slower and may require extra dependencies.

---

## Privacy
Nexus runs locally (unless you deploy it elsewhere). Your chat data stays wherever you run Nexus.  