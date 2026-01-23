const state = {
    activeChatId: null,
    activeChatName: null,

    mode: "focus",
    activeTopicId: null,

    // Timeline paging (from the bottom)
    timelineLimit: 80,
    timelineOffset: 0,
    timelineMessages: [],
    timelineHasMore: false,

    // Focus gaps expansion: key "a|b" -> array of messages between
    expandedBetween: {},

    cachedTopics: [],
    cachedChats: [],
    isSearching: false,
    lastMode: "focus",
  };

  function $(id){ return document.getElementById(id); }

    function setRightCollapsed(isCollapsed){
    const pane = $("rightPane");
    pane.classList.toggle("collapsed", !!isCollapsed);
    localStorage.setItem("nexus:rightCollapsed", isCollapsed ? "1" : "0");
  }

  // Restore on load
  setRightCollapsed(localStorage.getItem("nexus:rightCollapsed") === "1");

  $("rightCollapseBtn").onclick = () => setRightCollapsed(true);
  $("rightExpandBtn").onclick = () => setRightCollapsed(false);

  function esc(s){
    return String(s ?? "")
      .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#039;");
  }

  async function apiGet(url){
    const res = await fetch(url);
    if(!res.ok){
      const t = await res.text().catch(()=> "");
      throw new Error(`${res.status} ${res.statusText}${t ? " — " + t : ""}`);
    }
    return res.json();
  }

  async function apiPost(url, body){
    const res = await fetch(url, {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    if(!res.ok){
      const t = await res.text().catch(()=> "");
      throw new Error(`${res.status} ${res.statusText}${t ? " — " + t : ""}`);
    }
    return res.json();
  }

  function setLeftError(msg){
    const el = $("leftError");
    if(!msg){ el.style.display="none"; el.textContent=""; return; }
    el.style.display="block";
    el.textContent = msg;
  }
  function setRightError(msg){
    const el = $("rightError");
    if(!msg){ el.style.display="none"; el.textContent=""; return; }
    el.style.display="block";
    el.textContent = msg;
  }

  function setCenter(title, sub){
    $("centerTitle").textContent = title || "Nexus";
    $("centerSub").textContent = sub || "";
  }

  function showOverlay(show){ $("overlay").style.display = show ? "flex" : "none"; }
  function setJobUI({stage, progress, detail, error, status}){
    $("jobStage").textContent = stage || "";
    $("jobDetail").textContent = detail || "";
    $("jobBar").style.width = `${Math.max(0, Math.min(100, progress || 0))}%`;
    if(error){
      $("jobError").style.display = "block";
      $("jobError").textContent = error;
    } else {
      $("jobError").style.display = "none";
      $("jobError").textContent = "";
    }
    $("closeOverlay").disabled = !(status === "done" || status === "error");
  }

  // Deterministic color per username
  function userColor(name){
    if(!name) return "var(--text)";
    let h = 0;
    for(let i=0;i<name.length;i++){
      h = (h*31 + name.charCodeAt(i)) >>> 0;
    }
    // map to hue range with decent contrast
    const hue = (h % 280) + 40;
    return `hsl(${hue} 85% 72%)`;
  }

  function formatDateLabel(iso){
    const d = new Date(iso);
    const opts = { year: "numeric", month: "short", day: "numeric" };
    return d.toLocaleDateString(undefined, opts);
  }

  function formatTime(iso){
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, {hour:"2-digit", minute:"2-digit"});
  }

  // -----------------------------
  // LEFT: chats
  // -----------------------------
  function renderChats(chats){
    const box = $("chatList");
    if(!chats || chats.length === 0){
      box.innerHTML = `<div class="empty">No chats uploaded yet.</div>`;
      return;
    }

    box.innerHTML = chats.map(c => {
      const active = (c.chat_id === state.activeChatId) ? "active" : "";
      const metaL = c.is_ready ? `${c.topic_count} topics` : "Processing…";
      const metaR = c.message_count ? `${c.message_count} msgs` : "";
      return `
        <div class="chatCard ${active}" data-chat="${esc(c.chat_id)}" data-name="${esc(c.name)}">
          <div class="title">${esc(c.name)}</div>
          <div class="meta"><span>${esc(metaL)}</span><span>${esc(metaR)}</span></div>
        </div>
      `;
    }).join("");

    box.onclick = (e) => {
      const card = e.target.closest(".chatCard");
      if(!card) return;
      selectChat(card.getAttribute("data-chat"), card.getAttribute("data-name"));
    };
  }

  async function loadChats(){
    const chats = await apiGet("/api/chats");
    state.cachedChats = chats;
    renderChats(chats);
  }

  // -----------------------------
  // RIGHT: topics
  // -----------------------------
  function renderTopics(topics){
    const box = $("topicList");
    if(!topics || topics.length === 0){
      box.innerHTML = `<div class="empty">No topics yet.</div>`;
      return;
    }

    box.innerHTML = topics.map(t => {
      const active = (t.id === state.activeTopicId) ? "active" : "";
      return `
        <div class="topicCard ${active}">
          <div class="topicTop">
            <div style="min-width:0;">
              <div class="topicTitle">${esc(t.title || "Untitled topic")}</div>
              <div class="topicMeta">${esc(t.message_count || 0)} messages</div>
            </div>
            <button class="primary" data-topic="${esc(t.id)}" style="padding:8px 10px; border-radius:14px;">
              Focus
            </button>
          </div>

          <details>
            <summary>Summary</summary>
            <div class="summaryBox">${esc(t.summary || "Summary not available.")}</div>
          </details>
        </div>
      `;
    }).join("");

    box.querySelectorAll("button[data-topic]").forEach(btn => {
      btn.onclick = () => {
        state.activeTopicId = btn.getAttribute("data-topic");
        state.expandedBetween = {};
        setMode("focus");
      };
    });
  }

  async function loadTopics(){
    setRightError("");
    if(!state.activeChatId){
      renderTopics([]);
      return;
    }
    const topics = await apiGet(`/api/chats/${encodeURIComponent(state.activeChatId)}/topics`);
    state.cachedTopics = topics;
    $("rightHint").textContent = topics.length ? "" : "Topics appear after processing.";
    filterTopics();
  }

  function filterTopics(){
    const q = $("topicSearch").value.trim().toLowerCase();
    if(!q){
      renderTopics(state.cachedTopics);
      return;
    }
    const filtered = state.cachedTopics.filter(t =>
      String(t.title || "").toLowerCase().includes(q) ||
      String(t.summary || "").toLowerCase().includes(q)
    );
    renderTopics(filtered);
  }

  // -----------------------------
  // Users dropdown
  // -----------------------------
  async function loadUsers(){
    if(!state.activeChatId){
      $("userSelect").innerHTML = `<option value="">Pick user…</option>`;
      return;
    }
    const users = await apiGet(`/api/chats/${encodeURIComponent(state.activeChatId)}/users`);
    $("userSelect").innerHTML =
      `<option value="">Pick user…</option>` +
      users.map(u => `<option value="${esc(u)}">${esc(u)}</option>`).join("");
  }

  // -----------------------------
  // Render messages WhatsApp-like:
  // - chronological order
  // - date separators
  // - system notifications centered
  // - name color per user
  // - newest at bottom (scroll down)
  // -----------------------------
  function renderMessageList(messages, opts){
    const {
      showTopicPills=false,
      alignment="auto", // "auto" uses left for all; timeline often uses left; you can customize later
      prependLoadOlder=false,
      loadOlderHandler=null,
    } = (opts || {});

    if(!messages || messages.length === 0){
      $("content").innerHTML = `<div class="empty">No messages.</div>`;
      return;
    }

    let html = "";

    if(prependLoadOlder){
      html += `<div class="loadOlderWrap">
        <button id="loadOlderBtn" class="ghost">Load earlier messages</button>
      </div>`;
    }

    let lastDate = null;

    for(const m of messages){
      const day = formatDateLabel(m.timestamp);
      if(day !== lastDate){
        lastDate = day;
        html += `<div class="dateSep"><div class="datePill">${esc(day)}</div></div>`;
      }

      if(m.is_system){
        html += `
          <div class="sys">
            <div class="sysText">${esc(m.text)}</div>
          </div>
        `;
        continue;
      }

      const time = formatTime(m.timestamp);
      const nameCol = userColor(m.user);

      const pill = (showTopicPills && m.thread_title)
        ? `<span class="pill">${esc(m.thread_title)}</span>`
        : "";

      // all messages are left-aligned like group chats
      const side = "left";

      html += `
        <div class="msgRow ${side}">
          <div class="bubble ${side === "right" ? "right" : ""}">
            <div class="metaLine">
              <div style="display:flex; align-items:center; gap:8px; min-width:0;">
                ${pill}
                <span class="name" style="color:${esc(nameCol)}">${esc(m.user)}</span>
              </div>
              <div class="time">
                <span>${esc(time)}</span>
              </div>
            </div>
            <div class="text">${esc(m.text)}</div>
          </div>
        </div>
      `;
    }

    $("content").innerHTML = html;

    if(prependLoadOlder){
      const btn = $("loadOlderBtn");
      if(btn) btn.onclick = loadOlderHandler;
    }
  }

  function scrollToBottom(){
    const c = $("content");
    c.scrollTop = c.scrollHeight;
  }

  // Helper: Scroll to message and flash it
    function scrollToMessage(mid, doFlash = true) {
      setTimeout(() => {
        const el = document.getElementById(`msg_${mid}`);
        if (el) {
          // "center" is aggressive; "nearest" is usually better for browsing flow,
          // but if you want to force attention, "center" or "start" works.
          el.scrollIntoView({ behavior: "auto", block: "center" });

          if (doFlash) {
            const bubble = el.querySelector('.bubble');
            if(bubble) {
              bubble.classList.remove("flash-target"); // reset if already there
              void bubble.offsetWidth; // trigger reflow to restart animation
              bubble.classList.add("flash-target");

              // Cleanup class after animation ends (3s matches CSS)
              setTimeout(() => bubble.classList.remove("flash-target"), 3000);
            }
          }
        }
      }, 50);
    }

  // -----------------------------
  // Timeline mode (from bottom)
  // -----------------------------
  async function loadTimeline(reset){
    if(!state.activeChatId) return;

    if(reset){
      state.timelineOffset = 0;
      state.timelineMessages = [];
    }

    const beforeHeight = $("content").scrollHeight;
    const beforeTop = $("content").scrollTop;

    const data = await apiGet(`/api/chats/${encodeURIComponent(state.activeChatId)}/history?limit=${state.timelineLimit}&offset=${state.timelineOffset}`);
    state.timelineHasMore = !!data.has_more;

    // New chunk is older or latest depending on offset:
    // offset=0 => latest chunk. offset>0 => older chunk.
    if(state.timelineOffset === 0){
      state.timelineMessages = data.messages;
    } else {
      // prepend older messages
      state.timelineMessages = data.messages.concat(state.timelineMessages);
    }

    state.timelineOffset = data.next_offset;

    renderMessageList(state.timelineMessages, {
      showTopicPills:true,
      prependLoadOlder: state.timelineHasMore,
      loadOlderHandler: async () => {
        // preserve scroll position when prepending older chunk
        const oldScrollHeight = $("content").scrollHeight;
        await loadTimeline(false);
        const newScrollHeight = $("content").scrollHeight;
        $("content").scrollTop = (newScrollHeight - oldScrollHeight) + beforeTop;
      }
    });

    setCenter(state.activeChatName || "Chat", `Timeline • newest at bottom`);
    if(reset) scrollToBottom();
  }

  // -----------------------------
  // Focus mode with gap expansion
  // -----------------------------
  async function loadFocus(topicId, maintainPosition = false){
    if(!state.activeChatId) return;

    // Capture current scroll position BEFORE we touch anything
    const contentDiv = $("content");
    const prevScroll = contentDiv.scrollTop;

    const data = await apiGet(`/api/chats/${encodeURIComponent(state.activeChatId)}/focus/${encodeURIComponent(topicId)}`);
    const topic = data.topic || {};
    const focus = data.focus || [];

    setCenter(state.activeChatName || "Chat", `Focus • ${topic.title || "Topic"} • ${focus.length} messages`);

    const items = [];
    let lastDate = null;
    function pushDateIfNeeded(ts){
      const day = formatDateLabel(ts);
      if(day !== lastDate){
        lastDate = day;
        items.push({type:"date", label: day});
      }
    }

    for(let i=0;i<focus.length;i++){
      const m = focus[i];
      pushDateIfNeeded(m.timestamp);
      items.push({type:"msg", msg:m});

      if(i < focus.length - 1){
        const a = focus[i];
        const b = focus[i+1];
        const gap = (b.idx - a.idx) - 1;
        if(gap > 0){
          const key = `${a.id}|${b.id}`;
          const expanded = state.expandedBetween[key];
          if(expanded && expanded.length){
            for(const x of expanded){
              pushDateIfNeeded(x.timestamp);
              items.push({type:"msg", msg:x, dim:true});
            }
          } else {
            items.push({type:"gap", from:a.id, to:b.id, count:gap});
          }
        }
      }
    }

    let html = "";
    for(const it of items){
      if(it.type === "date"){
        html += `<div class="dateSep"><div class="datePill">${esc(it.label)}</div></div>`;
        continue;
      }
      if(it.type === "gap"){
        html += `
          <div class="gapWrap">
            <button class="gapBtn ghost" data-gap-from="${esc(it.from)}" data-gap-to="${esc(it.to)}">
              Uncollapse ${esc(it.count)} messages between
            </button>
          </div>
        `;
        continue;
      }

      const m = it.msg;
      if(m.is_system){
        html += `<div class="sys"><div class="sysText">${esc(m.text)}</div></div>`;
        continue;
      }

      const time = formatTime(m.timestamp);
      const nameCol = userColor(m.user);
      const side = "left";

      html += `
        <div class="msgRow ${side}" id="msg_${m.id}">
          <div class="bubble ${side === "right" ? "right" : ""}" style="${it.dim ? "opacity:.6;" : ""}">
            <div class="metaLine">
              <div style="display:flex; align-items:center; gap:8px; min-width:0;">
                <span class="name" style="color:${esc(nameCol)}">${esc(m.user)}</span>
              </div>
              <div class="time">
                <span>${esc(time)}</span>
              </div>
            </div>
            <div class="text">${esc(m.text)}</div>
          </div>
        </div>
      `;
    }

    contentDiv.innerHTML = html;

    // UPDATE GAP HANDLER
    contentDiv.querySelectorAll("button[data-gap-from]").forEach(btn => {
      btn.onclick = async () => {
        const from = btn.getAttribute("data-gap-from");
        const to = btn.getAttribute("data-gap-to");
        const key = `${from}|${to}`;

        // Fetch expanded data
        const res = await apiGet(`/api/chats/${encodeURIComponent(state.activeChatId)}/between?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}`);
        state.expandedBetween[key] = res.messages || [];

        // RELOAD with maintainPosition = TRUE
        // We do NOT call scrollToMessage here anymore.
        await loadFocus(state.activeTopicId, true);
      };
    });

    await loadTopics();

    // RESTORE SCROLL
    if(maintainPosition) {
      // Restore exactly where we were.
      // Since the new content expanded *below* or *at* the button,
      // keeping the scrollTop the same means the view stays stable.
      contentDiv.scrollTop = prevScroll;
    } else {
      scrollToBottom();
    }
  }

  // -----------------------------
  // Mode switching
  // -----------------------------
  function setMode(mode){
    state.mode = mode;
    $("modeFocus").classList.toggle("active", mode === "focus");
    $("modeTimeline").classList.toggle("active", mode === "timeline");

    if(!state.activeChatId){
      $("content").innerHTML = `<div class="empty">Upload a chat export (.txt) to start.</div>`;
      setCenter("Nexus", "Upload a chat to begin.");
      return;
    }

    if(mode === "timeline"){
      loadTimeline(true);
    } else {
      if(state.activeTopicId){
        loadFocus(state.activeTopicId);
      } else {
        $("content").innerHTML = `<div class="empty">Pick a topic on the right to enter Focus Mode.</div>`;
        setCenter(state.activeChatName || "Chat", "Pick a topic → only its messages will be shown.");
      }
    }
  }

  // -----------------------------
  // Selecting a chat
  // -----------------------------
  async function selectChat(chatId, name){
    state.activeChatId = chatId;
    state.activeChatName = name;
    state.activeTopicId = null;
    state.expandedBetween = {};

    // Fix highlight bug by re-rendering chat list with new active id
    renderChats(state.cachedChats);

    setCenter(name || "Chat", "Loading…");
    await loadTopics();
    await loadUsers();

    // Default view: focus
    setMode("focus");
  }

  // -----------------------------
  // Upload with progress polling
  // -----------------------------
  async function uploadChat(){
    setLeftError("");
    const file = $("fileInput").files[0];
    if(!file){
      setLeftError("Choose a WhatsApp .txt export first.");
      return;
    }

    $("uploadBtn").disabled = true;
    showOverlay(true);
    setJobUI({stage:"Uploading…", progress:1, detail:"", error:null, status:"running"});

    try{
      const fd = new FormData();
      fd.append("file", file);

      const res = await fetch("/api/chats/upload", { method:"POST", body: fd });
      if(!res.ok){
        const t = await res.text().catch(()=> "");
        throw new Error(`${res.status} ${res.statusText}${t ? " — " + t : ""}`);
      }
      const data = await res.json();

      const jobId = data.job_id;
      const chatId = data.chat_id;

      // Poll
      let done = false;
      while(!done){
        const j = await apiGet(`/api/jobs/${encodeURIComponent(jobId)}`);
        setJobUI(j);
        if(j.status === "done" || j.status === "error"){
          done = true;
          break;
        }
        await new Promise(r => setTimeout(r, 250));
      }

      await loadChats();

      // Auto-select new chat
      const newChat = state.cachedChats.find(c => c.chat_id === chatId);
      if(newChat){
        await selectChat(newChat.chat_id, newChat.name);
      }

    } catch(err){
      setJobUI({stage:"Upload failed", progress:0, detail:"", error: err.message, status:"error"});
      setLeftError(err.message);
    } finally {
      $("uploadBtn").disabled = false;
    }
  }

  // -----------------------------
  // Append message
  // -----------------------------
  async function waitMessageProcessed(chatId, messageId, timeoutMs = 15000){
      const start = Date.now();
      while(Date.now() - start < timeoutMs){
        const data = await apiGet(`/api/chats/${encodeURIComponent(chatId)}/history?limit=10&offset=0`);
        const m = (data.messages || []).find(x => x.id === messageId);
        if(m && m.processed) return true;
        await new Promise(r => setTimeout(r, 300));
      }
      return false;
    }

  async function sendMessage(){
    if(!state.activeChatId){
      alert("Pick a chat first.");
      return;
    }
    const user = $("userSelect").value;
    const text = $("msgInput").value; // keep newlines
    if(!user){ alert("Pick a user."); return; }
    if(!text.trim()) return;

    $("sendBtn").disabled = true;
    try{
      const res = await apiPost(`/api/chats/${encodeURIComponent(state.activeChatId)}/message`, { user, text });
      const mid = res.message_id;
      $("msgInput").value = "";


      await waitMessageProcessed(state.activeChatId, mid);
      await loadTopics();

      if(state.mode === "timeline"){
        // Reload the latest window (newest at bottom)
        await loadTimeline(true);
      } else {
        if(state.activeTopicId){
          await loadFocus(state.activeTopicId);
        }
      }
    } catch(err){
      alert("Send failed: " + err.message);
    } finally {
      $("sendBtn").disabled = false;
    }
  }

    // --- Search Functions ---

    async function performSearch() {
      const query = $("globalSearchInput").value.trim();
      if (!query) return;
      if (!state.activeChatId) return;

      // UI Updates
      state.isSearching = true;
      $("clearSearchBtn").style.display = "block";
      $("content").innerHTML = `<div class="empty">Searching for "${esc(query)}"...</div>`;

      // Disable mode buttons visually
      $("modeFocus").classList.remove("active");
      $("modeTimeline").classList.remove("active");

      try {
        // Call the backend API
        const results = await apiGet(`/api/chats/${state.activeChatId}/search?q=${encodeURIComponent(query)}`);
        renderSearchResults(results, query);
      } catch (e) {
        $("content").innerHTML = `<div class="empty" style="color:var(--danger)">Search failed: ${e.message}</div>`;
      }
    }

    function renderSearchResults(results, query) {
      if (!results || results.length === 0) {
        $("content").innerHTML = `<div class="empty">No relevant topics found for "${esc(query)}".</div>`;
        return;
      }

      let html = `<div style="padding-bottom: 20px;">
        <div class="sys" style="margin-bottom:10px;">Found ${results.length} relevant topics</div>`;

      results.forEach(group => {
        html += `
          <div class="searchSection">
            <div class="searchSectionHeader">
              <span>${esc(group.thread_title || "Untitled Topic")}</span>
              <span class="searchScore">Rel: ${(group.thread_score * 100).toFixed(0)}%</span>
            </div>
            <div class="searchSectionBody">
        `;

        group.messages.forEach(m => {
          const time = formatTime(m.timestamp);
          const nameCol = userColor(m.user);

          // Clickable Bubble containing the text ---
          html += `
            <div class="msgRow left" style="cursor:pointer;"
                 onclick="goToTopic('${group.thread_id}', '${m.message_id}')"
                 title="Jump to context">
              <div class="bubble">
                <div class="metaLine">
                  <span class="name" style="color:${esc(nameCol)}">${esc(m.user)}</span>
                  <span class="time">${esc(time)} <span style="opacity:0.5">(${(m.score*100).toFixed(0)}%)</span></span>
                </div>
                <div class="text">${esc(m.text)}</div>
              </div>
            </div>
          `;
        });

        html += `</div></div>`; // Close Body and Section
      });

      html += `</div>`;
      $("content").innerHTML = html;
    }

    function closeSearch() {
      state.isSearching = false;
      $("globalSearchInput").value = "";
      $("clearSearchBtn").style.display = "none";

      // Return to previous view
      setMode(state.lastMode || "focus");
    }

    // Helper to jump from search result to Focus mode
    window.goToTopic = async (tid, mid) => {
      state.activeTopicId = tid;
      closeSearch();
      setMode("focus");

      // Default load (scrolls to bottom)
      await loadFocus(tid, false);

      // If we have a specific message ID, we then scroll to it and flash
      if (mid) {
        scrollToMessage(mid, true);
      }
    };


  // -----------------------------
  // Wire up
  // -----------------------------
  $("modeFocus").onclick = () => setMode("focus");
  $("modeTimeline").onclick = () => setMode("timeline");
  $("topicSearch").addEventListener("input", filterTopics);

  $("sendBtn").onclick = sendMessage;
  $("msgInput").addEventListener("keydown", (e)=> { if(e.key==="Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

  $("closeCtx").onclick = closeCtx;
  $("ctxBackdrop").addEventListener("click", (e)=> { if(e.target === $("ctxBackdrop")) closeCtx(); });
  document.addEventListener("keydown", (e)=> { if(e.key==="Escape") closeCtx(); });

  $("closeOverlay").onclick = () => showOverlay(false);

  // Custom file picker
  $("fileBtn").onclick = () => $("fileInput").click();
  $("fileInput").addEventListener("change", () => {
    const f = $("fileInput").files[0];
    $("fileName").textContent = f ? f.name : "Choose WhatsApp export (.txt)";
  });

  $("uploadBtn").onclick = uploadChat;

  // Init
  (async function init(){
    await loadChats();
  })();

  // Search Wiring
  $("globalSearchInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") performSearch();
  });

  $("clearSearchBtn").onclick = closeSearch;

  // Update setMode to store history so we know where to go back to
  const originalSetMode = setMode;
  // We overwrite setMode slightly to track state
  setMode = function(mode) {
    if(mode !== "focus" && mode !== "timeline") return; // safety
    state.lastMode = mode;
    state.isSearching = false;
    $("clearSearchBtn").style.display = "none";
    $("globalSearchInput").value = "";

    // Call original logic (copy-paste your setMode logic here or refactor slightly)
    state.mode = mode;
    $("modeFocus").classList.toggle("active", mode === "focus");
    $("modeTimeline").classList.toggle("active", mode === "timeline");

    // ... existing setMode logic ...
    if(!state.activeChatId){
      $("content").innerHTML = `<div class="empty">Upload a chat export (.txt) to start.</div>`;
      setCenter("Nexus", "Upload a chat to begin.");
      return;
    }

    if(mode === "timeline"){
      loadTimeline(true);
    } else {
      if(state.activeTopicId){
        loadFocus(state.activeTopicId);
      } else {
        $("content").innerHTML = `<div class="empty">Pick a topic to enter Focus Mode.</div>`;
        setCenter(state.activeChatName || "Chat", "Pick a topic to enter Focus Mode.");
      }
    }
  }
