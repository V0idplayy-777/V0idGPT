/* ═══════════════════════ V0idGPT app ═══════════════════════
   ChatGPT-style interface. Every assistant reply is generated live,
   in this browser, by the bundled NeuronForge V0idGPT model
   (javascript/model.mjs) — there is no backend and no other AI.    */
const MODELS = {
  "v0idgpt": {
    label: "V0idGPT",
    blurb: "6.4K params · character-level",
    unit: "chars",
    unitLong: "characters",
    prompt: "qa",
    presets: {
      coherent: { temperature: 0.4, topK: 3, maxTokens: 140 },
      balanced: { temperature: 0.75, topK: 8, maxTokens: 140 },
      wild: { temperature: 1.15, topK: 20, maxTokens: 200 },
    },
    load: () => import("../javascript/model.mjs"),
  },
  "v0idgpt-2-40k": {
    label: "v0idgpt-2-40k",
    blurb: "41.8K params · BPE subword",
    unit: "tokens",
    unitLong: "tokens",
    prompt: "plain",
    presets: {
      coherent: { temperature: 0.45, topK: 4, maxTokens: 60 },
      balanced: { temperature: 0.7, topK: 12, maxTokens: 80 },
      wild: { temperature: 1.1, topK: 60, maxTokens: 120 },
    },
    load: () => import("../models/v0idgpt-2-40k/model.mjs"),
  },
  "v0idgpt-3-3M-reasoning": {
    label: "v0idgpt-3-3M-reasoning",
    blurb: "3.04M params · conversational reasoning",
    unit: "chars",
    unitLong: "characters",
    prompt: "chat",
    presets: {
      coherent: { temperature: 0.1, topK: 3, maxTokens: 100 },
      balanced: { temperature: 0.2, topK: 4, maxTokens: 140 },
      wild: { temperature: 0.75, topK: 12, maxTokens: 180 },
    },
    load: () => import("../models/v0idgpt-3-3M-reasoning/model.mjs"),
  },
  "v0idgpt-4-109M": {
    label: "v0idgpt-4-109M",
    blurb: "109M params · BPE subword · 256-token context",
    unit: "tokens",
    unitLong: "tokens",
    prompt: "chat",
    presets: {
      coherent: { temperature: 0.4, topK: 6, maxTokens: 100 },
      balanced: { temperature: 0.7, topK: 30, maxTokens: 150 },
      wild: { temperature: 1.0, topK: 80, maxTokens: 200 },
    },
    load: () => import("../models/v0idgpt-4-109M/model.mjs"),
  },
};
const DEFAULT_MODEL = "v0idgpt-4-109M";

/* ─────────────────────────── helpers ─────────────────────────── */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function refreshIcons() { if (window.lucide) window.lucide.createIcons(); }
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; }
  catch {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    let ok = false; try { ok = document.execCommand("copy"); } catch { /* noop */ }
    ta.remove(); return ok;
  }
}

/* ─────────────────────────── state ─────────────────────────── */
const LS_CHATS = "v0idgpt.chats.v1";
const LS_SETTINGS = "v0idgpt.settings.v1";
const LS_ACTIVE = "v0idgpt.active.v1";

const DEFAULT_SETTINGS = {
  model: DEFAULT_MODEL,
  theme: "system",          // system | light | dark
  stream: true,
  tidy: true,
  temperature: 0.2,
  topK: 4,
  maxTokens: 140,
};

let settings = { ...DEFAULT_SETTINGS };
try { settings = { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(LS_SETTINGS) || "{}") }; } catch { /* fresh */ }

if (!MODELS[settings.model]) settings.model = DEFAULT_MODEL;
if (!settings.samplers || typeof settings.samplers !== "object") {
  settings.samplers = { [DEFAULT_MODEL]: { ...MODELS[DEFAULT_MODEL].presets.balanced } };
}
// Earlier 3M defaults were too hot for this small character-level model and
// often wandered off-topic. Migrate those exact preset values; preserve custom
// sampler choices.
const oldThreeMSampler = settings.samplers[DEFAULT_MODEL];
if (oldThreeMSampler?.topK === 8 && oldThreeMSampler?.maxTokens === 140 && [0.75, 0.45].includes(oldThreeMSampler?.temperature)) {
  settings.samplers[DEFAULT_MODEL] = { ...MODELS[DEFAULT_MODEL].presets.balanced };
}

let chats = [];
try { chats = JSON.parse(localStorage.getItem(LS_CHATS) || "[]"); if (!Array.isArray(chats)) chats = []; } catch { chats = []; }

let activeId = localStorage.getItem(LS_ACTIVE) || null;
if (activeId && !chats.some((c) => c.id === activeId)) activeId = chats[0]?.id || null;
let model = null;
let modelId = null;
let modelReady = false;
let switchingModel = false;
const loadedModels = {};
let generating = null;        // { stopped:boolean, chatId } while a reply streams
let pendingAttachment = null; // { name, text }
let menuChatId = null;

const dropTrace = (k, v) => (k === "trace" ? undefined : v);
const tokenViewMsgs = new WeakSet();
const saveSettings = () => {
  if (modelId) settings.samplers[modelId] = { temperature: settings.temperature, topK: settings.topK, maxTokens: settings.maxTokens };
  localStorage.setItem(LS_SETTINGS, JSON.stringify(settings));
};
const saveChats = () => localStorage.setItem(LS_CHATS, JSON.stringify(chats, dropTrace));
function applySampler(id) {
  const s = settings.samplers[id] || MODELS[id].presets.balanced;
  settings.temperature = s.temperature;
  settings.topK = s.topK;
  settings.maxTokens = s.maxTokens;
}
const activeChat = () => chats.find((c) => c.id === activeId) || null;

function createChat() {
  const chat = { id: uid(), title: "New chat", messages: [], createdAt: Date.now(), updatedAt: Date.now() };
  chats.unshift(chat);
  activeId = chat.id;
  localStorage.setItem(LS_ACTIVE, activeId);
  saveChats();
  return chat;
}
function ensureChat() { return activeChat() || createChat(); }

/* ─────────────────────────── toasts ─────────────────────────── */
function toast(message, { icon = "check", actionLabel = null, onAction = null, ms = 3400 } = {}) {
  const wrap = $("#toastWrap");
  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = `<i data-lucide="${icon}"></i><span>${esc(message)}</span>`;
  if (actionLabel) {
    const btn = document.createElement("button");
    btn.textContent = actionLabel;
    btn.addEventListener("click", () => { dismiss(); onAction && onAction(); });
    el.appendChild(btn);
  }
  wrap.appendChild(el);
  refreshIcons();
  const dismiss = () => { el.classList.add("out"); setTimeout(() => el.remove(), 260); };
  setTimeout(dismiss, ms);
  return dismiss;
}

/* ─────────────────────── confirm dialog ─────────────────────── */
let confirmResolve = null;
function confirmDialog({ title = "Are you sure?", text = "", okLabel = "Delete" } = {}) {
  $("#confirmTitle").textContent = title;
  $("#confirmText").textContent = text;
  $("#confirmOk").textContent = okLabel;
  $("#confirmOverlay").hidden = false;
  refreshIcons();
  $("#confirmOk").focus();
  return new Promise((res) => { confirmResolve = res; });
}
function closeConfirm(result) {
  $("#confirmOverlay").hidden = true;
  if (confirmResolve) { confirmResolve(result); confirmResolve = null; }
}
$("#confirmCancel").addEventListener("click", () => closeConfirm(false));
$("#confirmOk").addEventListener("click", () => closeConfirm(true));
$("#confirmOverlay").addEventListener("click", (e) => { if (e.target.id === "confirmOverlay") closeConfirm(false); });

/* ─────────────────────────── theme ─────────────────────────── */
const darkMQ = window.matchMedia("(prefers-color-scheme: dark)");
function resolvedTheme() { return settings.theme === "system" ? (darkMQ.matches ? "dark" : "light") : settings.theme; }
function applyTheme() {
  const t = resolvedTheme();
  document.documentElement.dataset.theme = t;
  $("#themeBtn").innerHTML = `<i data-lucide="${t === "dark" ? "sun" : "moon"}"></i>`;
  $$("#themeSegmented button").forEach((b) => b.classList.toggle("active", b.dataset.themeVal === settings.theme));
  refreshIcons();
}
darkMQ.addEventListener?.("change", () => { if (settings.theme === "system") applyTheme(); });
$("#themeBtn").addEventListener("click", () => {
  settings.theme = resolvedTheme() === "dark" ? "light" : "dark";
  saveSettings(); applyTheme();
});
$$("#themeSegmented button").forEach((b) => b.addEventListener("click", () => {
  settings.theme = b.dataset.themeVal; saveSettings(); applyTheme();
}));

/* ───────────────────── sidebar: chat list ───────────────────── */
function groupLabel(ts) {
  const now = new Date(); const d = new Date(ts);
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 86400000;
  if (ts >= startOfToday) return "Today";
  if (ts >= startOfToday - day) return "Yesterday";
  if (ts >= startOfToday - 7 * day) return "Previous 7 days";
  if (ts >= startOfToday - 30 * day) return "Previous 30 days";
  return d.getFullYear() + "";
}
function chatMatches(chat, q) {
  if (!q) return true;
  const hay = (chat.title + " " + chat.messages.map((m) => m.content).join(" ")).toLowerCase();
  return hay.includes(q.toLowerCase());
}
function renderChatList() {
  const list = $("#chatList");
  const q = $("#searchInput").value.trim();
  list.innerHTML = "";
  const visible = chats.filter((c) => chatMatches(c, q));
  if (!visible.length) {
    const div = document.createElement("div");
    div.className = "chat-empty";
    div.textContent = q ? `No chats match “${q}”.` : "No chats yet. Start a new conversation and it will appear here.";
    list.appendChild(div);
    return;
  }
  let lastGroup = null;
  for (const chat of visible) {
    const g = groupLabel(chat.updatedAt);
    if (g !== lastGroup) {
      lastGroup = g;
      const label = document.createElement("div");
      label.className = "chat-group-label";
      label.textContent = g;
      list.appendChild(label);
    }
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === activeId ? " active" : "");
    item.setAttribute("role", "listitem");
    const title = document.createElement("span");
    title.className = "chat-item-title";
    title.textContent = chat.title || "New chat";
    title.title = chat.title || "New chat";
    const menuBtn = document.createElement("button");
    menuBtn.className = "chat-item-menu-btn";
    menuBtn.title = "Chat options";
    menuBtn.setAttribute("aria-label", "Chat options");
    menuBtn.innerHTML = `<i data-lucide="ellipsis"></i>`;
    menuBtn.addEventListener("click", (e) => { e.stopPropagation(); openChatMenu(chat.id, menuBtn); });
    item.append(title, menuBtn);
    item.addEventListener("click", () => {
      if (generating) stopGeneration();
      activeId = chat.id;
      localStorage.setItem(LS_ACTIVE, activeId);
      renderChatList(); renderThread();
      if (window.innerWidth <= 820) setSidebar(false);
      $("#composerInput").focus();
    });
    list.appendChild(item);
  }
  refreshIcons();
}

/* chat context menu */
function openChatMenu(chatId, anchor) {
  menuChatId = chatId;
  const menu = $("#chatMenu");
  menu.hidden = false;
  const r = anchor.getBoundingClientRect();
  menu.style.top = `${Math.min(r.bottom + 4, window.innerHeight - 170)}px`;
  menu.style.left = `${clamp(r.left - 140, 8, window.innerWidth - 200)}px`;
  refreshIcons();
}
function closeChatMenu() { $("#chatMenu").hidden = true; menuChatId = null; }
$("#chatMenuRename").addEventListener("click", () => { const id = menuChatId; closeChatMenu(); if (id) startRename(id); });
$("#chatMenuShare").addEventListener("click", () => { const id = menuChatId; closeChatMenu(); if (id) shareChat(chats.find((c) => c.id === id)); });
$("#chatMenuDelete").addEventListener("click", () => { const id = menuChatId; closeChatMenu(); if (id) deleteChat(id); });

function startRename(chatId) {
  const chat = chats.find((c) => c.id === chatId);
  if (!chat) return;
  activeId = chat.id; localStorage.setItem(LS_ACTIVE, activeId);
  renderChatList(); renderThread(false);
  const item = $$(".chat-item")[chats.filter((c) => chatMatches(c, $("#searchInput").value.trim())).indexOf(chat)];
  if (!item) return;
  const titleEl = item.querySelector(".chat-item-title");
  const input = document.createElement("input");
  input.className = "chat-rename-input";
  input.value = chat.title || "";
  input.setAttribute("aria-label", "Rename chat");
  titleEl.replaceWith(input);
  item.querySelector(".chat-item-menu-btn")?.remove();
  input.focus(); input.select();
  let done = false;
  const commit = (save) => {
    if (done) return; done = true;
    if (save && input.value.trim()) { chat.title = input.value.trim().slice(0, 80); chat.updatedAt = Date.now(); saveChats(); }
    renderChatList();
  };
  input.addEventListener("blur", () => commit(true));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit(true);
    if (e.key === "Escape") commit(false);
    e.stopPropagation();
  });
  input.addEventListener("click", (e) => e.stopPropagation());
}

function deleteChat(chatId, { silent = false } = {}) {
  const idx = chats.findIndex((c) => c.id === chatId);
  if (idx < 0) return;
  const [removed] = chats.splice(idx, 1);
  if (activeId === chatId) {
    activeId = chats[0]?.id || null;
    if (activeId) localStorage.setItem(LS_ACTIVE, activeId); else localStorage.removeItem(LS_ACTIVE);
  }
  saveChats(); renderChatList(); renderThread(false);
  if (!silent) {
    toast("Chat deleted", {
      icon: "trash-2", actionLabel: "Undo", ms: 5000,
      onAction: () => {
        chats.splice(Math.min(idx, chats.length), 0, removed);
        activeId = removed.id; localStorage.setItem(LS_ACTIVE, activeId);
        saveChats(); renderChatList(); renderThread(false);
      },
    });
  }
}

/* sidebar visibility */
function setSidebar(open) {
  const sb = $("#sidebar");
  sb.classList.toggle("closed", !open);
  const mobile = window.innerWidth <= 820;
  $("#openSidebarBtn").hidden = open;
  $("#newChatBtnHeader").hidden = open;
  $("#sidebarScrim").classList.toggle("hidden", !(open && mobile));
}
$("#closeSidebarBtn").addEventListener("click", () => setSidebar(false));
$("#openSidebarBtn").addEventListener("click", () => setSidebar(true));
$("#sidebarScrim").addEventListener("click", () => setSidebar(false));
function toggleSidebar() { setSidebar($("#sidebar").classList.contains("closed")); }

/* new chat + search */
function newChat() {
  if (generating) stopGeneration();
  const existing = activeChat();
  if (existing && existing.messages.length === 0) { $("#composerInput").focus(); renderThread(); return; }
  createChat();
  renderChatList(); renderThread();
  $("#composerInput").focus();
}
$("#newChatBtn").addEventListener("click", newChat);
$("#newChatBtnTop").addEventListener("click", newChat);
$("#newChatBtnHeader").addEventListener("click", newChat);

function toggleSearch(focus = true) {
  const box = $("#searchBox");
  const willShow = box.hidden;
  box.hidden = !willShow;
  if (willShow && focus) $("#searchInput").focus();
  if (!willShow) { $("#searchInput").value = ""; $("#searchClearBtn").hidden = true; renderChatList(); }
}
$("#searchToggleBtn").addEventListener("click", () => toggleSearch());
$("#searchNavBtn").addEventListener("click", () => toggleSearch());
$("#searchInput").addEventListener("input", () => {
  $("#searchClearBtn").hidden = !$("#searchInput").value;
  renderChatList();
});
$("#searchClearBtn").addEventListener("click", () => {
  $("#searchInput").value = ""; $("#searchClearBtn").hidden = true; renderChatList(); $("#searchInput").focus();
});

/* ─────────────────────── thread rendering ─────────────────────── */
const threadEl = $("#thread");
const scrollArea = $("#scrollArea");
let stickToBottom = true;

scrollArea.addEventListener("scroll", () => {
  const near = scrollArea.scrollHeight - scrollArea.scrollTop - scrollArea.clientHeight < 120;
  stickToBottom = near;
  $("#scrollBottomBtn").hidden = near || threadEl.hidden;
});
$("#scrollBottomBtn").addEventListener("click", () => scrollToBottom(true));
function scrollToBottom(smooth = false) {
  requestAnimationFrame(() => {
    scrollArea.scrollTo({ top: scrollArea.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  });
}

function showEmptyOrThread() {
  const chat = activeChat();
  const empty = !chat || chat.messages.length === 0;
  $("#emptyState").hidden = !empty;
  threadEl.hidden = empty;
  if (empty) $("#scrollBottomBtn").hidden = true;
  $("#composerInput").placeholder = empty ? "Ask anything" : "Ask anything";
  return empty;
}

function messageActions(msg, idx, { done = true } = {}) {
  const bar = document.createElement("div");
  bar.className = "msg-actions" + (done ? "" : "");
  const mk = (icon, title, handler, selected = false) => {
    const b = document.createElement("button");
    b.className = "icon-btn" + (selected ? " selected" : "");
    b.title = title; b.setAttribute("aria-label", title);
    b.innerHTML = `<i data-lucide="${icon}"></i>`;
    b.addEventListener("click", () => handler(b));
    return b;
  };
  if (msg.role === "user") {
    bar.append(
      mk("copy", "Copy", async (b) => { if (await copyText(msg.content)) flashBtn(b, "check"); }),
      mk("pencil", "Edit message", () => editUserMessage(idx)),
    );
  } else {
    bar.append(
      mk("copy", "Copy", async (b) => { if (await copyText(msg.content)) flashBtn(b, "check"); }),
      mk("volume-2", "Read aloud", (b) => toggleSpeak(msg, b)),
      mk("thumbs-up", "Good response", (b) => setFeedback(idx, "up", bar), msg.feedback === "up"),
      mk("thumbs-down", "Bad response", (b) => setFeedback(idx, "down", bar), msg.feedback === "down"),
      ...(msg.trace?.steps?.length ? [mk("scan-text", tokenViewMsgs.has(msg) ? "Hide token view" : "Show token view", () => toggleTokenView(msg), tokenViewMsgs.has(msg))] : []),
      mk("rotate-ccw", "Regenerate response", () => regenerate(idx)),
    );
    if (done && msg.stopped) {
      const meta = document.createElement("span");
      meta.className = "msg-meta";
      meta.style.alignSelf = "center";
      meta.textContent = "· stopped";
      bar.append(meta);
    }
  }
  return bar;
}
function flashBtn(btn, icon) {
  const old = btn.innerHTML;
  btn.innerHTML = `<i data-lucide="${icon}"></i>`;
  refreshIcons();
  setTimeout(() => { btn.innerHTML = old; refreshIcons(); }, 1200);
}
function setFeedback(idx, val) {
  const chat = activeChat(); if (!chat) return;
  const msg = chat.messages[idx]; if (!msg) return;
  msg.feedback = msg.feedback === val ? null : val;
  saveChats(); renderThread(false);
  if (msg.feedback) toast(val === "up" ? "Thanks for the feedback" : "Feedback noted — regenerate to try again", { icon: val === "up" ? "thumbs-up" : "thumbs-down" });
}

/* speech */
let speakingBtn = null;
function toggleSpeak(msg, btn) {
  const synth = window.speechSynthesis;
  if (!synth) { toast("Read aloud isn't supported in this browser", { icon: "volume-x" }); return; }
  if (synth.speaking && speakingBtn === btn) { synth.cancel(); speakingBtn = null; btn.classList.remove("selected"); return; }
  synth.cancel();
  const u = new SpeechSynthesisUtterance(msg.content);
  u.rate = 1; u.pitch = 1;
  speakingBtn = btn; btn.classList.add("selected");
  u.onend = u.onerror = () => { btn.classList.remove("selected"); if (speakingBtn === btn) speakingBtn = null; };
  synth.speak(u);
}

function renderThread(scroll = true) {
  const chat = activeChat();
  threadEl.innerHTML = "";
  showEmptyOrThread();
  if (!chat) return;
  chat.messages.forEach((msg, idx) => threadEl.appendChild(buildMessageEl(msg, idx)));
  refreshIcons();
  if (scroll) scrollToBottom(false);
}

function buildMessageEl(msg, idx) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + msg.role;
  wrap.dataset.idx = idx;
  if (msg.role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = msg.content;
    wrap.append(bubble, messageActions(msg, idx));
  } else if (msg.notice) {
    const note = document.createElement("div");
    note.className = "msg-notice";
    note.innerHTML = `<i data-lucide="info"></i><span>${esc(msg.content)}</span>`;
    wrap.append(note, messageActions({ ...msg, content: "" }, idx));
    // notice messages only get regenerate
    wrap.querySelector(".msg-actions").innerHTML = "";
    const rb = document.createElement("button");
    rb.className = "icon-btn"; rb.title = "Regenerate response"; rb.setAttribute("aria-label", "Regenerate response");
    rb.innerHTML = `<i data-lucide="rotate-ccw"></i>`;
    rb.addEventListener("click", () => regenerate(idx));
    wrap.querySelector(".msg-actions").append(rb);
  } else {
    const body = document.createElement("div");
    body.className = "msg-body";
    if (msg.trace?.steps?.length && tokenViewMsgs.has(msg)) renderTokenView(body, msg);
    else body.textContent = msg.content;
    wrap.append(body, messageActions(msg, idx));
  }
  return wrap;
}

function visTok(t) { return String(t).replace(/ /g, "␣").replace(/\n/g, "↵") || "∅"; }
function pct(x) { return (x * 100).toFixed(x < 0.01 ? 2 : 1) + "%"; }
function toggleTokenView(msg) {
  closeTokenPop();
  if (tokenViewMsgs.has(msg)) tokenViewMsgs.delete(msg); else tokenViewMsgs.add(msg);
  renderThread(false);
}
function renderTokenView(body, msg) {
  const legend = document.createElement("div");
  legend.className = "tok-legend";
  legend.textContent = "Raw model output, token by token. Darker underline means the model was less sure. Tap a token to see what else it considered.";
  const line = document.createElement("div");
  line.className = "tok-line";
  msg.trace.steps.forEach((st, i) => {
    const sp = document.createElement("span");
    sp.className = "tok";
    sp.tabIndex = 0;
    sp.setAttribute("role", "button");
    const picked = st.c.find((c) => c[3]);
    sp.style.setProperty("--u", (1 - Math.min(1, picked ? picked[1] : 1)).toFixed(2));
    sp.textContent = st.t;
    const open = (e) => { e.stopPropagation(); openTokenPop(msg, i, sp); };
    sp.addEventListener("click", open);
    sp.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(e); } });
    line.appendChild(sp);
  });
  body.append(legend, line);
}
function closeTokenPop() {
  const pop = $("#tokenPop");
  if (!pop || pop.hidden) return;
  pop.hidden = true;
  $$(".tok.active").forEach((t) => t.classList.remove("active"));
}
function openTokenPop(msg, i, anchor) {
  const pop = $("#tokenPop");
  const wasSame = !pop.hidden && anchor.classList.contains("active");
  closeTokenPop();
  if (wasSame) return;
  const st = msg.trace.steps[i];
  const rows = [...st.c].sort((a, b) => b[1] - a[1]);
  const top = rows[0][1] || 1;
  pop.innerHTML = `
    <div class="tp-head"><b>Step ${i + 1}</b><span>picked <code>${esc(visTok(st.t))}</code></span></div>
    <div class="tp-cols"><span>Candidate</span><span></span><span>Model</span><span>Drawn</span></div>
    ${rows.map((c) => `<div class="tp-row${c[3] ? " picked" : ""}${c[2] === 0 ? " cut" : ""}"><span class="tp-tok">${esc(visTok(c[0]))}</span><span class="tp-bar"><i style="width:${(c[1] / top * 100).toFixed(1)}%"></i></span><span class="tp-num">${pct(c[1])}</span><span class="tp-num">${c[2] ? pct(c[2]) : "—"}</span></div>`).join("")}
    <div class="tp-foot"><b>Model</b> is the raw probability. <b>Drawn</b> is the chance the sampler gave it after Temperature ${msg.trace.temperature.toFixed(2)} and Top-K ${msg.trace.topK}. Faded rows were cut by Top-K.</div>`;
  anchor.classList.add("active");
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  const left = clamp(r.left + r.width / 2 - pw / 2, 8, window.innerWidth - pw - 8);
  const below = r.bottom + 8 + ph <= window.innerHeight - 8;
  pop.style.left = `${left}px`;
  pop.style.top = `${below ? r.bottom + 8 : Math.max(8, r.top - ph - 8)}px`;
}
document.addEventListener("click", (e) => { if (!e.target.closest("#tokenPop") && !e.target.closest(".tok")) closeTokenPop(); });
scrollArea.addEventListener("scroll", closeTokenPop);
window.addEventListener("resize", closeTokenPop);

/* edit a user message: truncate everything after it and re-ask */
function editUserMessage(idx) {
  const chat = activeChat(); if (!chat || generating) return;
  const wrap = threadEl.querySelector(`.msg[data-idx="${idx}"]`);
  if (!wrap) return;
  const editor = document.createElement("div");
  editor.className = "msg-editor";
  const ta = document.createElement("textarea");
  ta.value = chat.messages[idx].content;
  ta.rows = Math.min(8, Math.max(1, ta.value.split("\n").length + 1));
  const actions = document.createElement("div");
  actions.className = "msg-editor-actions";
  const cancel = document.createElement("button"); cancel.className = "btn secondary"; cancel.textContent = "Cancel";
  const send = document.createElement("button"); send.className = "btn primary"; send.textContent = "Send";
  actions.append(cancel, send);
  editor.append(ta, actions);
  wrap.innerHTML = ""; wrap.appendChild(editor);
  ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
  const autoresize = () => { ta.style.height = "auto"; ta.style.height = Math.min(220, ta.scrollHeight) + "px"; };
  ta.addEventListener("input", autoresize); autoresize();
  cancel.addEventListener("click", () => renderThread(false));
  const commit = () => {
    const text = ta.value.trim();
    if (!text) return;
    if (switchingModel) { toast("Model is still loading — one moment", { icon: "loader" }); return; }
    chat.messages = chat.messages.slice(0, idx);
    chat.messages.push({ role: "user", content: text, ts: Date.now() });
    chat.updatedAt = Date.now();
    saveChats(); renderChatList(); renderThread(false);
    runGeneration();
  };
  send.addEventListener("click", commit);
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); } });
}

/* ─────────────────── the model: prompt + sampler ─────────────────── */
/* The 3M model is a locally trained character-level LSTM. Its replies are
   sampled from model probabilities; no prompt-to-response lookup is used. */
function cleanQuestion(text) {
  return text.replace(/\s+/g, " ").trim().replace(/[?.!]+$/, "");
}
function buildPrompt(chat) {
  if (MODELS[modelId].prompt === "plain") {
    const last = [...chat.messages].reverse().find((m) => m.role === "user");
    return last ? last.content.replace(/\s+/g, " ").trim().toLowerCase() : "";
  }
  if (MODELS[modelId].prompt === "chat") {
    const turns = chat.messages
      .filter((m) => !m.notice && m.content && (m.role === "user" || m.role === "assistant"))
      .slice(-8)
      .map((m) => `${m.role}: ${m.content.replace(/\s+/g, " ").trim().toLowerCase()}`);
    // Keep recent turns in the same format used by training. The model's
    // context window will naturally retain the most recent characters.
    return `${turns.join("\n")}\nassistant:`;
  }
  let s = "";
  for (const m of chat.messages) {
    if (m.role === "user") s += (s ? "\n" : "") + "q: " + cleanQuestion(m.content) + "?";
    else if (!m.notice && m.content) s += " a: " + m.content.replace(/\s+/g, " ").trim();
  }
  s += " a:";
  return s.slice(-256);
}

/* Same temperature + top-k sampler the runtime's generate() uses. */
function sampleToken(probs, temperature, topK) {
  const n = probs.length;
  let best = 0;
  for (let i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
  if (!Number.isFinite(temperature) || temperature <= 0.01) return best;
  const invT = 1 / temperature;
  const logits = Array.from(probs, (p) => Math.log(Math.max(Number(p), 1e-9)) * invT);
  let cand = logits.map((_, i) => i);
  const k = Math.max(1, Math.min(Math.floor(topK) || n, n));
  if (k < n) cand = cand.sort((a, b) => logits[b] - logits[a]).slice(0, k);
  let max = -Infinity;
  for (const i of cand) if (logits[i] > max) max = logits[i];
  let sum = 0; const weights = cand.map((i) => { const w = Math.exp(logits[i] - max); sum += w; return w; });
  if (!(sum > 0)) return best;
  let r = Math.random() * sum;
  for (let j = 0; j < cand.length; j++) { r -= weights[j]; if (r <= 0) return cand[j]; }
  return cand[cand.length - 1];
}

function polish(text) {
  if (!settings.tidy) return text;
  let t = text.replace(/\s+/g, " ").trim();
  if (!t) return t;
  t = t.replace(/^([^a-zA-Z]*)([a-z])/, (_, p, c) => p + c.toUpperCase());
  t = t.replace(/([.!?]\s+)([a-z])/g, (_, p, c) => p + c.toUpperCase());
  if (!/[.!?…]$/.test(t)) t += ".";
  return t;
}

function traceStep(probs, temperature, topK, chosen, tok) {
  const n = probs.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
  const greedy = !Number.isFinite(temperature) || temperature <= 0.01;
  const k = greedy ? 1 : Math.max(1, Math.min(Math.floor(topK) || n, n));
  const invT = greedy ? 1 : 1 / temperature;
  const pool = order.slice(0, k);
  const logits = pool.map((i) => Math.log(Math.max(Number(probs[i]), 1e-9)) * invT);
  const max = Math.max(...logits);
  const w = logits.map((l) => Math.exp(l - max));
  const sum = w.reduce((a, b) => a + b, 0);
  const drawn = new Map(pool.map((i, j) => [i, w[j] / sum]));
  const ids = order.slice(0, 8);
  if (!ids.includes(chosen)) ids.push(chosen);
  return { t: tok, c: ids.map((i) => [model.detokenize([i]), Number(probs[i]), drawn.get(i) || 0, i === chosen ? 1 : 0]) };
}

async function generateRaw(prompt, ctrl, onPartial, trace) {
  let ctx = prompt;
  let out = "";
  const session = typeof model.createGenerationSession === "function"
    ? model.createGenerationSession(prompt)
    : null;
  for (let step = 0; step < settings.maxTokens; step++) {
    if (ctrl.stopped) break;
    const res = session ? session.predict() : model.predict(ctx);
    const idx = sampleToken(res.raw, settings.temperature, settings.topK);
    const tok = model.detokenize([idx]);
    if (tok === "\n" || tok === "") break;
    if (trace) trace.push(traceStep(res.raw, settings.temperature, settings.topK, idx, tok));
    out += tok;
    if (session) session.push(idx);
    else ctx += tok;
    if (onPartial) onPartial(out);
    if (settings.stream) {
      const delay = session ? (out.length < 3 ? 20 : 2) : (out.length < 3 ? 60 : 14);
      await sleep(delay);
    } else if (step % 10 === 9) await sleep(0);  // keep UI responsive
  }
  return out;
}

/* ─────────────────────── generation flow ─────────────────────── */
function setComposerGenerating(on) {
  $("#sendBtn").hidden = on;
  $("#stopBtn").hidden = !on;
  $("#composerInput").placeholder = on ? `${MODELS[modelId].label} is responding…` : "Ask anything";
}
function stopGeneration() { if (generating) generating.stopped = true; }
$("#stopBtn").addEventListener("click", stopGeneration);

async function runGeneration() {
  const chat = activeChat();
  if (!chat || !modelReady || generating) return;
  const prompt = buildPrompt(chat);

  // placeholder assistant element
  const idx = chat.messages.length;
  const placeholder = { role: "assistant", content: "", ts: Date.now(), pending: true };
  chat.messages.push(placeholder);
  showEmptyOrThread();
  const el = buildMessageEl({ role: "assistant", content: "" }, idx);
  const body = document.createElement("div");
  body.className = "msg-body";
  body.innerHTML = `<span class="thinking" aria-label="Thinking"><i></i><i></i><i></i></span>`;
  el.innerHTML = ""; el.appendChild(body);
  threadEl.appendChild(el);
  refreshIcons(); scrollToBottom(true);

  const ctrl = { stopped: false, chatId: chat.id };
  generating = ctrl;
  setComposerGenerating(true);

  const caret = document.createElement("span"); caret.className = "stream-caret";
  let shown = "";
  const paint = (raw) => {
    shown = settings.stream ? raw : shown;
    body.textContent = settings.tidy ? polish(raw) : raw;
    if (settings.stream && !ctrl.stopped) body.appendChild(caret);
    if (stickToBottom) scrollToBottom(false);
  };

  let raw = "";
  const steps = [];
  const traceMeta = { model: modelId, temperature: settings.temperature, topK: settings.topK };
  try {
    await sleep(model.createGenerationSession ? 60 : 280); // skip most of the artificial delay on the fast decoder
    for (let attempt = 0; attempt < 3; attempt++) {
      steps.length = 0;
      raw = await generateRaw(prompt, ctrl, settings.stream ? paint : null, steps);
      if (raw.trim() || ctrl.stopped) break;   // empty = immediate EOS; resample
      body.innerHTML = `<span class="thinking" aria-label="Thinking"><i></i><i></i><i></i></span>`;
      await sleep(200);
    }
  } catch (err) {
    console.error(err);
    raw = "";
    toast("Generation hit an error — please try again", { icon: "circle-alert" });
  }

  generating = null;
  setComposerGenerating(false);

  const finalText = polish(raw);
  placeholder.pending = false;
  placeholder.stopped = ctrl.stopped || undefined;
  if (finalText) {
    placeholder.content = finalText;
    if (steps.length) placeholder.trace = { ...traceMeta, steps: [...steps] };
  } else {
    placeholder.notice = true;
    placeholder.content = ctrl.stopped
      ? `Generation was stopped before ${MODELS[modelId].label} produced any text.`
      : `${MODELS[modelId].label} ended its reply before producing any text (it predicted its end-of-sequence token right away). Try Regenerate, or raise the temperature in Settings → Model.`;
  }
  chat.updatedAt = Date.now();
  saveChats(); renderChatList(); renderThread(true);
  $("#composerInput").focus();
}

function regenerate(assistantIdx) {
  const chat = activeChat();
  if (!chat || generating) return;
  if (switchingModel) { toast("Model is still loading — one moment", { icon: "loader" }); return; }
  if (window.speechSynthesis?.speaking) window.speechSynthesis.cancel();
  // remove that assistant reply, then generate a fresh one in its place
  chat.messages.splice(assistantIdx, 1);
  saveChats();
  // if the following message was a user turn, drop everything after the target too
  renderThread(false);
  runGeneration();
}

function sendMessage(text) {
  if (switchingModel) { toast("Model is still loading — one moment", { icon: "loader" }); return; }
  const chat = ensureChat();
  const attachment = pendingAttachment;
  let content = text.trim();
  if (attachment) {
    content = (content ? content + "\n\n" : "") + attachment.text;
    clearAttachment();
  }
  if (!content || generating) return;
  chat.messages.push({ role: "user", content, ts: Date.now() });
  if (chat.messages.filter((m) => m.role === "user").length === 1) {
    chat.title = (text.trim() || attachment?.name || "New chat").replace(/\s+/g, " ").slice(0, 42) || "New chat";
  }
  chat.updatedAt = Date.now();
  saveChats(); renderChatList(); renderThread(true);
  runGeneration();
}

/* ───────────────────────── composer ───────────────────────── */
const input = $("#composerInput");
function autoresizeComposer() {
  input.style.height = "auto";
  input.style.height = Math.min(200, input.scrollHeight) + "px";
  updateSendBtn();
}
function updateSendBtn() {
  $("#sendBtn").disabled = !(input.value.trim() || pendingAttachment) || !!generating || switchingModel;
}
input.addEventListener("input", autoresizeComposer);
input.addEventListener("compositionend", autoresizeComposer);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submitComposer();
  }
});
function submitComposer() {
  const text = input.value;
  if (!text.trim() && !pendingAttachment) return;
  if (switchingModel) { toast("Model is still loading — one moment", { icon: "loader" }); return; }
  input.value = ""; autoresizeComposer();
  sendMessage(text);
}
$("#sendBtn").addEventListener("click", submitComposer);
input.addEventListener("paste", () => setTimeout(autoresizeComposer, 0));

/* attach menu */
const attachMenu = $("#attachMenu");
$("#attachBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  const open = attachMenu.hidden;
  closeAllMenus();
  attachMenu.hidden = !open;
  $("#attachBtn").setAttribute("aria-expanded", String(open));
  refreshIcons();
});
$("#menuUpload").addEventListener("click", () => { closeAllMenus(); $("#fileInput").click(); });
$("#menuSurprise").addEventListener("click", () => {
  closeAllMenus();
  const ideas = [
    "How are you?", "What is your name?", "What is a computer?", "Do you like music?",
    "What is the weather like?", "What makes you happy?", "Can you help me learn?",
    "What is your favourite colour?", "Tell me about the sun", "What is a neural network?",
    "Good morning", "Tell me a short story",
  ];
  input.value = "";
  sendMessage(ideas[Math.floor(Math.random() * ideas.length)]);
});
$("#menuClearChat").addEventListener("click", async () => {
  closeAllMenus();
  const chat = activeChat();
  if (!chat || chat.messages.length === 0) { toast("This chat is already empty", { icon: "eraser" }); return; }
  if (await confirmDialog({ title: "Clear this chat?", text: "All messages in this conversation will be removed. This can't be undone.", okLabel: "Clear" })) {
    chat.messages = []; chat.title = "New chat"; chat.updatedAt = Date.now();
    saveChats(); renderChatList(); renderThread();
    toast("Chat cleared", { icon: "eraser" });
  }
});
$("#fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (file) readTextFile(file);
});
async function readTextFile(file) {
  if (file.size > 2_000_000) { toast("That file is too large (max 2 MB)", { icon: "file-warning" }); return; }
  try {
    const text = await file.text();
    pendingAttachment = { name: file.name, text: text.slice(0, 12000) };
    $("#attachmentName").textContent = `${file.name} · ${text.length.toLocaleString()} chars`;
    $("#attachmentChip").hidden = false;
    refreshIcons(); updateSendBtn(); input.focus();
    toast("File attached — add a message or just press send", { icon: "paperclip" });
  } catch { toast("Couldn't read that file", { icon: "file-warning" }); }
}
function clearAttachment() {
  pendingAttachment = null;
  $("#attachmentChip").hidden = true;
  updateSendBtn();
}
$("#attachmentRemove").addEventListener("click", clearAttachment);

/* drag & drop text files */
["dragenter", "dragover"].forEach((ev) => window.addEventListener(ev, (e) => {
  e.preventDefault();
  if (e.dataTransfer?.types?.includes("Files")) $("#composer").classList.add("dragging");
}));
["dragleave", "drop"].forEach((ev) => window.addEventListener(ev, (e) => {
  e.preventDefault();
  if (ev === "drop" || !e.relatedTarget) $("#composer").classList.remove("dragging");
}));
window.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) readTextFile(file);
});

/* voice input */
const micBtn = $("#micBtn");
let recognition = null;
micBtn.addEventListener("click", () => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { toast("Voice input isn't supported in this browser", { icon: "mic-off" }); return; }
  if (recognition) { recognition.stop(); return; }
  recognition = new SR();
  recognition.lang = navigator.language || "en-US";
  recognition.interimResults = true;
  const base = input.value;
  recognition.onresult = (ev) => {
    let txt = "";
    for (const r of ev.results) txt += r[0].transcript;
    input.value = (base ? base.replace(/\s*$/, " ") : "") + txt;
    autoresizeComposer();
  };
  recognition.onend = recognition.onerror = () => {
    recognition = null;
    micBtn.classList.remove("recording");
    micBtn.innerHTML = `<i data-lucide="mic"></i>`; refreshIcons();
    input.focus();
  };
  micBtn.classList.add("recording");
  micBtn.innerHTML = `<i data-lucide="square"></i>`; refreshIcons();
  try { recognition.start(); toast("Listening… speak now", { icon: "mic" }); }
  catch { recognition = null; micBtn.classList.remove("recording"); }
});

/* ───────────────────── header menus ───────────────────── */
const modelMenu = $("#modelMenu");
$("#modelMenuBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  const open = modelMenu.hidden;
  closeAllMenus();
  modelMenu.hidden = !open;
  $("#modelMenuBtn").setAttribute("aria-expanded", String(open));
  refreshIcons();
});
$("#menuModelSettings").addEventListener("click", () => { closeAllMenus(); openSettings("model"); });
$("#menuVerify").addEventListener("click", async () => { closeAllMenus(); openSettings("model"); await runVerify(); });
$("#modelCardBtn").addEventListener("click", () => openSettings("model"));
$("#profileBtn").addEventListener("click", () => openSettings("general"));
$("#shareBtn").addEventListener("click", () => shareChat(activeChat()));

async function shareChat(chat) {
  if (!chat || chat.messages.length === 0) { toast("Nothing to share yet — send a message first", { icon: "share" }); return; }
  const lines = [`# ${chat.title || "V0idGPT chat"}`, ""];
  for (const m of chat.messages) {
    if (m.notice) continue;
    lines.push(m.role === "user" ? `**You:** ${m.content}` : `**V0idGPT:** ${m.content}`, "");
  }
  lines.push("— Shared from V0idGPT (runs 100% locally in the browser)");
  const text = lines.join("\n");
  if (navigator.share) {
    try { await navigator.share({ title: chat.title || "V0idGPT chat", text }); return; }
    catch (e) { if (e?.name === "AbortError") return; }
  }
  if (await copyText(text)) toast("Chat copied to clipboard", { icon: "clipboard-check" });
  else toast("Couldn't copy the chat", { icon: "circle-alert" });
}

function closeAllMenus() {
  closeTokenPop();
  modelMenu.hidden = true;
  attachMenu.hidden = true;
  closeChatMenu();
  $("#modelMenuBtn").setAttribute("aria-expanded", "false");
  $("#attachBtn").setAttribute("aria-expanded", "false");
}
document.addEventListener("click", (e) => {
  if (!e.target.closest(".dropdown") && !e.target.closest("#attachBtn") && !e.target.closest("#modelMenuBtn") && !e.target.closest(".chat-item-menu-btn")) closeAllMenus();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (generating) { stopGeneration(); return; }
    closeAllMenus();
    if (!$("#settingsOverlay").hidden) closeSettings();
    if (!$("#confirmOverlay").hidden) closeConfirm(false);
  }
});

/* ─────────────────────── settings modal ─────────────────────── */
function openSettings(tab = "general") {
  $("#settingsOverlay").hidden = false;
  switchTab(tab);
  bindSettingsUI();
  refreshIcons();
}
function closeSettings() { $("#settingsOverlay").hidden = true; }
$("#settingsClose").addEventListener("click", closeSettings);
$("#settingsOverlay").addEventListener("click", (e) => { if (e.target.id === "settingsOverlay") closeSettings(); });
function switchTab(tab) {
  $$(".settings-tab").forEach((b) => {
    const on = b.dataset.tab === tab;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
  });
  $$(".settings-panel").forEach((p) => { p.hidden = p.dataset.panel !== tab; });
}
$$(".settings-tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

function bindSwitch(id, key) {
  const el = $(id);
  el.setAttribute("aria-checked", String(!!settings[key]));
  el.onclick = () => {
    settings[key] = !settings[key];
    el.setAttribute("aria-checked", String(settings[key]));
    saveSettings();
  };
}
function bindSettingsUI() {
  if (model) {
    $("#topKSlider").max = model.info.tokenizer.vocabSize;
    if (settings.topK > model.info.tokenizer.vocabSize) { settings.topK = model.info.tokenizer.vocabSize; saveSettings(); }
  }
  bindSwitch("#toggleStream", "stream");
  bindSwitch("#toggleTidy", "tidy");
  const bindSlider = (slider, val, key, fmt, parse = Number) => {
    const s = $(slider), v = $(val);
    s.value = settings[key]; v.textContent = fmt(settings[key]);
    s.oninput = () => { settings[key] = parse(s.value); v.textContent = fmt(settings[key]); saveSettings(); renderPresets(); };
  };
  bindSlider("#tempSlider", "#tempVal", "temperature", (x) => Number(x).toFixed(2));
  bindSlider("#topKSlider", "#topKVal", "topK", (x) => String(Math.round(x)), (x) => Math.round(Number(x)));
  bindSlider("#maxTokSlider", "#maxTokVal", "maxTokens", (x) => `${Math.round(x)} ${MODELS[modelId || settings.model].unit}`, (x) => Math.round(Number(x)));
  renderPresets();
  applyTheme();
}
function renderPresets() {
  const presets = MODELS[modelId || settings.model].presets;
  $$("#presetSegmented button").forEach((b) => {
    const p = presets[b.dataset.preset];
    const on = p.temperature === settings.temperature && p.topK === settings.topK && p.maxTokens === settings.maxTokens;
    b.classList.toggle("active", on);
  });
}
$$("#presetSegmented button").forEach((b) => b.addEventListener("click", () => {
  const p = MODELS[modelId].presets[b.dataset.preset];
  settings.temperature = p.temperature;
  settings.topK = p.topK;
  settings.maxTokens = p.maxTokens;
  saveSettings(); bindSettingsUI();
}));
$("#resetSamplingBtn").addEventListener("click", () => {
  const d = MODELS[modelId].presets.balanced;
  settings.temperature = d.temperature;
  settings.topK = d.topK;
  settings.maxTokens = d.maxTokens;
  saveSettings(); bindSettingsUI();
  toast("Generation settings reset", { icon: "rotate-ccw" });
});

/* verify */
async function runVerify() {
  const box = $("#verifyResult");
  box.hidden = false;
  box.className = "verify-result";
  box.textContent = "Running NeuronForge test vectors…";
  await sleep(30);
  try {
    const report = model.verify();
    const passed = report.checks.filter((c) => c.status === "pass").length;
    box.className = "verify-result " + (report.ok ? "ok" : "fail");
    box.innerHTML = report.ok
      ? `<b>✓ Model verified.</b> All ${passed} checks passed — the weights in your browser match NeuronForge's reference exactly (runtime v${esc(report.runtimeVersion)}).`
      : `<b>Verification reported issues.</b> ${report.checks.filter((c) => c.status !== "pass").map((c) => esc(c.name + ": " + c.detail)).join("; ")}`;
  } catch (err) {
    box.className = "verify-result fail";
    box.textContent = "Verification failed: " + err.message;
  }
}
$("#verifyBtn").addEventListener("click", runVerify);

/* data: export / import / clear */
$("#exportBtn").addEventListener("click", () => {
  const payload = JSON.stringify({ app: "V0idGPT", version: 1, exportedAt: new Date().toISOString(), chats }, dropTrace, 2);
  const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url; a.download = "v0idgpt-chats.json"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  toast("Chats exported", { icon: "download" });
});
$("#importBtn").addEventListener("click", () => $("#importInput").click());
$("#importInput").addEventListener("change", async (e) => {
  const file = e.target.files[0]; e.target.value = "";
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    const incoming = Array.isArray(data) ? data : data.chats;
    if (!Array.isArray(incoming)) throw new Error("bad format");
    let added = 0;
    for (const c of incoming) {
      if (!c || !Array.isArray(c.messages)) continue;
      chats.unshift({
        id: uid(),
        title: String(c.title || "Imported chat").slice(0, 80),
        messages: c.messages.filter((m) => m && (m.role === "user" || m.role === "assistant")).map((m) => ({ role: m.role, content: String(m.content || ""), ts: Number(m.ts) || Date.now(), feedback: m.feedback || null, notice: !!m.notice })),
        createdAt: Number(c.createdAt) || Date.now(),
        updatedAt: Number(c.updatedAt) || Date.now(),
      });
      added++;
    }
    chats.sort((a, b) => b.updatedAt - a.updatedAt);
    if (chats[0]) { activeId = chats[0].id; localStorage.setItem(LS_ACTIVE, activeId); }
    saveChats(); renderChatList(); renderThread(false);
    toast(added ? `Imported ${added} chat${added === 1 ? "" : "s"}` : "No chats found in that file", { icon: "upload" });
  } catch { toast("That file isn't a valid V0idGPT export", { icon: "file-warning" }); }
});
$("#clearAllBtn").addEventListener("click", async () => {
  if (!chats.length) { toast("There are no chats to delete", { icon: "trash-2" }); return; }
  if (await confirmDialog({ title: "Delete all chats?", text: `This will permanently delete ${chats.length} conversation${chats.length === 1 ? "" : "s"} stored on this device.`, okLabel: "Delete all" })) {
    chats = []; activeId = null; localStorage.removeItem(LS_ACTIVE);
    saveChats(); renderChatList(); renderThread();
    closeSettings();
    toast("All chats deleted", { icon: "trash-2" });
  }
});

/* ─────────────────────── suggestions ─────────────────────── */
const SUGGESTIONS = [
  { icon: "heart", text: "How are you?" },
  { icon: "sparkles", text: "What is your name?" },
  { icon: "cpu", text: "What is a computer?" },
  { icon: "music", text: "Do you like music?" },
];
$("#suggestions").innerHTML = SUGGESTIONS.map((s) =>
  `<button class="suggestion" role="listitem" data-text="${esc(s.text)}"><i data-lucide="${s.icon}"></i><span>${esc(s.text)}</span></button>`
).join("");
$$(".suggestion").forEach((b) => b.addEventListener("click", () => { if (!generating) sendMessage(b.dataset.text); }));

/* ─────────────────────── keyboard shortcuts ─────────────────────── */
document.addEventListener("keydown", (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
  else if (mod && e.shiftKey && e.key.toLowerCase() === "s") { e.preventDefault(); toggleSidebar(); }
  else if (mod && e.key.toLowerCase() === "k") {
    e.preventDefault();
    if (window.innerWidth <= 820 || $("#sidebar").classList.contains("closed")) setSidebar(true);
    if ($("#searchBox").hidden) toggleSearch(); else $("#searchInput").focus();
  }
});

/* ─────────────────────── model info UI ─────────────────────── */
function renderModelPickers() {
  const seg = $("#modelSegmented");
  seg.innerHTML = Object.entries(MODELS).map(([id, m]) =>
    `<button data-model-id="${esc(id)}" class="${id === modelId ? "active" : ""}">${esc(m.label)}</button>`
  ).join("");
  $$("#modelSegmented button").forEach((b) => b.addEventListener("click", () => switchModel(b.dataset.modelId)));
  const list = $("#modelList");
  list.innerHTML = Object.entries(MODELS).map(([id, m]) =>
    `<button class="dropdown-item${id === modelId ? " active" : ""}" role="menuitem" data-model-id="${esc(id)}"><span><span>${esc(m.label)}</span><small>${esc(m.blurb)}</small></span>${id === modelId ? '<i data-lucide="check" class="model-list-check"></i>' : ""}</button>`
  ).join("");
  $$("#modelList .dropdown-item").forEach((b) => b.addEventListener("click", () => { closeAllMenus(); switchModel(b.dataset.modelId); }));
  refreshIcons();
}
function applyModelLabels() {
  const m = MODELS[modelId];
  $("#modelMenuLabel").textContent = m.label;
  $("#modelMenuName").textContent = m.label;
  $("#modelCardName").textContent = m.label;
  $("#maxTokUnit").textContent = m.unitLong;
  $("#topKVocab").textContent = model.info.tokenizer.vocabSize;
  $("#disclaimer").textContent = `${m.label} is a small ${model.info.paramCount.toLocaleString()}-parameter model running in your browser. Replies are generated from learned weights and can still be incorrect or miss context.`;
  $("#composerInput").setAttribute("aria-label", `Message ${m.label}`);
  document.title = m.label;
}
async function loadModelById(id, onStage = () => {}) {
  if (!loadedModels[id]) {
    onStage(25, `Downloading ${MODELS[id].label}…`);
    const mod = await MODELS[id].load();
    onStage(60, `Building ${MODELS[id].label}…`);
    await sleep(0);
    const loaded = await mod.loadModel();
    onStage(85, "Checking the model…");
    await sleep(0);
    const smokePrompt = MODELS[id].prompt === "chat" ? "user: hello\nassistant:" : (MODELS[id].prompt === "plain" ? "hello" : "q: hello? a:");
    const smoke = loaded.createGenerationSession
      ? loaded.createGenerationSession(smokePrompt).predict()
      : loaded.predict(smokePrompt);
    if (!smoke || !smoke.raw || smoke.raw.length !== loaded.info.tokenizer.vocabSize) throw new Error("Model smoke test failed");
    loadedModels[id] = loaded;
  }
  onStage(100, `${MODELS[id].label} ready`);
  return loadedModels[id];
}
function setSwitching(on) {
  switchingModel = on;
  $("#modelMenuBtn").disabled = on;
  $("#modelMenuSpinner").hidden = !on;
  $("#modelSegmented").classList.toggle("busy", on);
  $("#loadBar").hidden = !on;
  if (!on) $("#loadBarFill").style.width = "0%";
  if (!on) $("#composerInput").placeholder = "Ask anything";
  updateSendBtn();
}
function setLoadStage(pct, text) {
  $("#loadBarFill").style.width = `${pct}%`;
  $("#modelStatusText").textContent = text;
  $("#modelStatusDot").className = "status-dot";
  $("#composerInput").placeholder = text;
}
async function switchModel(id) {
  if (!MODELS[id] || id === modelId || switchingModel) return;
  if (generating) { stopGeneration(); await sleep(60); }
  closeAllMenus();
  closeTokenPop();
  saveSettings();
  const prevReady = modelReady;
  modelReady = false;
  setSwitching(true);
  setLoadStage(10, `Switching to ${MODELS[id].label}…`);
  const started = Date.now();
  try {
    const next = await loadModelById(id, setLoadStage);
    await sleep(Math.max(0, 450 - (Date.now() - started)));
    model = next; modelId = id;
    settings.model = id;
    applySampler(id);
    saveSettings();
    modelReady = true;
    fillModelInfo();
    $("#verifyResult").hidden = true;
    bindSettingsUI();
    setSwitching(false);
    toast(`Switched to ${MODELS[id].label}`, { icon: "cpu" });
  } catch (err) {
    console.error(err);
    modelReady = prevReady;
    $("#modelStatusText").textContent = "Ready · runs locally";
    $("#modelStatusDot").className = "status-dot ok";
    setSwitching(false);
    toast(`Couldn't load ${MODELS[id].label}`, { icon: "circle-alert", actionLabel: "Retry", onAction: () => switchModel(id), ms: 6000 });
  }
}
function fillModelInfo() {
  const info = model.info;
  const unit = MODELS[modelId].unit;
  const params = info.paramCount.toLocaleString();
  const layers = info.layers.map((l) => l.type).join(" → ");
  applyModelLabels();
  renderModelPickers();
  $("#modelSpecs").innerHTML = `
    <div class="spec"><b>${params}</b><span>parameters</span></div>
    <div class="spec"><b>${info.tokenizer.vocabSize} ${unit}</b><span>vocabulary</span></div>
    <div class="spec"><b>${info.tokenizer.seqLen} ${unit}</b><span>context window</span></div>
    <div class="spec"><b>v${esc(info.runtimeVersion)}</b><span>runtime</span></div>`;
  $("#modelMenuSub").textContent = `NeuronForge · ${params} params · runs locally`;
  $("#modelInfoCard").innerHTML = `
    <div class="mi-title"><i data-lucide="cpu"></i>${esc(info.name)} — the exact bundled model</div>
    <div class="mi-row"><span>Architecture</span><b>${esc(layers)}</b></div>
    <div class="mi-row"><span>Parameters</span><b>${params}</b></div>
    <div class="mi-row"><span>Tokenizer</span><b>${esc(info.tokenizer.level)}-level · ${info.tokenizer.vocabSize} tokens · ${info.tokenizer.seqLen}-${unit === "chars" ? "char" : "token"} context</b></div>
    <div class="mi-row"><span>Runtime</span><b>NeuronForge v${esc(info.runtimeVersion)}</b></div>
    <div class="mi-row"><span>Exported</span><b>${esc(new Date(info.exportedAt).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }))}</b></div>
    <div class="mi-row"><span>Where it runs</span><b>100% in your browser</b></div>`;
  $("#modelStatusText").textContent = "Ready · runs locally";
  $("#modelStatusDot").className = "status-dot ok";
}

/* ─────────────────────────── boot ─────────────────────────── */
async function boot() {
  applyTheme();
  setSidebar(window.innerWidth > 820);
  renderChatList();
  showEmptyOrThread();
  refreshIcons();
  autoresizeComposer();

  const splashStatus = $("#splashStatus");
  const splashBar = $("#splashBar");
  try {
    splashBar.style.width = "35%";
    splashStatus.textContent = "Loading model weights…";
    await sleep(60); // let the splash paint
    modelId = settings.model;
    applySampler(modelId);
    model = await loadModelById(modelId);
    splashBar.style.width = "72%";
    splashStatus.textContent = "Checking the model…";
    await sleep(30);
    modelReady = true;
    fillModelInfo();
    splashBar.style.width = "100%";
    splashStatus.textContent = "Ready";
    console.info(`[V0idGPT] ${model.info.name} loaded — ${model.info.paramCount} params, runtime v${model.info.runtimeVersion}. All inference is local.`);
  } catch (err) {
    console.error(err);
    splashStatus.textContent = "Failed to load the model. Check your connection and refresh.";
    $("#modelStatusText").textContent = "Failed to load";
    $("#modelStatusDot").className = "status-dot err";
    return;
  }
  await sleep(250);
  $("#splash").classList.add("gone");
  $("#app").hidden = false;
  setTimeout(() => $("#splash").remove(), 450);
  if (window.innerWidth > 820) input.focus();
}
window.addEventListener("resize", () => {
  if (window.innerWidth > 820) $("#sidebarScrim").classList.add("hidden");
});
boot();
