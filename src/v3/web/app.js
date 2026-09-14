const elements = {
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#question"),
  send: document.querySelector("#send-button"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  clear: document.querySelector("#clear-button"),
  picker: document.querySelector("#conversation-picker"),
  debug: document.querySelector("#debug-toggle"),
  statusDot: document.querySelector("#status-dot"),
  statusLabel: document.querySelector("#status-label"),
  statusDetail: document.querySelector("#status-detail"),
  documentCount: document.querySelector("#document-count"),
  sourceTemplate: document.querySelector("#source-template"),
};

const HISTORY_KEY = "northstar-conversations-v1";
const ACTIVE_KEY = "northstar-active-conversation";
const MAX_CONVERSATIONS = 20;
const MAX_MESSAGES = 60;
const welcomeMarkup = elements.welcome.outerHTML;
const newSessionId = () => `session-${crypto.randomUUID()}`;
let waiting = false;
let memoryTurns = 8;

function newConversation() {
  const now = new Date().toISOString();
  return {
    id: newSessionId(),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: [],
    turns: [],
  };
}

function loadConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item) => item && typeof item.id === "string")
      .map((item) => ({
        ...item,
        title: item.title || "New conversation",
        messages: Array.isArray(item.messages) ? item.messages : [],
        turns: Array.isArray(item.turns) ? item.turns : [],
      }));
  } catch (_) {
    return [];
  }
}

let conversations = loadConversations();
let sessionId = null;
try {
  sessionId = localStorage.getItem(ACTIVE_KEY);
} catch (_) {
  // Start an in-memory conversation if browser storage is unavailable.
}
if (!conversations.some((item) => item.id === sessionId)) {
  const conversation = newConversation();
  conversations.unshift(conversation);
  sessionId = conversation.id;
}

function currentConversation() {
  return conversations.find((item) => item.id === sessionId);
}

function saveConversations() {
  conversations.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  conversations = conversations.slice(0, MAX_CONVERSATIONS);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(conversations));
    localStorage.setItem(ACTIVE_KEY, sessionId);
  } catch (_) {
    // The active conversation still works if browser storage is unavailable.
  }
  renderPicker();
}

function renderPicker() {
  elements.picker.replaceChildren();
  conversations.forEach((conversation) => {
    const option = document.createElement("option");
    option.value = conversation.id;
    option.textContent = conversation.title || "New conversation";
    option.selected = conversation.id === sessionId;
    elements.picker.append(option);
  });
}

function scrollToLatest() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 140)}px`;
}

function bindSuggestions() {
  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => submitQuestion(button.dataset.question));
  });
}

function showWelcome() {
  elements.messages.insertAdjacentHTML("beforeend", welcomeMarkup);
  elements.welcome = document.querySelector("#welcome");
  bindSuggestions();
}

function hideWelcome() {
  if (elements.welcome) {
    elements.welcome.remove();
    elements.welcome = null;
  }
}

function recordMessage(message) {
  const conversation = currentConversation();
  conversation.messages.push(message);
  conversation.messages = conversation.messages.slice(-MAX_MESSAGES);
  conversation.updatedAt = new Date().toISOString();
  if (message.role === "user" && conversation.title === "New conversation") {
    conversation.title = message.text.length > 46
      ? `${message.text.slice(0, 43)}…`
      : message.text;
  }
  saveConversations();
}

function addUserMessage(text, persist = true) {
  hideWelcome();
  const wrapper = document.createElement("article");
  wrapper.className = "message user";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "You";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  wrapper.append(label, bubble);
  elements.messages.append(wrapper);
  if (persist) recordMessage({ role: "user", text });
  scrollToLatest();
}

function addLoadingMessage() {
  const wrapper = document.createElement("article");
  wrapper.className = "message assistant";
  wrapper.id = "loading-message";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "Policy assistant";
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.setAttribute("aria-label", "Assistant is thinking");
  typing.innerHTML = "<i></i><i></i><i></i>";
  wrapper.append(label, typing);
  elements.messages.append(wrapper);
  scrollToLatest();
}

function removeLoadingMessage() {
  document.querySelector("#loading-message")?.remove();
}

function storedPayload(payload) {
  return {
    answer: payload.answer,
    citations: payload.citations || [],
    search_query: payload.search_query || "",
    response_type: payload.response_type || "rag",
  };
}

function addAssistantMessage(payload, isError = false, persist = true) {
  removeLoadingMessage();
  const wrapper = document.createElement("article");
  wrapper.className = `message assistant${isError ? " error" : ""}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = isError ? "Request error" : "Policy assistant";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  const copy = document.createElement("div");
  copy.className = "answer-copy";
  copy.textContent = payload.answer;
  bubble.append(copy);

  if (payload.citations?.length) {
    const sources = document.createElement("section");
    sources.className = "sources";
    const title = document.createElement("div");
    title.className = "sources-title";
    title.textContent = "Sources used";
    const grid = document.createElement("div");
    grid.className = "source-grid";
    payload.citations.forEach((citation) => {
      const card = elements.sourceTemplate.content.cloneNode(true);
      card.querySelector(".source-number").textContent = citation.number;
      card.querySelector(".source-name").textContent = citation.source;
      const policy = citation.policy_id ? `${citation.policy_id} · ` : "";
      card.querySelector(".source-location").textContent = `${policy}Page ${citation.page} · ${citation.section}`;
      grid.append(card);
    });
    sources.append(title, grid);
    bubble.append(sources);
  }

  if (payload.retrieval?.length) {
    const details = document.createElement("details");
    details.className = "debug-box";
    const summary = document.createElement("summary");
    summary.textContent = `Retrieval trace · ${payload.retrieval.length} candidates`;
    const content = document.createElement("div");
    content.className = "debug-content";
    const lines = [`Search query: ${payload.search_query}`, ""];
    payload.retrieval.forEach((item) => {
      const distance = item.semantic_distance == null ? "n/a" : item.semantic_distance.toFixed(3);
      lines.push(`${item.rank}. ${item.source} · p.${item.page} · ${item.section}`);
      lines.push(`   cosine distance ${distance} · BM25 ${item.bm25_score.toFixed(2)} · fused ${item.fused_score.toFixed(4)}`);
    });
    content.textContent = lines.join("\n");
    details.append(summary, content);
    bubble.append(details);
  }
  wrapper.append(label, bubble);
  elements.messages.append(wrapper);
  if (persist) {
    recordMessage({ role: "assistant", payload: storedPayload(payload), isError });
  }
  scrollToLatest();
}

function renderConversation() {
  elements.messages.replaceChildren();
  elements.welcome = null;
  const conversation = currentConversation();
  if (!conversation.messages.length) {
    showWelcome();
    return;
  }
  conversation.messages.forEach((message) => {
    if (message.role === "user") addUserMessage(message.text, false);
    if (message.role === "assistant") {
      addAssistantMessage(message.payload, Boolean(message.isError), false);
    }
  });
}

function setWaiting(value) {
  waiting = value;
  elements.send.disabled = value;
  elements.input.disabled = value;
  elements.picker.disabled = value;
  elements.clear.disabled = value;
}

async function submitQuestion(question) {
  if (!question || waiting) return;
  const conversation = currentConversation();
  const history = conversation.turns.slice(-memoryTurns);
  addUserMessage(question);
  elements.input.value = "";
  resizeInput();
  setWaiting(true);
  addLoadingMessage();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: sessionId,
        history,
        debug: elements.debug.checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The assistant could not answer right now.");
    if ((payload.response_type || "rag") === "rag") {
      conversation.turns.push({
        question,
        search_query: payload.search_query.slice(0, 500),
        answer: payload.answer.slice(0, 4000),
      });
      conversation.turns = conversation.turns.slice(-memoryTurns);
    }
    addAssistantMessage(payload);
  } catch (error) {
    addAssistantMessage({ answer: error.message || "Unable to reach the RAG service." }, true);
  } finally {
    setWaiting(false);
    elements.input.focus();
  }
}

function startConversation() {
  if (waiting) return;
  const conversation = newConversation();
  conversations.unshift(conversation);
  sessionId = conversation.id;
  saveConversations();
  renderConversation();
  elements.input.focus();
}

function selectConversation(id) {
  if (waiting || !conversations.some((item) => item.id === id)) return;
  sessionId = id;
  try {
    localStorage.setItem(ACTIVE_KEY, sessionId);
  } catch (_) {
    // Selection still works for the current page.
  }
  renderPicker();
  renderConversation();
  elements.input.focus();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    const ready = health.status === "healthy" && health.indexed_chunks > 0;
    if (health.indexed_documents) elements.documentCount.textContent = `${health.indexed_documents}-policy`;
    if (health.memory_turns) memoryTurns = health.memory_turns;
    elements.statusDot.className = `status-dot${ready ? "" : " error"}`;
    elements.statusLabel.textContent = ready ? "Systems ready" : "Setup required";
    const provider = health.model_provider === "gemini" ? "Gemini" : "Ollama";
    elements.statusDetail.textContent = ready
      ? `${health.indexed_chunks} chunks · ${provider} connected`
      : health.indexed_chunks === 0 ? "Run Version 3 ingestion" : `Check ${provider} and Chroma`;
  } catch (_) {
    elements.statusDot.className = "status-dot error";
    elements.statusLabel.textContent = "API unavailable";
    elements.statusDetail.textContent = "Start the FastAPI server";
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(elements.input.value.trim());
});
elements.input.addEventListener("input", resizeInput);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.clear.addEventListener("click", startConversation);
elements.picker.addEventListener("change", (event) => selectConversation(event.target.value));

saveConversations();
renderConversation();
checkHealth();
elements.input.focus();
