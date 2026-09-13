const elements = {
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#question"),
  send: document.querySelector("#send-button"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  clear: document.querySelector("#clear-button"),
  debug: document.querySelector("#debug-toggle"),
  statusDot: document.querySelector("#status-dot"),
  statusLabel: document.querySelector("#status-label"),
  statusDetail: document.querySelector("#status-detail"),
  documentCount: document.querySelector("#document-count"),
  sourceTemplate: document.querySelector("#source-template"),
};

const newSessionId = () => `session-${crypto.randomUUID()}`;
let sessionId = sessionStorage.getItem("northstar-session") || newSessionId();
sessionStorage.setItem("northstar-session", sessionId);
let waiting = false;

function scrollToLatest() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 140)}px`;
}

function hideWelcome() {
  if (elements.welcome) {
    elements.welcome.remove();
    elements.welcome = null;
  }
}

function addUserMessage(text) {
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

function addAssistantMessage(payload, isError = false) {
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
  scrollToLatest();
}

function setWaiting(value) {
  waiting = value;
  elements.send.disabled = value;
  elements.input.disabled = value;
}

async function submitQuestion(question) {
  if (!question || waiting) return;
  addUserMessage(question);
  elements.input.value = "";
  resizeInput();
  setWaiting(true);
  addLoadingMessage();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId, debug: elements.debug.checked }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The assistant could not answer right now.");
    addAssistantMessage(payload);
  } catch (error) {
    addAssistantMessage({ answer: error.message || "Unable to reach the RAG service." }, true);
  } finally {
    setWaiting(false);
    elements.input.focus();
  }
}

async function clearConversation() {
  if (waiting) return;
  try {
    await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  } catch (_) {
    // A new client ID still guarantees fresh context if the server is unavailable.
  }
  sessionId = newSessionId();
  sessionStorage.setItem("northstar-session", sessionId);
  window.location.reload();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    const ready = health.status === "healthy" && health.indexed_chunks > 0;
    if (health.indexed_documents) elements.documentCount.textContent = `${health.indexed_documents}-policy`;
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
elements.clear.addEventListener("click", clearConversation);
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => submitQuestion(button.dataset.question));
});

checkHealth();
elements.input.focus();
