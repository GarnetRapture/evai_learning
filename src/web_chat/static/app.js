const LANGUAGE_LABELS = { ko: "한국어", en: "English", zh_tw: "繁體中文" };
const ROLE_LABELS = { system: "system", user: "구원자", assistant: "정령" };

const elements = {
  language: document.getElementById("language-select"),
  filter: document.getElementById("spirit-filter"),
  spirit: document.getElementById("spirit-select"),
  newSession: document.getElementById("new-session"),
  systemPrompt: document.getElementById("system-prompt"),
  title: document.getElementById("chat-title"),
  status: document.getElementById("chat-status"),
  messages: document.getElementById("message-list"),
  form: document.getElementById("message-form"),
  input: document.getElementById("message-input"),
  send: document.getElementById("send-button"),
};

const state = { spirits: [], sessionId: null, busy: false };

async function requestJson(path, body) {
  const response = await fetch(path, body === undefined
    ? { method: "GET" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error ?? response.statusText);
  }
  return payload;
}

function selectedSpirit() {
  return state.spirits.find((spirit) => spirit.slug === elements.spirit.value) ?? null;
}

function renderSpiritOptions() {
  const language = elements.language.value;
  const query = elements.filter.value.trim().toLowerCase();
  const previous = elements.spirit.value;
  elements.spirit.replaceChildren(
    ...state.spirits
      .filter((spirit) => !query
        || spirit.slug.includes(query)
        || Object.values(spirit.names).some((name) => name.toLowerCase().includes(query)))
      .map((spirit) => new Option(`${spirit.names[language]} · ${spirit.slug}`, spirit.slug)),
  );
  if ([...elements.spirit.options].some((option) => option.value === previous)) {
    elements.spirit.value = previous;
  }
  renderSystemPrompt();
}

function renderSystemPrompt() {
  const spirit = selectedSpirit();
  elements.systemPrompt.textContent = spirit ? spirit.system_prompt : "";
}

function renderMessages(messages) {
  elements.messages.replaceChildren(...messages.map((message) => {
    const item = document.createElement("li");
    item.className = `message message-${message.role}`;
    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = ROLE_LABELS[message.role] ?? message.role;
    item.append(role, document.createTextNode(message.content));
    return item;
  }));
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function appendError(text) {
  const item = document.createElement("li");
  item.className = "message message-error";
  item.textContent = text;
  elements.messages.append(item);
}

function setBusy(busy) {
  state.busy = busy;
  const ready = state.sessionId !== null && !busy;
  elements.input.disabled = !ready;
  elements.send.disabled = !ready;
  elements.newSession.disabled = busy;
  elements.status.textContent = busy ? "응답 생성 중…" : "";
}

async function closeSession() {
  if (state.sessionId !== null) {
    const sessionId = state.sessionId;
    state.sessionId = null;
    await requestJson("/api/sessions/close", { session_id: sessionId });
  }
}

async function startSession() {
  const spirit = selectedSpirit();
  if (spirit === null) {
    return;
  }
  setBusy(true);
  try {
    await closeSession();
    const session = await requestJson("/api/sessions", {
      spirit_id: spirit.slug,
      language: elements.language.value,
    });
    state.sessionId = session.session_id;
    elements.title.textContent = `${session.spirit_name} (${session.spirit_id}) · ${LANGUAGE_LABELS[session.language]}`;
    elements.systemPrompt.textContent = session.system_prompt;
    renderMessages(session.messages);
  } catch (error) {
    appendError(error.message);
  } finally {
    setBusy(false);
    elements.input.focus();
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (state.sessionId === null || message === "" || state.busy) {
    return;
  }
  elements.input.value = "";
  setBusy(true);
  try {
    const result = await requestJson("/api/chat", { session_id: state.sessionId, message });
    renderMessages(result.messages);
  } catch (error) {
    appendError(error.message);
  } finally {
    setBusy(false);
    elements.input.focus();
  }
}

function submitOnEnter(event) {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    elements.form.requestSubmit();
  }
}

async function initialize() {
  const catalog = await requestJson("/api/spirits");
  state.spirits = catalog.spirits;
  elements.language.replaceChildren(
    ...catalog.languages.map((language) => new Option(LANGUAGE_LABELS[language] ?? language, language)),
  );
  renderSpiritOptions();
  if (elements.spirit.options.length > 0) {
    elements.spirit.selectedIndex = 0;
    renderSystemPrompt();
  }
  elements.language.addEventListener("change", renderSpiritOptions);
  elements.filter.addEventListener("input", renderSpiritOptions);
  elements.spirit.addEventListener("change", renderSystemPrompt);
  elements.spirit.addEventListener("dblclick", startSession);
  elements.newSession.addEventListener("click", startSession);
  elements.form.addEventListener("submit", sendMessage);
  elements.input.addEventListener("keydown", submitOnEnter);
  setBusy(false);
}

initialize().catch((error) => appendError(error.message));
