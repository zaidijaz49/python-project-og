// Talks to the FastAPI backend (through the Vite proxy: /api/v1/... -> :8002/api/v1/...).

const TIMEOUT_MS = 60_000;

// Friendly messages the customer sees. Never show raw server errors.
const COMMON_ERRORS = {
  timeout: "Sorry, this is taking too long. Please try again in a moment.",
  network: "Sorry, I can't reach the support server. Please check your connection.",
};

const CHAT_ERRORS = {
  ...COMMON_ERRORS,
  400: "Sorry, I can't help with that request. Please ask me about your orders, your account or our store policies.",
  422: "Sorry, I couldn't read that message. Please type a short question and try again.",
  502: "Sorry, I couldn't finish that answer. Please try asking again.",
  503: "Sorry, I'm a little busy right now. Please try again in a minute.",
};

export class ApiError extends Error {
  constructor(kind, messages) {
    super(messages[kind] ?? messages[503]);
    this.kind = kind;
    // Only some problems are worth a "Retry" button. A blocked message is not.
    this.canRetry = kind !== 400 && kind !== 422;
  }
}

async function request(path, body, messages = CHAT_ERRORS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`/api/v1${path}`, {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!response.ok) throw new ApiError(response.status, messages);
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(error.name === "AbortError" ? "timeout" : "network", messages);
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => request("/health"),
  chat: (message) => request("/ai/chat", { message }, CHAT_ERRORS),
};
