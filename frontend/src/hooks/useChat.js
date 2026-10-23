import { useEffect, useState } from "react";
import { api } from "../api/client";

const STORAGE_KEY = "techmart-chat-v1";
export const MAX_MESSAGE_LENGTH = 2000; // same limit as the backend (ChatRequest)

const WELCOME = {
  id: "welcome",
  role: "bot",
  text: "Hi! I'm Robo, the TechMart assistant. I can track your orders, check your account and answer questions about refunds, shipping and payments. How can I help?",
};

function loadMessages() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return Array.isArray(saved) && saved.length ? saved : [WELCOME];
  } catch {
    return [WELCOME];
  }
}

function saveMessages(messages) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch {
    // Storage blocked (private mode etc.): the chat still works, just not saved.
  }
}

const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

// Checks we can do before sending, so the user gets instant feedback.
export function validateMessage(text) {
  const trimmed = text.trim();
  if (!trimmed) return "Please type a question first.";
  if (trimmed.length > MAX_MESSAGE_LENGTH)
    return `Please keep it under ${MAX_MESSAGE_LENGTH} characters.`;
  return null;
}

export function useChat() {
  const [messages, setMessages] = useState(loadMessages);
  const [loading, setLoading] = useState(false);

  useEffect(() => saveMessages(messages), [messages]);

  const add = (message) => setMessages((list) => [...list, { id: newId(), ...message }]);

  async function ask(text) {
    setLoading(true);
    try {
      const data = await api.chat(text);
      add({ role: "bot", text: data.answer, tools: data.tools_used, model: data.model_used });
    } catch (error) {
      add({ role: "error", text: error.message, retryText: error.canRetry ? text : null });
    } finally {
      setLoading(false);
    }
  }

  // Returns an error string if the message can't be sent, otherwise null.
  function send(text) {
    if (loading) return "Please wait for my answer first."; // one message at a time
    const problem = validateMessage(text);
    if (problem) return problem;

    const clean = text.trim();
    add({ role: "user", text: clean });
    ask(clean);
    return null;
  }

  function retry(messageId, text) {
    if (loading) return;
    setMessages((list) => list.filter((m) => m.id !== messageId)); // drop the error bubble
    ask(text);
  }

  function clear() {
    if (!loading) setMessages([WELCOME]);
  }

  // On logout: always forget the chat, so the next person can't read it.
  function reset() {
    setMessages([WELCOME]);
  }

  return { messages, loading, send, retry, clear, reset };
}
