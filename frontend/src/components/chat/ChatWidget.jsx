import { useEffect, useRef, useState } from "react";
import { QUICK_REPLIES } from "../../data/products";
import { useServerStatus } from "../../hooks/useServerStatus";
import LoginForm from "../auth/LoginForm";
import ChatHeader from "./ChatHeader";
import ChatInput from "./ChatInput";
import MessageBubble from "./MessageBubble";
import QuickReplies from "./QuickReplies";
import RobotIcon from "./RobotIcon";
import TypingIndicator from "./TypingIndicator";

export default function ChatWidget({ chat, auth, open, onOpen, onClose }) {
  const { messages, loading, send, retry, clear } = chat;
  const status = useServerStatus(open);
  const [showGreeting, setShowGreeting] = useState(true);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  // Keep the newest message in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, open]);

  // Focus the input when the chat opens, and let Esc close it.
  useEffect(() => {
    if (!open) return;
    setShowGreeting(false);
    inputRef.current?.focus();
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const onlyWelcome = messages.length === 1;
  const loggedIn = Boolean(auth.session);

  return (
    <>
      {open && (
        <div
          role="dialog"
          aria-label="Support chat"
          className="fixed inset-0 z-50 flex animate-pop-in flex-col overflow-hidden bg-slate-50 sm:inset-auto sm:bottom-24 sm:right-6 sm:h-[600px] sm:max-h-[calc(100vh-8rem)] sm:w-[380px] sm:rounded-2xl sm:shadow-2xl sm:ring-1 sm:ring-slate-200"
        >
          <ChatHeader
            status={status}
            email={auth.email}
            onClear={clear}
            onLogOut={auth.logOut}
            onClose={onClose}
            disabled={loading}
          />

          {auth.checking ? (
            <div className="flex-1" />
          ) : !loggedIn ? (
            <LoginForm onLogIn={auth.logIn} />
          ) : (
            <>
              <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
                {messages.map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    onRetry={retry}
                    retryDisabled={loading}
                  />
                ))}
                {loading && <TypingIndicator />}
                <div ref={bottomRef} />
              </div>

              {onlyWelcome && !loading && (
                <QuickReplies options={QUICK_REPLIES} onPick={send} disabled={loading} />
              )}
              <ChatInput ref={inputRef} onSend={send} loading={loading} />
            </>
          )}
        </div>
      )}

      {/* Floating robot button */}
      <div className="fixed bottom-6 right-6 z-40 flex items-end gap-3">
        {showGreeting && !open && (
          <button
            onClick={onOpen}
            className="mb-2 hidden animate-pop-in rounded-2xl rounded-br-sm bg-white px-4 py-2 text-sm text-slate-700 shadow-lg ring-1 ring-slate-200 sm:block"
          >
            👋 Need help with an order?
          </button>
        )}
        <button
          onClick={open ? onClose : onOpen}
          className="relative grid h-16 w-16 place-items-center rounded-full bg-gradient-to-br from-brand-500 to-indigo-600 text-white shadow-xl transition hover:scale-105 focus:outline-none focus:ring-4 focus:ring-brand-100"
          aria-label={open ? "Close support chat" : "Open support chat"}
        >
          {open ? (
            <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" d="M6 9l6 6 6-6" />
            </svg>
          ) : (
            <RobotIcon className="h-10 w-10 animate-float text-white" />
          )}
          {!open && loading && (
            <span className="absolute right-1 top-1 h-3 w-3 animate-ping rounded-full bg-amber-300" />
          )}
        </button>
      </div>
    </>
  );
}
