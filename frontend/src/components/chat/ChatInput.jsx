import { forwardRef, useState } from "react";
import { MAX_MESSAGE_LENGTH } from "../../hooks/useChat";

const ChatInput = forwardRef(function ChatInput({ onSend, loading }, ref) {
  const [text, setText] = useState("");
  const [hint, setHint] = useState("");

  const tooLong = text.length > MAX_MESSAGE_LENGTH;
  const canSend = !loading && text.trim().length > 0 && !tooLong;

  function submit() {
    const problem = onSend(text);
    if (problem) {
      setHint(problem);
      return;
    }
    setText("");
    setHint("");
  }

  function handleKeyDown(e) {
    // Enter sends, Shift+Enter makes a new line.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white p-3">
      <div className="flex items-end gap-2">
        <textarea
          ref={ref}
          rows={1}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setHint("");
          }}
          onKeyDown={handleKeyDown}
          placeholder={loading ? "Robo is answering…" : "Ask about an order or account…"}
          className="max-h-28 flex-1 resize-none rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm outline-none focus:border-brand-500 focus:bg-white focus:ring-2 focus:ring-brand-100"
          aria-label="Message"
        />
        <button
          onClick={submit}
          disabled={!canSend}
          className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-brand-600 text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          aria-label="Send message"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M13 6l6 6-6 6" />
          </svg>
        </button>
      </div>

      <div className="mt-1 flex justify-between px-1 text-[11px]">
        <span className="text-rose-500">{hint}</span>
        {text.length > MAX_MESSAGE_LENGTH * 0.8 && (
          <span className={tooLong ? "text-rose-500" : "text-slate-400"}>
            {text.length}/{MAX_MESSAGE_LENGTH}
          </span>
        )}
      </div>
    </div>
  );
});

export default ChatInput;
