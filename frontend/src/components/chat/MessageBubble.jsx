// Turns "**bold**" into <strong> safely (no HTML injection), in case the AI
// still uses a little Markdown.
function FormattedText({ text }) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{part}</span>
    ),
  );
}

const TOOL_LABELS = {
  check_order_status: "Checked order",
  get_user_account: "Checked account",
};

export default function MessageBubble({ message, onRetry, retryDisabled }) {
  const { role, text, tools, model, retryText } = message;

  if (role === "user") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm text-white">
          {text}
        </p>
      </div>
    );
  }

  if (role === "error") {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-rose-200 bg-rose-50 px-4 py-2.5 text-sm text-rose-700">
          <p>{text}</p>
          {retryText && (
            <button
              onClick={() => onRetry(message.id, retryText)}
              disabled={retryDisabled}
              className="mt-2 rounded-lg bg-white px-3 py-1 text-xs font-semibold text-rose-700 ring-1 ring-rose-200 hover:bg-rose-100 disabled:opacity-50"
            >
              ↻ Try again
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%]">
        <p className="whitespace-pre-wrap rounded-2xl rounded-bl-md bg-white px-4 py-2.5 text-sm text-slate-800 shadow-sm ring-1 ring-slate-200">
          <FormattedText text={text} />
        </p>
        {(tools?.length > 0 || model) && (
          <div className="mt-1 flex flex-wrap gap-1.5 px-1 text-[11px] text-slate-400">
            {tools?.map((tool, i) => (
              <span key={i} className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-500">
                ✓ {TOOL_LABELS[tool] ?? tool}
              </span>
            ))}
            {model && <span className="py-0.5">via {model}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
