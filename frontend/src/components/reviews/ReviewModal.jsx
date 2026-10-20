import { useEffect, useRef, useState } from "react";
import { MAX_REVIEW_LENGTH, countSentiments, validateReview } from "../../hooks/useReviews";
import SentimentBadge, { SENTIMENTS } from "./SentimentBadge";

function Tags({ tags }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {tags.map((tag) => (
        <span key={tag} className="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
          {tag}
        </span>
      ))}
    </div>
  );
}

// step: "writing" -> "loading" -> "done", or "error" (with Try again).
export default function ReviewModal({ product, reviews, onSubmit, onClose }) {
  const [text, setText] = useState("");
  const [step, setStep] = useState("writing");
  const [hint, setHint] = useState("");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const inputRef = useRef(null);

  const counts = countSentiments(reviews);

  useEffect(() => {
    inputRef.current?.focus();
    const onKey = (e) => e.key === "Escape" && step !== "loading" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, step]);

  async function submit() {
    const problem = validateReview(text);
    if (problem) return setHint(problem); // checked here, so no request is wasted

    setStep("loading");
    setError(null);
    try {
      setResult(await onSubmit(product.id, text));
      setText("");
      setStep("done");
    } catch (err) {
      setError(err);
      setStep("error");
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-slate-900/50 p-4"
      onClick={(e) => e.target === e.currentTarget && step !== "loading" && onClose()}
    >
      <div
        role="dialog"
        aria-label={`Reviews for ${product.name}`}
        className="flex max-h-[90vh] w-full max-w-lg animate-pop-in flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
      >
        {/* Header */}
        <div className="flex items-center gap-3 border-b border-slate-200 px-5 py-4">
          <span className="text-3xl" aria-hidden>{product.emoji}</span>
          <div className="flex-1">
            <h2 className="font-semibold text-slate-900">{product.name}</h2>
            <p className="text-xs text-slate-500">
              {Object.entries(counts)
                .map(([key, n]) => `${SENTIMENTS[key].emoji} ${n}`)
                .join("  ·  ")}
            </p>
          </div>
          <button
            onClick={onClose}
            disabled={step === "loading"}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 disabled:opacity-40"
            aria-label="Close reviews"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        <div className="space-y-5 overflow-y-auto px-5 py-4">
          {/* Write / result */}
          {step === "done" && result ? (
            <div className="space-y-2 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
              <p className="font-semibold text-emerald-800">✓ Thanks for your review!</p>
              <p className="text-sm text-slate-700">
                <span className="font-medium">AI summary:</span> {result.summary}
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <SentimentBadge sentiment={result.sentiment} />
                <Tags tags={result.tags} />
              </div>
              <button
                onClick={() => setStep("writing")}
                className="text-sm font-semibold text-brand-600 hover:underline"
              >
                Write another review
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <label htmlFor="review" className="text-sm font-medium text-slate-700">
                Write a review
              </label>
              <textarea
                id="review"
                ref={inputRef}
                rows={3}
                value={text}
                disabled={step === "loading"}
                onChange={(e) => {
                  setText(e.target.value);
                  setHint("");
                }}
                placeholder="What did you like or dislike?"
                className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm outline-none focus:border-brand-500 focus:bg-white focus:ring-2 focus:ring-brand-100 disabled:opacity-60"
              />
              <div className="flex items-center justify-between text-xs">
                <span className="text-rose-500">{hint}</span>
                <span className={text.length > MAX_REVIEW_LENGTH ? "text-rose-500" : "text-slate-400"}>
                  {text.length}/{MAX_REVIEW_LENGTH}
                </span>
              </div>

              {step === "error" && error && (
                <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                  {error.message}
                </div>
              )}

              <button
                onClick={submit}
                disabled={step === "loading"}
                className="w-full rounded-xl bg-brand-600 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-wait disabled:opacity-70"
              >
                {step === "loading"
                  ? "Analyzing your review…"
                  : step === "error" && error?.canRetry
                    ? "↻ Try again"
                    : "Submit review"}
              </button>
              <p className="text-center text-[11px] text-slate-400">
                Our AI reads your review to find its mood and topics.
              </p>
            </div>
          )}

          {/* Existing reviews */}
          <div>
            <h3 className="mb-2 text-sm font-semibold text-slate-900">Reviews ({reviews.length})</h3>
            {reviews.length === 0 ? (
              <p className="text-sm text-slate-500">No reviews yet. Be the first!</p>
            ) : (
              <ul className="space-y-3">
                {reviews.map((review) => (
                  <li key={review.id} className="rounded-xl border border-slate-200 p-3">
                    <div className="mb-1.5 flex items-start justify-between gap-2">
                      <p className="text-sm text-slate-800">“{review.text}”</p>
                      <SentimentBadge sentiment={review.sentiment} />
                    </div>
                    <Tags tags={review.tags} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
