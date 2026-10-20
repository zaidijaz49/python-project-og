export const SENTIMENTS = {
  positive: { emoji: "😊", label: "Positive", style: "bg-emerald-100 text-emerald-700" },
  neutral: { emoji: "😐", label: "Neutral", style: "bg-slate-100 text-slate-600" },
  negative: { emoji: "😡", label: "Negative", style: "bg-rose-100 text-rose-700" },
};

export default function SentimentBadge({ sentiment }) {
  const { emoji, label, style } = SENTIMENTS[sentiment] ?? SENTIMENTS.neutral;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ${style}`}>
      {emoji} {label}
    </span>
  );
}
