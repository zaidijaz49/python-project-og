export default function QuickReplies({ options, onPick, disabled }) {
  return (
    <div className="flex gap-2 overflow-x-auto px-4 pb-2 pt-1">
      {options.map((option) => (
        <button
          key={option}
          onClick={() => onPick(option)}
          disabled={disabled}
          className="shrink-0 rounded-full border border-brand-100 bg-brand-50 px-3 py-1.5 text-xs font-medium text-brand-700 hover:bg-brand-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {option}
        </button>
      ))}
    </div>
  );
}
