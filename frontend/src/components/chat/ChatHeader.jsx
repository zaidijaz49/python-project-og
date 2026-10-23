import RobotIcon from "./RobotIcon";

const STATUS = {
  online: { dot: "bg-emerald-400", label: "Online" },
  offline: { dot: "bg-rose-400", label: "Offline" },
  checking: { dot: "bg-amber-300", label: "Connecting…" },
};

export default function ChatHeader({ status, email, onClear, onLogOut, onClose, disabled }) {
  const { dot, label } = STATUS[status];

  return (
    <div className="flex items-center gap-3 bg-gradient-to-r from-brand-600 to-indigo-600 px-4 py-3 text-white">
      <div className="grid h-10 w-10 place-items-center rounded-full bg-white/15">
        <RobotIcon className="h-7 w-7 text-white" />
      </div>
      <div className="flex-1">
        <p className="font-semibold leading-tight">Robo · TechMart Support</p>
        <p className="flex items-center gap-1.5 text-xs text-white/80">
          <span className={`h-2 w-2 rounded-full ${dot}`} />
          {email ? `${label} · ${email}` : label}
        </p>
      </div>
      {email && (
        <>
          <button
            onClick={onClear}
            disabled={disabled}
            className="rounded-lg px-2 py-1 text-xs text-white/80 hover:bg-white/10 disabled:opacity-40"
            title="Start a new chat"
          >
            New chat
          </button>
          <button
            onClick={onLogOut}
            disabled={disabled}
            className="rounded-lg px-2 py-1 text-xs text-white/80 hover:bg-white/10 disabled:opacity-40"
          >
            Log out
          </button>
        </>
      )}
      <button
        onClick={onClose}
        className="rounded-lg p-1.5 hover:bg-white/10"
        aria-label="Close chat"
      >
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
    </div>
  );
}
