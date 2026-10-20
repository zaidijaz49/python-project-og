// A small friendly robot face, drawn with SVG so it stays sharp at any size.
export default function RobotIcon({ className = "h-8 w-8" }) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden>
      <line x1="32" y1="6" x2="32" y2="14" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <circle cx="32" cy="5" r="3.5" fill="#fbbf24" />
      <rect x="10" y="14" width="44" height="36" rx="12" fill="currentColor" />
      <rect x="16" y="22" width="32" height="18" rx="9" fill="#0f172a" opacity="0.85" />
      <circle cx="25" cy="31" r="3.5" fill="#67e8f9" />
      <circle cx="39" cy="31" r="3.5" fill="#67e8f9" />
      <path d="M27 44 q5 4 10 0" stroke="#0f172a" strokeWidth="2.5" fill="none" strokeLinecap="round" />
      <rect x="4" y="26" width="6" height="12" rx="3" fill="currentColor" />
      <rect x="54" y="26" width="6" height="12" rx="3" fill="currentColor" />
    </svg>
  );
}
