import { useState } from "react";

// Shown inside the chat window until the customer logs in.
export default function LoginForm({ onLogIn }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    const problem = await onLogIn(email.trim(), password);
    setLoading(false);
    if (problem) setError(problem);
  }

  return (
    <form onSubmit={submit} className="flex flex-1 flex-col justify-center gap-3 px-6">
      <div className="mb-2 text-center">
        <p className="font-semibold text-slate-900">Log in to chat with Robo</p>
        <p className="text-sm text-slate-500">So Robo can check your own orders and account.</p>
      </div>
      <input
        type="email"
        required
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
        autoComplete="email"
        className="rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
      />
      <input
        type="password"
        required
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        autoComplete="current-password"
        className="rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
      />
      {error && <p className="text-sm text-rose-500">{error}</p>}
      <button
        type="submit"
        disabled={loading}
        className="rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:bg-slate-300"
      >
        {loading ? "Logging in…" : "Log in"}
      </button>
    </form>
  );
}
