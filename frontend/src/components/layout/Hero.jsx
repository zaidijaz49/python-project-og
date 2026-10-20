export default function Hero({ onOpenChat }) {
  return (
    <section className="bg-gradient-to-br from-brand-600 to-indigo-700 text-white">
      <div className="mx-auto flex max-w-6xl flex-col items-start gap-5 px-4 py-14 sm:py-20">
        <span className="rounded-full bg-white/15 px-3 py-1 text-xs font-medium">
          New season · Free delivery over Rs 5,000
        </span>
        <h1 className="max-w-xl text-3xl font-bold leading-tight sm:text-5xl">
          Tech you love, support that answers.
        </h1>
        <p className="max-w-lg text-white/80">
          Questions about an order or your account? Our assistant Robo checks it for you
          in seconds.
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href="#products"
            className="rounded-xl bg-white px-5 py-3 font-semibold text-brand-700 shadow-sm hover:bg-brand-50"
          >
            Shop now
          </a>
          <button
            onClick={onOpenChat}
            className="rounded-xl border border-white/40 px-5 py-3 font-semibold hover:bg-white/10"
          >
            Ask Robo 🤖
          </button>
        </div>
      </div>
    </section>
  );
}
