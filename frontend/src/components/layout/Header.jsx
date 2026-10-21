export default function Header({ cartCount }) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4">
        <a href="#" className="flex items-center gap-2 text-xl font-bold text-slate-900">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand-600 text-white">
            T
          </span>
          TechMart
        </a>

        <nav className="hidden gap-6 text-sm font-medium text-slate-600 sm:flex">
          <a href="#products" className="hover:text-brand-600">Products</a>
          <a href="#" className="hover:text-brand-600">Deals</a>
          <a href="#" className="hover:text-brand-600">Support</a>
        </nav>

        <button
          className="relative rounded-full p-2 text-slate-700 hover:bg-slate-100"
          aria-label={`Cart, ${cartCount} items`}
        >
          <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 3h2l2.4 12.2a2 2 0 0 0 2 1.6h7.7a2 2 0 0 0 2-1.5L21 8H6" />
            <circle cx="10" cy="20" r="1.3" />
            <circle cx="17" cy="20" r="1.3" />
          </svg>
          {cartCount > 0 && (
            <span className="absolute -right-0.5 -top-0.5 grid h-5 min-w-5 place-items-center rounded-full bg-rose-500 px-1 text-xs font-semibold text-white">
              {cartCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
