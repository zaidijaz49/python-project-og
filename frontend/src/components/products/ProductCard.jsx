import { useState } from "react";
import { formatPrice } from "../../data/products";

const STATUS_STYLES = {
  Shipped: "bg-sky-100 text-sky-700",
  Pending: "bg-amber-100 text-amber-700",
  Delivered: "bg-emerald-100 text-emerald-700",
};

export default function ProductCard({ product, onAddToCart, onTrackOrder }) {
  const [added, setAdded] = useState(false);
  const { name, price, emoji, color, rating, order } = product;

  function handleAdd() {
    onAddToCart();
    setAdded(true);
    setTimeout(() => setAdded(false), 1200);
  }

  return (
    <article className="group flex flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition hover:-translate-y-0.5 hover:shadow-md">
      <div className={`relative grid h-44 place-items-center bg-gradient-to-br ${color}`}>
        <span className="text-6xl transition group-hover:scale-110" aria-hidden>
          {emoji}
        </span>
        {order && (
          <span
            className={`absolute left-3 top-3 rounded-full px-2.5 py-1 text-xs font-semibold ${STATUS_STYLES[order.status]}`}
          >
            Your order · {order.status}
          </span>
        )}
      </div>

      <div className="flex flex-1 flex-col gap-3 p-4">
        <div>
          <h3 className="font-semibold text-slate-900">{name}</h3>
          <p className="text-sm text-slate-500">★ {rating.toFixed(1)}</p>
        </div>
        <p className="text-lg font-bold text-slate-900">{formatPrice(price)}</p>

        <div className="mt-auto flex gap-2">
          <button
            onClick={handleAdd}
            className="flex-1 rounded-xl bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-700"
          >
            {added ? "Added ✓" : "Add to cart"}
          </button>
          {order && (
            <button
              onClick={() => onTrackOrder(order.id)}
              className="rounded-xl border border-brand-500 px-3 py-2 text-sm font-semibold text-brand-600 hover:bg-brand-50"
              title={`Ask Robo about order ${order.id}`}
            >
              Track {order.id}
            </button>
          )}
        </div>
      </div>
    </article>
  );
}
