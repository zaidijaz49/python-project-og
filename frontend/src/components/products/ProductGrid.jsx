import ProductCard from "./ProductCard";

export default function ProductGrid({ products, onAddToCart, onTrackOrder }) {
  return (
    <section id="products" className="mx-auto max-w-6xl px-4 py-12">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900">Popular products</h2>
        <p className="text-sm text-slate-500">
          Items marked “Your order” can be tracked with Robo.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {products.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            onAddToCart={onAddToCart}
            onTrackOrder={onTrackOrder}
          />
        ))}
      </div>
    </section>
  );
}
