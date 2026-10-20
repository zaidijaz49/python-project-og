import { useCallback, useState } from "react";
import ChatWidget from "./components/chat/ChatWidget";
import Footer from "./components/layout/Footer";
import Header from "./components/layout/Header";
import Hero from "./components/layout/Hero";
import ProductGrid from "./components/products/ProductGrid";
import ReviewModal from "./components/reviews/ReviewModal";
import { PRODUCTS } from "./data/products";
import { useChat } from "./hooks/useChat";
import { useReviews } from "./hooks/useReviews";

export default function App() {
  const chat = useChat();
  const { reviews, submitReview } = useReviews();
  const [chatOpen, setChatOpen] = useState(false);
  const [cartCount, setCartCount] = useState(0);
  const [reviewProduct, setReviewProduct] = useState(null); // product whose reviews are open

  const openChat = useCallback(() => setChatOpen(true), []);
  const closeChat = useCallback(() => setChatOpen(false), []);
  const closeReviews = useCallback(() => setReviewProduct(null), []);

  // "Track O105" on a product: open the chat and ask for the user.
  function trackOrder(orderId) {
    openChat();
    chat.send(`Where is my order ${orderId}?`);
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header cartCount={cartCount} />
      <main className="flex-1">
        <Hero onOpenChat={openChat} />
        <ProductGrid
          products={PRODUCTS}
          reviews={reviews}
          onAddToCart={() => setCartCount((n) => n + 1)}
          onTrackOrder={trackOrder}
          onOpenReviews={setReviewProduct}
        />
      </main>
      <Footer />

      {reviewProduct && (
        <ReviewModal
          product={reviewProduct}
          reviews={reviews[reviewProduct.id] ?? []}
          onSubmit={submitReview}
          onClose={closeReviews}
        />
      )}
      <ChatWidget chat={chat} open={chatOpen} onOpen={openChat} onClose={closeChat} />
    </div>
  );
}
