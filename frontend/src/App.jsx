import { useCallback, useState } from "react";
import ChatWidget from "./components/chat/ChatWidget";
import Footer from "./components/layout/Footer";
import Header from "./components/layout/Header";
import Hero from "./components/layout/Hero";
import ProductGrid from "./components/products/ProductGrid";
import { PRODUCTS } from "./data/products";
import { useChat } from "./hooks/useChat";

export default function App() {
  const chat = useChat();
  const [chatOpen, setChatOpen] = useState(false);
  const [cartCount, setCartCount] = useState(0);

  const openChat = useCallback(() => setChatOpen(true), []);
  const closeChat = useCallback(() => setChatOpen(false), []);

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
          onAddToCart={() => setCartCount((n) => n + 1)}
          onTrackOrder={trackOrder}
        />
      </main>
      <Footer />

      <ChatWidget chat={chat} open={chatOpen} onOpen={openChat} onClose={closeChat} />
    </div>
  );
}
