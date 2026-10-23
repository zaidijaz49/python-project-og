// Store products. `order` links a product to an order in the backend's fake
// database (tools.py), so "Track order" can ask the assistant about it.

export const PRODUCTS = [
  {
    id: 1,
    name: "Wireless Headphones",
    price: 12999,
    emoji: "🎧",
    color: "from-sky-100 to-indigo-100",
    rating: 4.6,
    order: { id: "O105", status: "Shipped" },
  },
  {
    id: 2,
    name: "Phone Case",
    price: 1499,
    emoji: "📱",
    color: "from-rose-100 to-orange-100",
    rating: 4.2,
    order: { id: "O106", status: "Pending" },
  },
  {
    id: 3,
    name: "Laptop Stand",
    price: 4999,
    emoji: "💻",
    color: "from-emerald-100 to-teal-100",
    rating: 4.8,
    order: { id: "O107", status: "Delivered" },
  },
  {
    id: 4,
    name: "Smart Watch",
    price: 18999,
    emoji: "⌚",
    color: "from-violet-100 to-fuchsia-100",
    rating: 4.4,
  },
  {
    id: 5,
    name: "Bluetooth Speaker",
    price: 7499,
    emoji: "🔊",
    color: "from-amber-100 to-yellow-100",
    rating: 4.5,
  },
  {
    id: 6,
    name: "Mechanical Keyboard",
    price: 10999,
    emoji: "⌨️",
    color: "from-slate-100 to-zinc-200",
    rating: 4.7,
  },
  {
    id: 7,
    name: "USB-C Hub",
    price: 3499,
    emoji: "🔌",
    color: "from-cyan-100 to-sky-100",
    rating: 4.1,
  },
  {
    id: 8,
    name: "HD Webcam",
    price: 6999,
    emoji: "📷",
    color: "from-lime-100 to-green-100",
    rating: 4.3,
  },
];

export const QUICK_REPLIES = [
  "Where is my order O105?",
  "What plan am I on?",
  "How long does a refund take?",
  "What can you help me with?",
];

export function formatPrice(pkr) {
  return `Rs ${pkr.toLocaleString("en-PK")}`;
}
