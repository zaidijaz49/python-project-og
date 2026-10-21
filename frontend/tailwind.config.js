/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef4ff",
          100: "#dbe6fe",
          500: "#3b6cf6",
          600: "#2752e3",
          700: "#1f41b8",
        },
      },
      keyframes: {
        "pop-in": {
          "0%": { opacity: "0", transform: "translateY(12px) scale(0.96)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-4px)" },
        },
      },
      animation: {
        "pop-in": "pop-in 0.18s ease-out",
        float: "float 3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
