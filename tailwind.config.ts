import type { Config } from "tailwindcss";

export default {
  content: ["./frontend/index.html", "./frontend/src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        graphite: {
          950: "#05070b",
          900: "#09111b",
          850: "#101926",
          800: "#162131",
          700: "#223147",
        },
        accent: {
          cyan: "#3ecbff",
          blue: "#3b82f6",
          emerald: "#2fd29b",
          amber: "#ffbb55",
          red: "#ff5c7c",
        },
      },
      fontFamily: {
        display: ["Space Grotesk", "Segoe UI", "sans-serif"],
        sans: ["IBM Plex Sans", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(62,203,255,0.15), 0 20px 60px rgba(5,7,11,0.45)",
      },
    },
  },
  plugins: [],
} satisfies Config;

