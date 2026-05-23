/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Elastic brand-adjacent palette
        elastic: {
          ink:    "#0B1628",   // deep navy
          slate:  "#1E2A3A",
          teal:   "#00BFB3",   // signature teal
          pink:   "#FA744E",   // accent
          yellow: "#FEC514",   // highlight
          gray:   "#69707D",
        },
      },
      fontFamily: {
        sans:  ["Inter", "system-ui", "sans-serif"],
        mono:  ["JetBrains Mono", "ui-monospace", "monospace"],
        serif: ["Spectral", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
