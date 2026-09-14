/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        canopy: "#1f8a55",
        leaf: "#37a169",
        soil: "#7b5e3b",
        warning: "#b7791f",
        danger: "#c2410c",
        ink: "#17201b",
      },
      boxShadow: {
        soft: "0 12px 28px rgba(20, 83, 45, 0.08)",
      },
    },
  },
  plugins: [],
};
