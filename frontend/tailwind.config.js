/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        border: "var(--border)",
        primary: {
          DEFAULT: "var(--primary)",
          hover: "var(--primary-hover)",
        },
        card: "var(--card-bg)",
        sidebar: "var(--sidebar-bg)",
      },
      fontFamily: {
        sans: ["Outfit", "Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
