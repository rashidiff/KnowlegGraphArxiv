import './globals.css';
import React from 'react';

export const metadata = {
  title: 'Agentic Research Navigator',
  description: 'Semantic citation knowledge graph traversal and discovery for AI agents research papers.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-screen w-screen bg-[#0B0F19] text-[#F1F5F9] font-sans antialiased overflow-hidden">
        {children}
      </body>
    </html>
  );
}
