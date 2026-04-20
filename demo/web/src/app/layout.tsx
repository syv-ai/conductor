import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Conductor Playground",
  description: "Visual playground for the conductor DAG engine",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
