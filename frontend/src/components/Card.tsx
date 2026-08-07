import type { ReactNode } from "react";

/**
 * The one raised surface in the app.
 *
 * Extracted because four places needed the same wrapper — the result card, the
 * intro block, the loading skeleton and the browse row — and the skeleton's
 * geometry was hand-matched to the result card, so restyling one silently
 * desynced the other. It is also what keeps the duplication gate at zero.
 */
type CardProps = {
  children: ReactNode;
  /** `flat` drops the shadow for rows in a dense list. */
  variant?: "raised" | "flat" | "dashed";
  className?: string;
  as?: "div" | "article" | "section";
};

const VARIANTS = {
  raised: "border-border bg-card shadow-sm transition hover:shadow-md",
  flat: "border-border bg-card",
  dashed: "border-dashed border-border bg-card/50",
} as const;

export function Card({ children, variant = "raised", className = "", as = "div" }: CardProps) {
  const Tag = as;
  return (
    <Tag className={`rounded-lg border p-4 sm:p-5 ${VARIANTS[variant]} ${className}`}>
      {children}
    </Tag>
  );
}
