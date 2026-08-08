import { SITE_TITLE } from "@ipp/shared";
import { useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";

/**
 * Keeping the browser tab honest now that the server writes a title into it.
 *
 * The server prerenders each page's own `<title>` (see `backend/src/lib/seo/`),
 * which is what a search result and a WhatsApp preview show. Nothing in the SPA
 * used to touch `document.title` — it did not have to, because the title was
 * one constant for the whole site. It is not any more: the visitor can navigate
 * away client-side and leave the tab advertising a page they left, which is
 * worse than the generic title it used to always show.
 *
 * So: whatever the server wrote for the page actually landed on stays (that
 * page's title is by definition correct, and re-deriving it in the browser
 * would only introduce a way for the two to disagree), and any client-side
 * navigation away from it falls back to the site title — except onto a sermon,
 * where `SermonPage` has the real one and sets it itself.
 */
export function useRouteTitle(): void {
  const { pathname } = useLocation();
  const landed = useRef(pathname);

  useEffect(() => {
    if (pathname === landed.current) return;
    if (pathname.startsWith("/sermao/")) return;
    document.title = SITE_TITLE;
  }, [pathname]);
}

/** Sets the tab to a page's own title once the page knows it. */
export function useDocumentTitle(title: string | null): void {
  useEffect(() => {
    if (title) document.title = title;
  }, [title]);
}
