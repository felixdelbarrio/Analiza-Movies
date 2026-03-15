import { X } from "lucide-react";
import { useEffect } from "react";
import { createPortal } from "react-dom";

import { useI18n } from "../i18n/provider";
import type { ReportRow } from "../lib/types";
import { MovieDetailPanel } from "./movie-detail-panel";

interface MovieDetailModalProps {
  open: boolean;
  row?: ReportRow | null;
  onClose: () => void;
}

export function MovieDetailModal({ open, row, onClose }: MovieDetailModalProps) {
  const { t } = useI18n();

  useEffect(() => {
    if (!open) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose, open]);

  if (!open || !row) {
    return null;
  }

  return createPortal(
    <div className="detail-modal" role="dialog" aria-modal="true">
      <button
        className="detail-modal__backdrop"
        onClick={onClose}
        type="button"
        aria-label={t("app.action.close")}
      />
      <div className="detail-modal__dialog">
        <button
          className="detail-modal__close"
          onClick={onClose}
          type="button"
          aria-label={t("app.action.close")}
        >
          <X size={18} />
        </button>
        <MovieDetailPanel row={row} variant="modal" />
      </div>
    </div>,
    document.body
  );
}
