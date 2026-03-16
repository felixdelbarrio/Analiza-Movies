import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useEffect } from "react";
import { createPortal } from "react-dom";

import { useI18n } from "../i18n/provider";
import { ApiError, fetchConsolidatedRecord } from "../lib/api";
import type { ReportRow } from "../lib/types";
import { MovieDetailPanel } from "./movie-detail-panel";

interface MovieDetailModalProps {
  open: boolean;
  row?: ReportRow | null;
  profileId?: string | null;
  onClose: () => void;
}

export function MovieDetailModal({
  open,
  row,
  profileId,
  onClose
}: MovieDetailModalProps) {
  const { t } = useI18n();
  const detailsQuery = useQuery({
    queryKey: [
      "movie-detail",
      profileId ?? "__default__",
      row?.imdb_id ?? "",
      row?.title ?? "",
      row?.year ?? ""
    ],
    queryFn: async () => {
      if (!row) {
        return null;
      }
      try {
        return await fetchConsolidatedRecord(row, profileId);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          return null;
        }
        throw error;
      }
    },
    enabled: open && Boolean(row),
    staleTime: 300_000,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 1
  });

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
        <MovieDetailPanel
          details={detailsQuery.data ?? null}
          detailsLoading={detailsQuery.isLoading}
          row={row}
          variant="modal"
        />
      </div>
    </div>,
    document.body
  );
}
