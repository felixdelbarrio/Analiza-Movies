import type { ReactNode } from "react";

import { useI18n } from "../i18n/provider";
import { SectionCard } from "./section-card";

interface PageStateProps {
  title: string;
  message: string;
  eyebrow?: string;
  action?: ReactNode;
}

export function PageState({ title, message, eyebrow, action }: PageStateProps) {
  const { t } = useI18n();
  return (
    <SectionCard actions={action} eyebrow={eyebrow ?? t("page_state.default_eyebrow")} title={title}>
      <div className="page-state">
        <p>{message}</p>
      </div>
    </SectionCard>
  );
}
