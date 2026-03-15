import type { PropsWithChildren, ReactNode } from "react";

interface SectionCardProps extends PropsWithChildren {
  title: string;
  eyebrow?: string;
  actions?: ReactNode;
  className?: string;
}

export function SectionCard({
  title,
  eyebrow,
  actions,
  children,
  className
}: SectionCardProps) {
  return (
    <section className={`section-card${className ? ` ${className}` : ""}`}>
      <header className="section-card__header">
        <div>
          {eyebrow ? <span className="section-card__eyebrow">{eyebrow}</span> : null}
          <h2>{title}</h2>
        </div>
        {actions ? <div className="section-card__actions">{actions}</div> : null}
      </header>
      <div className="section-card__body">{children}</div>
    </section>
  );
}
