import type { ReactNode } from "react";

interface PageHeroProps {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}

export function PageHero({ eyebrow, title, description, actions }: PageHeroProps) {
  return (
    <header className="page-hero">
      <div>
        <span className="page-hero__eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-hero__actions">{actions}</div> : null}
    </header>
  );
}
