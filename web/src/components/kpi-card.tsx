interface KpiCardProps {
  label: string;
  value: string;
  detail?: string;
  tone?: "neutral" | "keep" | "maybe" | "delete";
}

export function KpiCard({ label, value, detail, tone = "neutral" }: KpiCardProps) {
  return (
    <article className={`kpi-card kpi-card--${tone}`}>
      <span className="kpi-card__label">{label}</span>
      <strong className="kpi-card__value">{value}</strong>
      {detail ? <small className="kpi-card__detail">{detail}</small> : null}
    </article>
  );
}
