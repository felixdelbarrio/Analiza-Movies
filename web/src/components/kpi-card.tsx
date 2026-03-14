interface KpiCardProps {
  label: string;
  value: string;
  tone?: "neutral" | "keep" | "maybe" | "delete";
}

export function KpiCard({ label, value, tone = "neutral" }: KpiCardProps) {
  return (
    <article className={`kpi-card kpi-card--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}
