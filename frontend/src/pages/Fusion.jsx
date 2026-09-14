export default function Fusion({ latestResult }) {
  const fusion = latestResult?.fusion;
  const weights = fusion?.weights || {};

  return (
    <div className="grid gap-6">
      <section>
        <h2 className="page-title">Modality Fusion</h2>
        <p className="page-subtitle">
          Visualizes only fusion information exposed by the current API response.
        </p>
      </section>

      <section className="card p-5">
        <h2 className="text-lg font-semibold text-ink">Learnable Gating / Fusion Weights</h2>
        {Object.keys(weights).length ? (
          <div className="mt-5 grid gap-4">
            {Object.entries(weights).map(([name, value]) => (
              <div key={name}>
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="font-semibold capitalize text-slate-700">{name}</span>
                  <span className="font-bold text-canopy">{(Number(value) * 100).toFixed(1)}%</span>
                </div>
                <div className="h-3 rounded-full bg-slate-100">
                  <div
                    className="h-3 rounded-full bg-canopy"
                    style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-4 rounded-lg border border-dashed border-slate-200 p-6 text-sm text-slate-500">
            Learnable fusion is active in the model. Modality weights are not
            currently exposed by the latest API response.
          </p>
        )}
      </section>

      <section className="card p-5">
        <h2 className="text-lg font-semibold text-ink">Fusion Summary</h2>
        <dl className="mt-4 grid gap-3 md:grid-cols-3">
          <Info label="Prediction Mode" value={fusion?.prediction_mode || "Unavailable"} />
          <Info label="Overall Status" value={fusion?.overall_status || "Unavailable"} />
          <Info
            label="Overall Health Score"
            value={
              fusion?.overall_health_score !== undefined
                ? `${Number(fusion.overall_health_score).toFixed(1)}/100`
                : "Unavailable"
            }
          />
        </dl>
      </section>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <dt className="metric-label">{label}</dt>
      <dd className="mt-2 font-semibold text-ink">{value}</dd>
    </div>
  );
}
