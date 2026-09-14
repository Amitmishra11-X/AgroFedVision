import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { UAV_FIELDS, formatFieldName } from "../services/api";

export default function UAVMonitoring({ latestResult }) {
  const context = latestResult?.uav_context;
  const stats = context?.uav_statistics;
  const data = context
    ? [
        { name: "NDVI Mean", value: context.NDVI_Mean },
        { name: "NDRE Mean", value: context.NDRE_Mean },
      ].filter((item) => item.value !== undefined && item.value !== null)
    : [];

  return (
    <div className="grid gap-6">
      <section>
        <h2 className="page-title">UAV Monitoring</h2>
        <p className="page-subtitle">
          Displays UAV values only when the backend computes them from uploaded
          multispectral captures.
        </p>
      </section>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {UAV_FIELDS.map((field) => (
          <div key={field} className="card p-5">
            <p className="metric-label">{formatFieldName(field)}</p>
            <p className="mt-3 text-2xl font-bold text-ink">
              {context?.[field] !== undefined && context?.[field] !== null
                ? Number(context[field]).toFixed(4)
                : "Unavailable"}
            </p>
          </div>
        ))}
      </div>

      <section className="card p-5">
        <h2 className="text-lg font-semibold text-ink">Vegetation Index Snapshot</h2>
        {data.length ? (
          <div className="mt-5 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#1f8a55" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="mt-4 rounded-lg border border-dashed border-slate-200 p-6 text-sm text-slate-500">
            No UAV interpretation was returned by the latest prediction.
          </p>
        )}
        {stats && (
          <p className="mt-4 text-sm text-slate-600">
            Captures processed: <span className="font-semibold">{stats.total_images}</span>
          </p>
        )}
      </section>

      <section className="card p-5">
        <p className="metric-label">Spatial UAV Mapping</p>
        <h2 className="mt-2 text-xl font-semibold text-ink">Coming Soon</h2>
        <p className="mt-2 text-sm text-slate-600">
          Future spatial NDVI/NDRE maps will appear here when exposed by the backend.
        </p>
      </section>
    </div>
  );
}
