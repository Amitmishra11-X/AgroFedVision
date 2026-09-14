import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function SHAPChart({ result }) {
  const shap = result?.explainability?.shap_explanation;
  const features = shap?.top_features || [];
  const data = features.map((item) => ({
    feature: item.feature,
    value: item.shap_value,
    absolute: Math.abs(Number(item.shap_value || 0)),
    direction: item.direction,
  }));

  return (
    <section className="card p-5">
      <div className="mb-5">
        <p className="metric-label">SHAP</p>
        <h2 className="mt-2 text-xl font-semibold text-ink">Structured Feature Importance</h2>
      </div>
      {!data.length ? (
        <div className="rounded-lg border border-dashed border-slate-200 p-6 text-sm text-slate-500">
          SHAP explanation is currently unavailable for this prediction.
        </div>
      ) : (
        <>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(value, name, props) => [
                    Number(value).toFixed(6),
                    props.payload.direction,
                  ]}
                />
                <Bar dataKey="value" fill="#37a169" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 text-sm text-slate-500">
            Explained class: <span className="font-semibold text-slate-700">{shap.explained_class}</span>
          </p>
        </>
      )}
    </section>
  );
}
