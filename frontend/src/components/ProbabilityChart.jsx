import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ProbabilityChart({ probabilities }) {
  const data = Object.entries(probabilities || {}).map(([name, value]) => ({
    name: name.replaceAll("_", " "),
    probability: Number(value) * 100,
  }));

  if (!data.length) {
    return (
      <div className="rounded-lg border border-dashed border-slate-200 p-6 text-sm text-slate-500">
        Class probabilities were not returned by the API.
      </div>
    );
  }

  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
          <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(value) => [`${Number(value).toFixed(2)}%`, "Probability"]} />
          <Bar dataKey="probability" fill="#1f8a55" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
