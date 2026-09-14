import PredictionCard from "../components/PredictionCard";
import ProbabilityChart from "../components/ProbabilityChart";

export default function Results({ latestResult }) {
  return (
    <div className="grid gap-6">
      <section>
        <h2 className="page-title">Results</h2>
        <p className="page-subtitle">
          Real prediction outputs from the latest backend response.
        </p>
      </section>
      <PredictionCard result={latestResult} />
      <section className="card p-5">
        <h2 className="text-lg font-semibold text-ink">Probability Distribution</h2>
        <ProbabilityChart probabilities={latestResult?.class_probabilities} />
      </section>
    </div>
  );
}
