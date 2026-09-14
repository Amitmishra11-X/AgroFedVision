export default function FederatedLearning() {
  const nodes = ["Farm / Client A", "Farm / Client B", "Farm / Client C", "Federated Server", "FedProx", "Global Model"];

  return (
    <div className="grid gap-6">
      <section>
        <h2 className="page-title">Federated Learning - Research Module</h2>
        <p className="page-subtitle">
          FedProx integration is planned / under development. This section is
          intentionally architectural and does not show fabricated accuracy,
          rounds, or communication statistics.
        </p>
      </section>

      <section className="card p-5">
        <div className="grid gap-3 md:grid-cols-6 md:items-center">
          {nodes.map((node, index) => (
            <div key={node} className="contents">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-center text-sm font-semibold text-ink">
                {node}
              </div>
              {index < nodes.length - 1 && <div className="hidden h-px bg-slate-300 md:block" aria-hidden="true" />}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
