import {
  Activity,
  BarChart3,
  Cpu,
  Gauge,
  History,
  Layers3,
  Leaf,
  Map,
  Network,
  ScanSearch,
} from "lucide-react";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: Gauge },
  { id: "crop", label: "Crop Analysis", icon: Leaf },
  { id: "multimodal", label: "Multimodal", icon: Activity },
  { id: "results", label: "Results", icon: BarChart3 },
  { id: "explainability", label: "Explainable AI", icon: ScanSearch },
  { id: "fusion", label: "Fusion", icon: Layers3 },
  { id: "uav", label: "UAV Monitoring", icon: Map },
  { id: "history", label: "History", icon: History },
  { id: "about", label: "Architecture", icon: Cpu },
  { id: "federated", label: "Federated", icon: Network },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="border-r border-slate-200 bg-white lg:min-h-screen lg:w-72">
      <div className="border-b border-slate-200 px-5 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-canopy text-white">
            <Leaf size={23} aria-hidden="true" />
          </div>
          <div>
            <p className="text-base font-bold text-ink">AGROFEDVISION</p>
            <p className="text-xs text-slate-500">Research Dashboard</p>
          </div>
        </div>
      </div>

      <nav className="grid gap-1 p-3" aria-label="Primary navigation">
        {navItems.map(({ id, label, icon: Icon }) => {
          const active = activePage === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onNavigate(id)}
              className={`focus-ring flex items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-medium transition ${
                active
                  ? "bg-emerald-50 text-canopy"
                  : "text-slate-600 hover:bg-slate-50 hover:text-ink"
              }`}
              aria-current={active ? "page" : undefined}
            >
              <Icon size={18} aria-hidden="true" />
              <span>{label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mx-5 mt-4 rounded-lg border border-emerald-100 bg-emerald-50 p-4 text-sm text-emerald-900">
        <div className="flex items-center gap-2 font-semibold">
          <BarChart3 size={16} aria-hidden="true" />
          API Integration
        </div>
        <p className="mt-2 text-xs leading-5 text-emerald-800">
          Connected to the existing FastAPI endpoints. No synthetic prediction
          data is displayed.
        </p>
      </div>
    </aside>
  );
}
