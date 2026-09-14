import { useEffect, useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import About from "./pages/About";
import CropAnalysis from "./pages/CropAnalysis";
import Dashboard from "./pages/Dashboard";
import Explainability from "./pages/Explainability";
import FederatedLearning from "./pages/FederatedLearning";
import Fusion from "./pages/Fusion";
import History from "./pages/History";
import MultimodalAnalysis from "./pages/MultimodalAnalysis";
import Results from "./pages/Results";
import UAVMonitoring from "./pages/UAVMonitoring";
import { getCrops, getHealth } from "./services/api";
import { loadHistory } from "./state/history";

const fallbackCrops = ["guava", "paddy", "maize"];

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [apiStatus, setApiStatus] = useState("checking");
  const [crops, setCrops] = useState(fallbackCrops);
  const [latestResult, setLatestResult] = useState(() => loadHistory()[0]?.result || null);

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      try {
        await getHealth();
        const cropList = await getCrops();
        if (!cancelled) {
          setApiStatus("healthy");
          if (cropList.length) setCrops(cropList);
        }
      } catch {
        if (!cancelled) setApiStatus("unavailable");
      }
    }
    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  function handleResult(result) {
    setLatestResult(result);
    setActivePage("dashboard");
  }

  function renderPage() {
    switch (activePage) {
      case "crop":
        return <CropAnalysis crops={crops} onResult={handleResult} />;
      case "multimodal":
        return <MultimodalAnalysis crops={crops} onResult={handleResult} />;
      case "results":
        return <Results latestResult={latestResult} />;
      case "explainability":
        return <Explainability latestResult={latestResult} />;
      case "fusion":
        return <Fusion latestResult={latestResult} />;
      case "uav":
        return <UAVMonitoring latestResult={latestResult} />;
      case "history":
        return (
          <History
            onSelectResult={(result) => {
              setLatestResult(result);
              setActivePage("dashboard");
            }}
          />
        );
      case "about":
        return <About />;
      case "federated":
        return <FederatedLearning />;
      default:
        return <Dashboard latestResult={latestResult} onQuickAnalysis={() => setActivePage("crop")} />;
    }
  }

  return (
    <div className="min-h-screen bg-[#f7faf7] text-ink">
      <div className="lg:flex">
        <Sidebar activePage={activePage} onNavigate={setActivePage} />
        <div className="min-w-0 flex-1">
          <Header apiStatus={apiStatus} onQuickAnalysis={() => setActivePage("crop")} />
          <main className="mx-auto max-w-7xl px-5 py-6">{renderPage()}</main>
        </div>
      </div>
    </div>
  );
}
