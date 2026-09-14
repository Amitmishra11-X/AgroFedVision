const STORAGE_KEY = "agrofedvision.predictionHistory.v1";

export function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveHistory(items) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, 20)));
}

export function addHistoryItem(item) {
  const next = [item, ...loadHistory()].slice(0, 20);
  saveHistory(next);
  return next;
}

export function clearHistory() {
  localStorage.removeItem(STORAGE_KEY);
}
