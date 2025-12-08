// ===== CONFIG =====

const MODEL_URL   = "https://raw.githubusercontent.com/olesiakocheva/final_project/main/model/model.json";
const SCALER_URL  = "model/scaler_config_returns.json";
const HISTORY_URL = "data/history_returns.json";

// тот же LOOKBACK, что в ноутбуке
const WINDOW_SIZE = 60;


// ===== GLOBAL STATE =====

let model = null;
let scalerConfig = null;
let historyData = [];   // [{date, close, features_raw: [...]}, ...]
let chart = null;


// ===== HELPERS =====

function formatDate(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function getNextDateStr(lastDateStr) {
  const last = new Date(lastDateStr);
  last.setDate(last.getDate() + 1);
  return formatDate(last);
}

// StandardScaler: (x - mean_) / scale_
function scaleFeatures(rawFeatures) {
  if (!scalerConfig || !scalerConfig.mean_ || !scalerConfig.scale_) {
    return rawFeatures;
  }
  const mean = scalerConfig.mean_;
  const scale = scalerConfig.scale_;
  return rawFeatures.map((x, i) => {
    const m = mean[i] ?? 0;
    const s = scale[i] ?? 1;
    return (x - m) / s;
  });
}

// последнее окно длиной WINDOW_SIZE
function getFeatureWindow() {
  if (historyData.length < WINDOW_SIZE) {
    throw new Error("Not enough history for this WINDOW_SIZE");
  }
  const start = historyData.length - WINDOW_SIZE;
  const slice = historyData.slice(start);

  const windowArray = slice.map(row => {
    const raw = row.features_raw;
    return scaleFeatures(raw);
  });

  return windowArray;
}


// ===== MODEL LOADING & PREDICTION =====

async function loadModel() {
  console.log("Loading model:", MODEL_URL);
  model = await tf.loadLayersModel(MODEL_URL);
  console.log("Model loaded");
}

async function loadScaler() {
  console.log("Loading scaler:", SCALER_URL);
  const resp = await fetch(SCALER_URL);
  if (!resp.ok) {
    throw new Error(`Failed to load scaler: ${resp.status} ${resp.statusText}`);
  }
  scalerConfig = await resp.json();
  console.log("Scaler loaded");
}

// модель предсказывает 5-дневный log-return
async function predictNextReturn() {
  if (!model) throw new Error("Model not loaded");
  if (!historyData.length) throw new Error("No history data");

  const featureWindow = getFeatureWindow();
  const featureSize = featureWindow[0].length;

  const inputTensor = tf.tensor(featureWindow).reshape([1, WINDOW_SIZE, featureSize]);
  const predictionTensor = model.predict(inputTensor);
  const predictionArray = await predictionTensor.data();
  const predictedReturn = predictionArray[0];

  inputTensor.dispose();
  predictionTensor.dispose();

  return predictedReturn;
}


// ===== CHART =====

function createChart() {
  const ctx = document.getElementById("priceChart").getContext("2d");

  const labels = historyData.map(row => row.date);
  const prices = historyData.map(row => row.close);

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "SPX Close (historical + prediction)",
          data: prices,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#e5e7eb" }
        }
      },
      scales: {
        x: {
          ticks: { color: "#9ca3af", maxTicksLimit: 8 },
          grid: { display: false }
        },
        y: {
          ticks: { color: "#9ca3af" },
          grid: { color: "rgba(75,85,99,0.4)" }
        }
      }
    }
  });
}

function addPredictionToChart(dateStr, predictedPrice) {
  if (!chart) return;
  chart.data.labels.push(dateStr);
  chart.data.datasets[0].data.push(predictedPrice);
  chart.update();
}


// ===== DATA LOADING =====

async function loadHistory() {
  console.log("Loading history:", HISTORY_URL);
  const resp = await fetch(HISTORY_URL);
  if (!resp.ok) {
    throw new Error(`Failed to fetch history: ${resp.status} ${resp.statusText}`);
  }
  const json = await resp.json();
  // на всякий случай сортируем по дате
  historyData = json.sort((a, b) => (a.date > b.date ? 1 : -1));
  console.log("History loaded. Rows:", historyData.length);
}


// ===== APP INIT =====

async function initApp() {
  const statusEl = document.getElementById("status");
  const predictBtn = document.getElementById("predictBtn");

  try {
    statusEl.textContent = "Loading model, scaler & data…";

    await Promise.all([loadModel(), loadScaler(), loadHistory()]);

    createChart();

    statusEl.textContent = "Ready. Click “Predict Tomorrow”.";
    predictBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message;
  }
}


// ===== PREDICT BUTTON HANDLER =====

async function onPredictClick() {
  const statusEl = document.getElementById("status");
  const predictBtn = document.getElementById("predictBtn");

  try {
    predictBtn.disabled = true;
    statusEl.textContent = "Predicting…";

    const predictedReturn = await predictNextReturn();

    const lastRow = historyData[historyData.length - 1];
    const lastPrice = lastRow.close;

    // лог-доходность → цена
    const predictedPrice = lastPrice * Math.exp(predictedReturn);
    const nextDateStr = getNextDateStr(lastRow.date);

    historyData.push({
      date: nextDateStr,
      close: predictedPrice,
      features_raw: lastRow.features_raw, // заглушка
      isPrediction: true
    });

    addPredictionToChart(nextDateStr, predictedPrice);

    statusEl.textContent =
      `Predicted 5d return: ${(predictedReturn * 100).toFixed(2)}% ` +
      `→ price ≈ ${predictedPrice.toFixed(2)} on ${nextDateStr}`;
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Prediction error: " + err.message;
  } finally {
    predictBtn.disabled = false;
  }
}


// ===== START =====

window.addEventListener("DOMContentLoaded", () => {
  initApp();
  document.getElementById("predictBtn").addEventListener("click", onPredictClick);
});
