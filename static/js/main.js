// static/js/main.js

// الحالة العامة للتطبيق
let appState = {
  selectedFile: null,
  processing: false,
  results: null,
  error: null,
  models: [],
  selectedModel: null,
  stats: {
    active_models: 0,
    total_predictions: 0,
    predictions_per_hour: 0,
    avg_accuracy: 0,
    accuracy_trend: 0,
    system_health: 100,
    health_status: "Healthy",
    new_models_24h: 0,
  },
  charts: {},
};

// تهيئة الرسوم البيانية
function initializeCharts() {
  // تنفيذ إعداد الرسوم البيانية هنا
}

// تحديث الرسوم البيانية
function updateDashboardCharts() {
  // تحديث الرسوم البيانية هنا
}

// تحديث واجهة المستخدم بالبيانات الجديدة
function updateUI() {
  // تحديث العناصر في DOM بناءً على حالة appState
  // على سبيل المثال، تحديث الرسوم البيانية، عرض النتائج، إلخ.
}

// جلب البيانات من الـ API وتحديث الواجهة
async function fetchDashboardData() {
  try {
    const [modelsResponse, statsResponse] = await Promise.all([
      fetch("/api/v1/models"),
      fetch("/api/v1/stats"),
    ]);

    appState.models = await modelsResponse.json();
    appState.stats = await statsResponse.json();

    updateDashboardCharts();
    updateUI();
  } catch (error) {
    console.error("Error fetching dashboard data:", error);
  }
}

// التعامل مع تحميل الملف
function handleFileSelect(event) {
  appState.selectedFile = event.target.files[0];
  appState.error = null;
}

// التعامل مع سحب وإفلات الملف
function handleFileDrop(event) {
  event.preventDefault();
  appState.selectedFile = event.dataTransfer.files[0];
  appState.error = null;
}

// تنسيق حجم الملف
function formatFileSize(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// معالجة الملف
async function processFile() {
  if (!appState.selectedFile) return;

  appState.processing = true;
  appState.error = null;

  try {
    const formData = new FormData();
    formData.append("file", appState.selectedFile);

    const response = await fetch("/api/v1/train", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    appState.results = await response.json();
    updateCharts();
  } catch (error) {
    appState.error = error.message;
  } finally {
    appState.processing = false;
    updateUI();
  }
}

// بدء التحديث التلقائي
function startAutoRefresh() {
  setInterval(() => {
    fetchDashboardData();
  }, 30000); // تحديث كل 30 ثانية
}

// جلب تفاصيل النموذج
async function viewModelDetails(modelId) {
  try {
    const response = await fetch(`/api/v1/models/${modelId}`);
    appState.selectedModel = await response.json();
    updateUI();
  } catch (error) {
    console.error("Error fetching model details:", error);
  }
}

// تنسيق التاريخ
function formatDate(date) {
  return new Date(date).toLocaleString();
}

// تنسيق اسم المتريكس
function formatMetricName(metric) {
  return metric
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

// تنسيق قيمة المتريكس
function formatMetricValue(value) {
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  return value;
}

// تحميل النموذج
function downloadModel(modelId) {
  window.location.href = `/api/v1/models/${modelId}/download`;
}

// إعادة تدريب النموذج
async function retrainModel(modelId) {
  try {
    await fetch(`/api/v1/models/${modelId}/retrain`, {
      method: "POST",
    });
    fetchDashboardData();
  } catch (error) {
    console.error("Error retraining model:", error);
  }
}

// بدء التطبيق
document.addEventListener("DOMContentLoaded", () => {
  // تهيئة الرسوم البيانية عند تحميل الصفحة
  initializeCharts();

  // جلب البيانات الأولية
  fetchDashboardData();

  // بدء التحديث التلقائي
  startAutoRefresh();

  // إضافة مستمعي الأحداث لعناصر الإدخال
  const fileInput = document.querySelector("#fileInput");
  if (fileInput) {
    fileInput.addEventListener("change", handleFileSelect);
  }

  const fileDropZone = document.querySelector("#fileDropZone");
  if (fileDropZone) {
    fileDropZone.addEventListener("drop", handleFileDrop);
    fileDropZone.addEventListener("dragover", (e) => e.preventDefault());
  }

  // إضافة أحداث أخرى حسب الحاجة
});
