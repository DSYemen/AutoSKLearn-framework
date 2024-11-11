// static/js/main.js

// حالة التطبيق
const appState = {
    selectedFile: null,
    processing: false,
    uploadProgress: 0,
    processingStep: null,
    websocket: null
};

// تهيئة التطبيق عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', () => {
    initializeFileUpload();
    setupWebSocket();
    initializeStats();
});

// تهيئة منطقة تحميل الملفات
function initializeFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const selectFileButton = document.getElementById('selectFileButton');
    const uploadButton = document.getElementById('uploadButton');

    // إعداد منطقة السحب والإفلات
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-indigo-500');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-indigo-500');
        });

        dropZone.addEventListener('drop', handleFileDrop);
    }

    // إعداد زر اختيار الملف
    if (selectFileButton) {
        selectFileButton.addEventListener('click', () => {
            fileInput.click();
        });
    }

    // إعداد حدث تغيير الملف
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // إعداد زر التحميل
    if (uploadButton) {
        uploadButton.addEventListener('click', uploadFile);
    }
}

// معالجة إسقاط الملف
function handleFileDrop(e) {
    e.preventDefault();
    const dropZone = document.getElementById('dropZone');
    dropZone.classList.remove('border-indigo-500');

    const file = e.dataTransfer.files[0];
    if (file) {
        handleFile(file);
    }
}

// معالجة اختيار الملف
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// معالجة الملف المختار
function handleFile(file) {
    // التحقق من نوع وحجم الملف
    const validTypes = ['.csv', '.xlsx', '.parquet'];
    const maxSize = 100 * 1024 * 1024; // 100 MB

    if (!validTypes.some(type => file.name.toLowerCase().endsWith(type))) {
        showError('نوع الملف غير مدعوم. الرجاء اختيار ملف CSV, XLSX, أو Parquet.');
        return;
    }

    if (file.size > maxSize) {
        showError('حجم الملف يتجاوز الحد الأقصى المسموح به (100 MB)');
        return;
    }

    // تحديث حالة التطبيق وواجهة المستخدم
    appState.selectedFile = file;
    updateFileDisplay();
    document.getElementById('uploadButton').classList.remove('hidden');
}

// تحديث عرض الملف المختار
function updateFileDisplay() {
    const fileDisplay = document.getElementById('fileDisplay');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');

    if (appState.selectedFile) {
        fileDisplay.classList.remove('hidden');
        fileName.textContent = appState.selectedFile.name;
        fileSize.textContent = formatFileSize(appState.selectedFile.size);
    } else {
        fileDisplay.classList.add('hidden');
    }
}

// تنسيق حجم الملف
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// إضافة الدوال المفقودة لمعالجة تقدم لتحميل
function showUploadProgress() {
    const uploadProgress = document.getElementById('uploadProgress');
    if (uploadProgress) {
        uploadProgress.classList.remove('hidden');
    }
    
    // إعادة تعيين شريط التقدم
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const progressStatus = document.getElementById('progressStatus');
    
    if (progressBar) progressBar.style.width = '0%';
    if (progressText) progressText.textContent = '0%';
    if (progressStatus) progressStatus.textContent = 'جاري التحميل...';
}

function hideUploadProgress() {
    const uploadProgress = document.getElementById('uploadProgress');
    if (uploadProgress) {
        uploadProgress.classList.add('hidden');
    }
}

// تعديل دالة uploadFile لتتعامل مع التحميل بشكل صحيح
async function uploadFile() {
    if (!appState.selectedFile || appState.processing) return;

    try {
        appState.processing = true;
        showUploadProgress();

        const formData = new FormData();
        formData.append('file', appState.selectedFile);

        const response = await fetch('/api/v1/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('فشل تحميل الملف');

        const result = await response.json();
        appState.jobId = result.job_id;
        
        // إخفاء زر التحميل وإظهار زر المعالجة
        document.getElementById('uploadButton').classList.add('hidden');
        document.getElementById('processButton').classList.remove('hidden');
        
        hideUploadProgress();
        showSuccess('تم تحميل الملف بنجاح');

    } catch (error) {
        showError('حدث خطأ أثناء تحميل الملف: ' + error.message);
        hideUploadProgress();
    } finally {
        appState.processing = false;
    }
}

// تحديث دالة updateProgress
function updateProgress(progress) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const progressStatus = document.getElementById('progressStatus');
    
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
    }
    
    if (progressText) {
        progressText.textContent = `${Math.round(progress)}%`;
    }
    
    if (progress === 100 && progressStatus) {
        progressStatus.textContent = 'اكتمل التحميل';
    }
}

// إعداد اتصال WebSocket
function setupWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    appState.websocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/updates`);
    
    appState.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
}

// معالجة رسائل WebSocket
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'processing_update':
            updateProcessingStatus(data);
            break;
        case 'stats_update':
            updateStats(data.stats);
            break;
        case 'error':
            showError(data.message);
            break;
    }
}

// تحديث حالة المعالجة
function updateProcessingStatus(data) {
    const { step, progress, status } = data;
    
    // تحديث الخطوة الحالية
    document.querySelectorAll('.processing-step').forEach(stepElement => {
        const stepName = stepElement.dataset.step;
        const indicator = stepElement.querySelector('span');
        
        if (stepName === step) {
            indicator.textContent = '🔄';
        } else if (data.completed_steps.includes(stepName)) {
            indicator.textContent = '✅';
        }
    });

    // تحديث شريط التقدم
    document.getElementById('progressBar').style.width = `${progress}%`;
    document.getElementById('progressPercentage').textContent = `${progress}%`;

    // إذا اكتملت المعالجة
    if (status === 'completed') {
        handleProcessingComplete(data.result_url);
    }
}

// تحديث الإحصائيات
function updateStats(stats) {
    document.getElementById('activeModelsCount').textContent = stats.active_models;
    document.getElementById('totalPredictions').textContent = stats.total_predictions;
    document.getElementById('averageAccuracy').textContent = `${(stats.avg_accuracy * 100).toFixed(1)}%`;
    document.getElementById('systemHealth').textContent = `${stats.system_health}%`;
}

// إظهار قسم المعالجة
function showProcessingSection(jobId) {
    document.getElementById('uploadSection').classList.add('hidden');
    document.getElementById('processingSection').classList.remove('hidden');
    
    // إنشاء اتصال WebSocket لمتابعة حالة المعالجة
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const processingSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/processing/${jobId}`);
    
    processingSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateProcessingUI(data);
    };
}

// إضافة دالة تحديث واجهة المعالجة
function updateProcessingUI(data) {
    // تحديث شريط التقدم
    const progressBar = document.getElementById('processingProgressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    
    if (progressBar) {
        progressBar.style.width = `${data.progress}%`;
    }
    if (progressPercentage) {
        progressPercentage.textContent = `${data.progress}%`;
    }

    // تحديث حالة الخطوات
    const steps = document.querySelectorAll('.processing-step');
    steps.forEach(step => {
        const stepName = step.dataset.step;
        const indicator = step.querySelector('.step-indicator');
        
        if (stepName === data.step) {
            indicator.textContent = '🔄'; // جاري التنفيذ
        } else if (data.completed_steps.includes(stepName)) {
            indicator.textContent = '✅'; // مكتمل
        }
    });

    // إذا اكتملت المعالجة
    if (data.step === 'completed') {
        handleProcessingComplete(data.result_url);
    } else if (data.step === 'failed') {
        handleProcessingError(data.message);
    }
}

// دالة مساعدة لتحديد نسبة تقدم كل خطوة
function getStepProgress(stepName) {
    const stepProgress = {
        'data-loading': 20,
        'preprocessing': 40,
        'feature-engineering': 60,
        'model-selection': 80,
        'training': 90
    };
    return stepProgress[stepName] || 0;
}

// معالجة اكتمال المعالجة
function handleProcessingComplete(resultUrl) {
    showSuccess('تم اكتمال معالجة البيانات بنجاح');
    if (resultUrl) {
        window.location.href = resultUrl;
    }
}

// معالجة أخطاء المعالجة
function handleProcessingError(message) {
    showError(message);
    document.getElementById('processingSection').classList.add('hidden');
    document.getElementById('uploadSection').classList.remove('hidden');
}

// إضافة دالة لعرض رسائل النجاح
function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4';
    alert.role = 'alert';
    
    alert.innerHTML = `
        <strong class="font-bold">نجاح!</strong>
        <span class="block sm:inline"> ${message}</span>
        <span class="absolute top-0 bottom-0 left-0 px-4 py-3">
            <svg class="fill-current h-6 w-6 text-green-500" role="button" onclick="this.parentElement.parentElement.remove()"
                 xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <title>Close</title>
                <path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.2 1.2 0 0 1 0 1.698z"/>
            </svg>
        </span>
    `;
    
    const container = document.querySelector('.container') || document.body;
    container.insertBefore(alert, container.firstChild);
    
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// إزالة الملف
function removeFile() {
    appState.selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileDisplay').classList.add('hidden');
    document.getElementById('uploadButton').classList.add('hidden');
}

// تحديث دالة showError لعرض الأخطاء بشكل أفضل
function showError(message) {
    console.error(message);
    
    // إنشاء عنصر تنبيه
    const alert = document.createElement('div');
    alert.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4';
    alert.role = 'alert';
    
    alert.innerHTML = `
        <strong class="font-bold">خطأ!</strong>
        <span class="block sm:inline"> ${message}</span>
        <span class="absolute top-0 bottom-0 left-0 px-4 py-3">
            <svg class="fill-current h-6 w-6 text-red-500" role="button" onclick="this.parentElement.parentElement.remove()"
                 xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <title>Close</title>
                <path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.2 1.2 0 0 1 0 1.698z"/>
            </svg>
        </span>
    `;
    
    // إضافة التنبيه إلى الصفحة
    const container = document.querySelector('.container') || document.body;
    container.insertBefore(alert, container.firstChild);
    
    // إزالة التنبيه بعد 5 ثواني
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// تحديث دالة initializeStats للتعامل مع الخطأ 404
function initializeStats() {
    // تحديث الإحصائيات كل 30 ثانية
    const updateStats = async () => {
        try {
            const response = await fetch('/api/v1/stats');
            if (!response.ok) {
                if (response.status === 404) {
                    console.warn('مسار الإحصائيات غير متوفر');
                    return;
                }
                throw new Error(`خطأ في الاستجابة: ${response.status}`);
            }
            const stats = await response.json();
            updateStatsDisplay(stats);
        } catch (error) {
            console.error('Error updating stats:', error);
        }
    };

    // التحديث الأولي
    updateStats();
    
    // تحديث دوري
    setInterval(updateStats, 30000);
}

// دالة تحديث عرض الإحصائيات
function updateStatsDisplay(stats) {
    const elements = {
        'activeModelsCount': stats.active_models,
        'totalPredictions': stats.total_predictions,
        'averageAccuracy': `${(stats.avg_accuracy * 100).toFixed(1)}%`,
        'systemHealth': `${stats.system_health}%`
    };

    for (const [id, value] of Object.entries(elements)) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
}

// إضافة دالة معالجة البيانات
async function processData() {
    if (!appState.jobId || appState.processing) return;

    try {
        appState.processing = true;
        document.getElementById('uploadSection').classList.add('hidden');
        document.getElementById('processingSection').classList.remove('hidden');
        
        // إنشاء اتصال WebSocket لمتابعة حالة المعالجة
        setupProcessingWebSocket(appState.jobId);

        // بدء المعالجة
        const response = await fetch(`/api/v1/process/${appState.jobId}`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('فشل بدء المعالجة');

    } catch (error) {
        showError('حدث خطأ أثناء معالجة البيانات: ' + error.message);
        document.getElementById('processingSection').classList.add('hidden');
        document.getElementById('uploadSection').classList.remove('hidden');
    }
}

// إضافة مستمع الحدث لزر المعالجة
document.getElementById('processButton')?.addEventListener('click', processData);

// إضافة دالة setupProcessingWebSocket
function setupProcessingWebSocket(jobId) {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const processingSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/processing/${jobId}`);
    
    processingSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateProcessingUI(data);
    };

    processingSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('حدث خطأ في الاتصال');
    };

    processingSocket.onclose = () => {
        console.log('WebSocket connection closed');
    };
}
