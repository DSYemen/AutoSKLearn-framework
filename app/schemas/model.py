# app/schemas/model.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """أنواع النماذج المدعومة"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class ModelStatus(str, Enum):
    """حالات النموذج"""
    ACTIVE = "active"
    TRAINING = "training"
    FAILED = "failed"
    INACTIVE = "inactive"

class ModelBase(BaseModel):
    """النموذج الأساسي"""
    type: ModelType
    name: str = Field(..., description="اسم النموذج")
    description: Optional[str] = Field(None, description="وصف النموذج")
    parameters: Dict[str, Any] = Field(..., description="معلمات النموذج")
    features: List[str] = Field(..., description="قائمة المميزات المستخدمة")

    model_config = ConfigDict(protected_namespaces=())

class ModelCreate(ModelBase):
    """إنشاء نموذج جديد"""
    target_column: str = Field(..., description="عمود الهدف")
    training_config: Optional[Dict[str, Any]] = Field(default={}, description="إعدادات التدريب")

    model_config = ConfigDict(protected_namespaces=())

class ModelUpdate(BaseModel):
    """تحديث النموذج"""
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    status: Optional[ModelStatus] = None

class ModelMetrics(BaseModel):
    """مقاييس أداء النموذج"""
    accuracy: float = Field(..., ge=0, le=1)
    f1: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    roc_auc: Optional[float] = Field(None, ge=0, le=1)
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    accuracy_trend: float = Field(0.0, description="اتجاه تغير الدقة")

class ModelStats(BaseModel):
    """إحصائيات النموذج"""
    total_predictions: int = 0
    predictions_per_hour: float = 0
    training_time: float
    size: str
    system_health: int = Field(100, ge=0, le=100)
    health_status: str = "Healthy"

    model_config = ConfigDict(protected_namespaces=())

class ModelResponse(ModelBase):
    """استجابة النموذج"""
    id: str
    status: ModelStatus
    created_at: datetime
    updated_at: Optional[datetime]
    metrics: ModelMetrics
    stats: ModelStats
    feature_importance: Dict[str, float]
    version: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    )

class ModelListResponse(BaseModel):
    """استجابة قائمة النماذج"""
    models: List[ModelResponse]
    total: int
    page: int
    pages: int

class PredictionRequest(BaseModel):
    """طلب التنبؤ"""
    features: Dict[str, Any] = Field(..., description="قيم المميزات للتنبؤ")
    return_confidence: bool = Field(False, description="إرجاع درجة الثقة")

class PredictionResponse(BaseModel):
    """استجابة التنبؤ"""
    id: str
    prediction: Union[float, List[float]]
    confidence: Optional[float] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(protected_namespaces=())

class BatchPredictionRequest(BaseModel):
    """طلب تنبؤات متعددة"""
    instances: List[Dict[str, Any]]
    return_confidence: bool = False

class BatchPredictionResponse(BaseModel):
    """استجابة تنبؤات متعددة"""
    predictions: List[PredictionResponse]
    total_time: float

class PredictionHistoryResponse(BaseModel):
    """استجابة سجل التنبؤات"""
    predictions: List[PredictionResponse]
    total: int
    page: int
    pages: int

class TrainingStatus(BaseModel):
    """حالة التدريب"""
    status: str
    progress: int
    step: str
    message: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelVersion(BaseModel):
    """نسخة النموذج"""
    version: str
    created_at: datetime
    metrics: ModelMetrics
    changes: Dict[str, Any]
    parent_version: Optional[str] = None

class ModelVersionResponse(BaseModel):
    """استجابة نسخ النموذج"""
    versions: List[ModelVersion]
    current_version: str

class ModelAlert(BaseModel):
    """تنبيه النموذج"""
    alert_id: str
    model_id: str
    type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class ModelComparison(BaseModel):
    """مقارنة النماذج"""
    models: List[str]
    metric: str
    values: Dict[str, float]
    winner: str
    comparison_time: datetime = Field(default_factory=datetime.utcnow)

class DataProfileResponse(BaseModel):
    """استجابة تحليل البيانات"""
    dataset_name: str
    overview: Dict[str, Any]
    variables: List[Dict[str, Any]]
    variable_stats: Dict[str, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelDriftResponse(BaseModel):
    """استجابة انحراف النموذج"""
    model_id: str
    drift_detected: bool
    drift_score: float
    feature_drifts: Dict[str, float]
    period: Dict[str, datetime]
    recommendations: List[str]

class MetricsResponse(BaseModel):
    """استجابة مقاييس النموذج"""
    model_id: str
    metrics: ModelMetrics
    performance_trend: Dict[str, List[float]]
    predictions_distribution: Dict[str, int]
    feature_correlations: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DashboardData(BaseModel):
    """بيانات لوحة التحكم"""
    model_info: ModelResponse
    performance_data: Dict[str, List[float]]
    predictions_data: List[PredictionResponse]
    alerts: List[ModelAlert]
    system_stats: Dict[str, Any]

class ProcessingStatus(BaseModel):
    """حالة معالجة البيانات"""
    job_id: str
    status: str
    progress: int
    step: str
    message: Optional[str]
    details: Optional[Dict[str, Any]]
    started_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime]

class FeatureMetadata(BaseModel):
    """بيانات وصفية للميزات"""
    name: str
    type: str
    description: Optional[str]
    required: bool = True
    constraints: Optional[Dict[str, Any]]
    example: Any

class ModelEndpoints(BaseModel):
    """نقاط نهاية النموذج"""
    predict_url: str
    batch_predict_url: str
    metrics_url: str
    update_url: str
    version_url: str

class ModelDeployment(BaseModel):
    """نشر النموذج"""
    deployment_id: str
    model_id: str
    environment: str
    status: str
    endpoint_url: str
    deployed_at: datetime
    deployed_by: str
    resources: Dict[str, Any]
    config: Dict[str, Any]

class ModelExport(BaseModel):
    """تصدير النموذج"""
    model_id: str
    format: str
    include_metadata: bool = True
    include_preprocessing: bool = True
    compression: Optional[str] = None
    encryption: Optional[Dict[str, Any]] = None

class DataValidationResult(BaseModel):
    """نتيجة التحقق من البيانات"""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    stats: Dict[str, Any]
    recommendations: List[str]
    validation_time: datetime = Field(default_factory=datetime.utcnow)

class ModelOptimizationConfig(BaseModel):
    """إعدادات تحسين النموذج"""
    optimization_metric: str
    max_trials: int = 100
    timeout: int = 3600
    optimization_parameters: Dict[str, Dict[str, Any]]
    cross_validation: Dict[str, Any]
    early_stopping: Optional[Dict[str, Any]]

class ModelComparisonResult(BaseModel):
    """نتيجة مقارنة النماذج"""
    models: List[str]
    metrics: Dict[str, Dict[str, float]]
    best_model: str
    comparison_details: Dict[str, Any]
    comparison_plots: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AutoMLConfig(BaseModel):
    """إعدادات التعلم الآلي التلقائي"""
    time_limit: int = 3600
    metric: str = "accuracy"
    problem_type: Optional[str] = None
    max_models: int = 10
    ensemble: bool = True
    feature_engineering: bool = True
    optimization: bool = True

class ModelMonitoringConfig(BaseModel):
    """إعدادات مراقبة النموذج"""
    metrics_to_monitor: List[str]
    alert_thresholds: Dict[str, float]
    update_interval: int
    drift_detection: Dict[str, Any]
    performance_tracking: Dict[str, Any]
    alert_channels: List[str]

class BatchProcessingResult(BaseModel):
    """نتيجة المعالجة المجمعة"""
    batch_id: str
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    errors: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WebhookConfig(BaseModel):
    """إعدادات Webhook"""
    url: str
    events: List[str]
    headers: Optional[Dict[str, str]]
    retry_config: Optional[Dict[str, Any]]
    secret: Optional[str]
    active: bool = True

class APIKey(BaseModel):
    """مفتاح API"""
    key_id: str
    key: str
    name: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    created_by: str

class UsageStatistics(BaseModel):
    """إحصائيات الاستخدام"""
    period_start: datetime
    period_end: datetime
    total_requests: int
    total_predictions: int
    average_response_time: float
    error_rate: float
    resource_usage: Dict[str, float]
    cost_estimate: Optional[float]