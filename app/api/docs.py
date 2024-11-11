# app/api/docs.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.core.config import settings
from typing import Dict, Any

docs_router = APIRouter()

@docs_router.get("/docs/api", response_class=HTMLResponse)
async def get_api_docs():
    """
    توليد وثائق API تفاعلية
    """
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>ML Framework API Documentation</title>
            <link href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
            <style>
                body { margin: 0; }
                .swagger-ui .topbar { display: none; }
                .swagger-ui .info { margin: 20px; }
                .swagger-ui .info h2 { color: #3b4151; }
                .swagger-ui .scheme-container { box-shadow: none; }
            </style>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script>
                window.onload = () => {
                    SwaggerUIBundle({
                        url: "/openapi.json",
                        dom_id: '#swagger-ui',
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIBundle.SwaggerUIStandalonePreset
                        ],
                        layout: "BaseLayout",
                        deepLinking: true,
                        showExtensions: true,
                        showCommonExtensions: true,
                        defaultModelsExpandDepth: 3,
                        defaultModelExpandDepth: 3,
                        displayRequestDuration: true,
                        docExpansion: "list",
                        filter: true,
                        syntaxHighlight: {
                            activate: true,
                            theme: "monokai"
                        }
                    });
                };
            </script>
        </body>
    </html>
    """

@docs_router.get("/docs/models")
async def get_models_documentation() -> Dict[str, Any]:
    """
    الحصول على وثائق حول النماذج المتاحة وقدراتها
    """
    return {
        "available_models": {
            "classification": [
                {
                    "name": "RandomForest",
                    "description": "خوارزمية تعلم جماعي للتصنيف",
                    "use_cases": ["التصنيف الثنائي", "التصنيف متعدد الفئات"],
                    "advantages": [
                        "التعامل مع العلاقات غير الخطية",
                        "مقاومة للقيم الشاذة",
                        "توفير أهمية المميزات"
                    ],
                    "parameters": {
                        "n_estimators": {
                            "type": "integer",
                            "default": 100,
                            "description": "عدد الأشجار في الغابة"
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": None,
                            "description": "أقصى عمق للشجرة"
                        }
                    }
                },
                {
                    "name": "XGBoost",
                    "description": "إطار عمل التعزيز المتدرج",
                    "use_cases": ["التصنيف الثنائي", "التصنيف متعدد الفئات"],
                    "advantages": [
                        "أداء عالي",
                        "التعامل مع القيم المفقودة",
                        "تنظيم مدمج"
                    ],
                    "parameters": {
                        "learning_rate": {
                            "type": "float",
                            "default": 0.1,
                            "description": "معدل التعلم"
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 6,
                            "description": "أقصى عمق للشجرة"
                        }
                    }
                }
            ],
            "regression": [
                {
                    "name": "XGBoost",
                    "description": "إطار عمل التعزيز المتدرج",
                    "use_cases": ["التنبؤ العددي", "التنبؤ بالسلاسل الزمنية"],
                    "advantages": [
                        "أداء عالي",
                        "التعامل مع القيم المفقودة",
                        "تنظيم مدمج"
                    ],
                    "parameters": {
                        "learning_rate": {
                            "type": "float",
                            "default": 0.1,
                            "description": "معدل التعلم"
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 6,
                            "description": "أقصى عمق للشجرة"
                        }
                    }
                }
            ]
        },
        "metrics_explanation": {
            "classification": {
                "accuracy": "نسبة التنبؤات الصحيحة إلى إجمالي التنبؤات",
                "precision": "نسبة التنبؤات الإيجابية الصحيحة إلى إجمالي التنبؤات الإيجابية",
                "recall": "نسبة التنبؤات الإيجابية الصحيحة إلى إجمالي الحالات الإيجابية الفعلية",
                "f1": "الوسط التوافقي للدقة والاستدعاء",
                "roc_auc": "مساحة المنحنى تحت منحنى ROC"
            },
            "regression": {
                "mse": "متوسط مربع الخطأ",
                "rmse": "الجذر التربيعي لمتوسط مربع الخطأ",
                "mae": "متوسط القيمة المطلقة للخطأ",
                "r2": "معامل التحديد"
            }
        },
        "preprocessing_steps": {
            "data_cleaning": [
                {
                    "name": "missing_values",
                    "description": "معالجة القيم المفقودة",
                    "methods": ["mean", "median", "mode", "knn"]
                },
                {
                    "name": "outliers",
                    "description": "اكتشاف ومعالجة القيم الشاذة",
                    "methods": ["z-score", "iqr", "isolation_forest"]
                },
                {
                    "name": "duplicates",
                    "description": "إزالة القيم المكررة",
                    "methods": ["exact", "fuzzy"]
                }
            ],
            "feature_engineering": [
                {
                    "name": "automatic_feature_creation",
                    "description": "إنشاء ميزات تلقائياً",
                    "methods": ["polynomial", "interaction", "aggregation"]
                },
                {
                    "name": "datetime_features",
                    "description": "استخراج ميزات من التواريخ",
                    "features": ["year", "month", "day", "hour", "weekday"]
                }
            ],
            "encoding": [
                {
                    "name": "categorical_encoding",
                    "description": "ترميز المتغيرات الفئوية",
                    "methods": [
                        "one_hot",
                        "label",
                        "target",
                        "frequency",
                        "weight_of_evidence"
                    ]
                }
            ]
        },
        "api_endpoints": {
            "training": {
                "/train": {
                    "method": "POST",
                    "description": "تدريب نموذج جديد",
                    "parameters": ["file", "model_type", "target_column"]
                },
                "/models/batch/train": {
                    "method": "POST",
                    "description": "تدريب عدة نماذج",
                    "parameters": ["files", "model_types"]
                }
            },
            "prediction": {
                "/predict/{model_id}": {
                    "method": "POST",
                    "description": "تنفيذ تنبؤ",
                    "parameters": ["features"]
                },
                "/predict/batch": {
                    "method": "POST",
                    "description": "تنفيذ تنبؤات متعددة",
                    "parameters": ["files"]
                }
            },
            "model_management": {
                "/models": {
                    "method": "GET",
                    "description": "قائمة النماذج",
                    "parameters": ["skip", "limit", "sort_by", "order"]
                },
                "/models/{model_id}/version": {
                    "method": "POST",
                    "description": "إنشاء نسخة من النموذج",
                    "parameters": ["version_name"]
                }
            },
            "/validate": {
                "description": "التحقق من صحة البيانات والنماذج",
                "parameters": {
                    "file": "ملف البيانات للتحقق",
                    "model_id": "معرف النموذج (اختياري)"
                },
                "responses": {
                    "200": {
                        "description": "نتيجة التحقق",
                        "schema": "DataValidationResult"
                    }
                }
            }
        }
    }

@docs_router.get("/docs/examples")
async def get_usage_examples() -> Dict[str, Any]:
    """
    الحصول على أمثلة استخدام API
    """
    return {
        "training_example": {
            "description": "مثال على تدريب نموذج",
            "code": """
                import requests
                
                # تحميل ملف البيانات
                files = {'file': open('data.csv', 'rb')}
                
                # تدريب النموذج
                response = requests.post(
                    'http://api.example.com/train',
                    files=files
                )
                
                # الحصول على معرف النموذج
                model_id = response.json()['model_id']
            """
        },
        "prediction_example": {
            "description": "مثال على تنفيذ تنبؤ",
            "code": """
                import requests
                
                # بيانات التنبؤ
                features = {
                    'feature1': 1.0,
                    'feature2': 'value'
                }
                
                # تنفيذ التنبؤ
                response = requests.post(
                    f'http://api.example.com/predict/{model_id}',
                    json={'features': features}
                )
                
                # الحصول على النتيجة
                prediction = response.json()['prediction']
            """
        }
    }