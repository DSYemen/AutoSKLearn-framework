import os
from typing import Dict, Any, Optional
import joblib
import onnx
import json
from pathlib import Path
from datetime import datetime
from app.core.config import settings
from app.core.logging_config import logger
from app.schemas.model import ModelExport

class ModelExporter:
    """فئة لتصدير النماذج بتنسيقات مختلفة"""

    def __init__(self):
        self.export_formats = {
            'joblib': self._export_joblib,
            'onnx': self._export_onnx,
            'json': self._export_json,
            'pickle': self._export_pickle
        }
        self.exports_dir = settings.MODELS_DIR / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    async def export_model(
        self,
        model: Any,
        format: str,
        model_id: str,
        include_metadata: bool = True,
        include_preprocessing: bool = True,
        compression: Optional[str] = None,
        encryption: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        تصدير النموذج بالتنسيق المطلوب
        """
        try:
            if format not in self.export_formats:
                raise ValueError(f"تنسيق التصدير غير مدعوم: {format}")

            # إنشاء مجلد للتصدير
            export_path = self.exports_dir / model_id / format
            export_path.mkdir(parents=True, exist_ok=True)

            # إنشاء اسم الملف
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = export_path / f"model_{timestamp}.{format}"

            # تصدير النموذج
            export_func = self.export_formats[format]
            await export_func(model, file_path, include_metadata, include_preprocessing)

            # ضغط الملف إذا تم طلب ذلك
            if compression:
                file_path = self._compress_file(file_path, compression)

            # تشفير الملف إذا تم طلب ذلك
            if encryption:
                file_path = self._encrypt_file(file_path, encryption)

            logger.info(f"تم تصدير النموذج بنجاح: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"خطأ في تصدير النموذج: {str(e)}")
            raise

    async def _export_joblib(
        self,
        model: Any,
        file_path: Path,
        include_metadata: bool,
        include_preprocessing: bool
    ) -> None:
        """تصدير النموذج بتنسيق joblib"""
        export_data = {
            'model': model,
            'metadata': self._get_metadata(model) if include_metadata else None,
            'preprocessing': self._get_preprocessing(model) if include_preprocessing else None
        }
        joblib.dump(export_data, file_path)

    async def _export_onnx(
        self,
        model: Any,
        file_path: Path,
        include_metadata: bool,
        include_preprocessing: bool
    ) -> None:
        """تصدير النموذج بتنسيق ONNX"""
        try:
            # تحويل النموذج إلى ONNX
            onnx_model = self._convert_to_onnx(model)
            
            # إضافة البيانات الوصفية
            if include_metadata:
                metadata = self._get_metadata(model)
                onnx_model.metadata_props.extend([
                    onnx.StringStringEntryProto(key=k, value=str(v))
                    for k, v in metadata.items()
                ])
            
            # حفظ النموذج
            onnx.save(onnx_model, str(file_path))
        except Exception as e:
            logger.error(f"خطأ في تصدير ONNX: {str(e)}")
            raise

    async def _export_json(
        self,
        model: Any,
        file_path: Path,
        include_metadata: bool,
        include_preprocessing: bool
    ) -> None:
        """تصدير النموذج بتنسيق JSON"""
        export_data = {
            'model_params': self._get_model_params(model),
            'metadata': self._get_metadata(model) if include_metadata else None,
            'preprocessing': self._get_preprocessing(model) if include_preprocessing else None
        }
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    async def _export_pickle(
        self,
        model: Any,
        file_path: Path,
        include_metadata: bool,
        include_preprocessing: bool
    ) -> None:
        """تصدير النموذج بتنسيق pickle"""
        import pickle
        export_data = {
            'model': model,
            'metadata': self._get_metadata(model) if include_metadata else None,
            'preprocessing': self._get_preprocessing(model) if include_preprocessing else None
        }
        with open(file_path, 'wb') as f:
            pickle.dump(export_data, f)

    def _get_metadata(self, model: Any) -> Dict[str, Any]:
        """استخراج البيانات الوصفية للنموذج"""
        return {
            'type': type(model).__name__,
            'params': getattr(model, 'get_params', lambda: {})(),
            'features': getattr(model, 'feature_names_', []),
            'target': getattr(model, 'target_names_', []),
            'timestamp': datetime.now().isoformat()
        }

    def _get_preprocessing(self, model: Any) -> Dict[str, Any]:
        """استخراج خطوات المعالجة المسبقة"""
        return {
            'scalers': getattr(model, 'preprocessing_steps_', {}),
            'encoders': getattr(model, 'encoding_steps_', {}),
            'feature_selection': getattr(model, 'feature_selection_', {})
        }

    def _get_model_params(self, model: Any) -> Dict[str, Any]:
        """استخراج معلمات النموذج"""
        return {
            'params': getattr(model, 'get_params', lambda: {})(),
            'coefficients': getattr(model, 'coef_', []).tolist() if hasattr(model, 'coef_') else None,
            'intercept': getattr(model, 'intercept_', None),
            'classes': getattr(model, 'classes_', []).tolist() if hasattr(model, 'classes_') else None
        }

    def _convert_to_onnx(self, model: Any) -> onnx.ModelProto:
        """تحويل النموذج إلى تنسيق ONNX"""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # تحديد أبعاد المدخلات
            initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
            
            # تحويل النموذج
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            return onnx_model
        except Exception as e:
            logger.error(f"خطأ في تحويل النموذج إلى ONNX: {str(e)}")
            raise

    def _compress_file(self, file_path: Path, compression: str) -> Path:
        """ضغط ملف التصدير"""
        import shutil
        
        compressed_path = file_path.with_suffix(f"{file_path.suffix}.{compression}")
        if compression == 'zip':
            shutil.make_archive(str(file_path), 'zip', file_path.parent, file_path.name)
        elif compression == 'gzip':
            import gzip
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        return compressed_path

    def _encrypt_file(self, file_path: Path, encryption: Dict[str, Any]) -> Path:
        """تشفير ملف التصدير"""
        from cryptography.fernet import Fernet
        
        encrypted_path = file_path.with_suffix(f"{file_path.suffix}.encrypted")
        key = encryption.get('key') or Fernet.generate_key()
        
        f = Fernet(key)
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = f.encrypt(file_data)
        with open(encrypted_path, 'wb') as file:
            file.write(encrypted_data)
        
        # حفظ المفتاح في ملف منفصل
        key_path = encrypted_path.with_suffix('.key')
        with open(key_path, 'wb') as key_file:
            key_file.write(key)
        
        return encrypted_path 