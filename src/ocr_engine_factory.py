"""Factory for creating different types of PaddleOCR engines.

FIXES APPLIED:
- ✅ Removed deprecated parameters: use_mp, total_process_num, det_limit_type
- ✅ Compatible with PaddleOCR 3.x API
- ✅ Proper parameter filtering by signature inspection
"""

from __future__ import annotations

import inspect
import logging
from typing import Literal

from paddleocr import PaddleOCR

from config.settings import Settings


class OCREngineFactory:
    """Factory for creating PaddleOCR engines with different configurations.
    
    FIXES:
    - Removed deprecated parameters: use_mp, total_process_num, det_limit_type
    - Compatible with PaddleOCR 3.x API
    - Proper parameter filtering by signature inspection
    """

    @staticmethod
    def create_full_engine(settings: Settings, logger: logging.Logger) -> PaddleOCR:
        """Create full OCR engine with detection, recognition, and classification.

        Args:
            settings: Application settings
            logger: Logger instance

        Returns:
            Initialized PaddleOCR engine
        """
        return OCREngineFactory._create_engine(
            settings, logger, det=True, rec=True, cls=True, engine_type="full"
        )

    @staticmethod
    def create_detection_engine(settings: Settings, logger: logging.Logger) -> PaddleOCR:
        """Create detection-only OCR engine.

        Args:
            settings: Application settings
            logger: Logger instance

        Returns:
            Initialized PaddleOCR engine for detection only
        """
        return OCREngineFactory._create_engine(
            settings, logger, det=True, rec=False, cls=False, engine_type="detection"
        )

    @staticmethod
    def create_recognition_engine(settings: Settings, logger: logging.Logger) -> PaddleOCR:
        """Create recognition-only OCR engine.

        Args:
            settings: Application settings
            logger: Logger instance

        Returns:
            Initialized PaddleOCR engine for recognition only
        """
        return OCREngineFactory._create_engine(
            settings, logger, det=False, rec=True, cls=True, engine_type="recognition"
        )

    @staticmethod
    def _create_engine(
        settings: Settings,
        logger: logging.Logger,
        det: bool,
        rec: bool,
        cls: bool,
        engine_type: str,
    ) -> PaddleOCR:
        """Internal method to create OCR engine with specified configuration.

        Args:
            settings: Application settings
            logger: Logger instance
            det: Enable detection
            rec: Enable recognition
            cls: Enable classification
            engine_type: Type description for logging

        Returns:
            Initialized PaddleOCR engine
        """
        logger.info("Initializing PaddleOCR %s engine (det=%s, rec=%s, cls=%s)...", 
                   engine_type, det, rec, cls)

        # Suppress warnings during initialization
        import warnings
        import os
        
        # Suppress PaddlePaddle warnings about ccache
        os.environ.setdefault('PADDLE_SILENT', '1')
        
        # ✅ FIX: Правильные параметры для PaddleOCR 3.x
        # УДАЛЕНЫ: use_mp, total_process_num, det_limit_type, show_log
        ocr_params = {
            'use_angle_cls': cls,  # ✅ Классификация углов при инициализации
            'lang': settings.ocr_language,
            # ❌ УДАЛЕНО: 'show_log': False,        # Удалён в PaddleOCR 3.x
            'det_limit_side_len': settings.ocr_det_limit_side_len,
            # ❌ УДАЛЕНО: 'det_limit_type': 'max',  # Удалён в PaddleOCR 3.x
            # ❌ УДАЛЕНО: 'use_mp': False,          # Удалён в PaddleOCR 3.x
            # ❌ УДАЛЕНО: 'total_process_num': 1,   # Удалён в PaddleOCR 3.x
        }

        # Add GPU parameter only if enabled
        if settings.ocr_use_gpu:
            ocr_params['use_gpu'] = True

        # ✅ FIX: Фильтровать параметры через сигнатуру функции
        # Это обеспечит совместимость с разными версиями PaddleOCR
        try:
            sig = inspect.signature(PaddleOCR.__init__)
            supported_params = set(sig.parameters.keys())

            # Filter params to only include supported ones
            filtered_params = {}
            skipped_params = []

            for key, value in ocr_params.items():
                if key in supported_params:
                    filtered_params[key] = value
                else:
                    skipped_params.append(key)

            # Log skipped parameters for debugging
            if skipped_params:
                logger.debug(
                    "Skipping unsupported PaddleOCR parameters: %s",
                    ", ".join(skipped_params)
                )

            ocr_params = filtered_params

        except Exception as e:
            logger.debug(
                "Could not inspect PaddleOCR signature: %s. Using all parameters.", e
            )

        # Check if det/rec/cls parameters are supported
        try:
            sig = inspect.signature(PaddleOCR.__init__)
            supports_det_rec = 'det' in sig.parameters and 'rec' in sig.parameters
        except Exception:
            supports_det_rec = False

        if supports_det_rec:
            # ✅ Версия 3.x с поддержкой det/rec/cls параметров
            try:
                logger.debug("Initializing with det=%s, rec=%s, cls=%s", det, rec, cls)
                ocr_engine = PaddleOCR(det=det, rec=rec, cls=cls, **ocr_params)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Failed to initialize %s engine with det/rec/cls parameters: %s. "
                    "Falling back to standard initialization.",
                    engine_type, str(e)[:200]
                )

                # Fallback to standard initialization without det/rec/cls
                try:
                    ocr_engine = PaddleOCR(**ocr_params)
                except Exception as e2:
                    logger.error(
                        "Standard initialization also failed: %s. Trying minimal params.",
                        str(e2)[:200]
                    )

                    # Last resort: minimal parameters
                    ocr_engine = PaddleOCR(
                        lang=settings.ocr_language
                    )
        else:
            # ✅ Старая версия без поддержки det/rec/cls
            logger.debug(
                "PaddleOCR version does not support det/rec/cls parameters, "
                "using full engine for all modes"
            )

            try:
                ocr_engine = PaddleOCR(**ocr_params)
            except Exception as e:
                # Try with minimal parameters if advanced ones fail
                logger.warning(
                    "Failed to initialize %s engine with advanced parameters: %s",
                    engine_type, str(e)[:200]
                )

                logger.info("Falling back to minimal PaddleOCR initialization...")

                try:
                    # Try without advanced stability params
                    minimal_params = {
                        'lang': settings.ocr_language,
                    }

                    if settings.ocr_use_gpu:
                        minimal_params['use_gpu'] = True

                    ocr_engine = PaddleOCR(**minimal_params)

                except Exception as e2:
                    logger.error(
                        "Failed to initialize %s engine even with minimal parameters: %s. "
                        "Using bare defaults.",
                        engine_type, str(e2)[:200]
                    )

                    # Last resort: absolute bare minimum
                    ocr_engine = PaddleOCR(lang=settings.ocr_language)
        
        logger.info("PaddleOCR %s engine initialized successfully", engine_type)
        return ocr_engine

