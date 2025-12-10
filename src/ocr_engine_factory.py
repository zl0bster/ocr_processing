"""Factory for creating different types of PaddleOCR engines."""

from __future__ import annotations

import inspect
import logging
from typing import Literal

from paddleocr import PaddleOCR

from config.settings import Settings


class OCREngineFactory:
    """Factory for creating PaddleOCR engines with different configurations."""

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
        
        # Build desired parameters
        ocr_params = {
            'use_angle_cls': cls,  # Enable text angle classification if cls is True
            'lang': settings.ocr_language,
            'show_log': False,  # Suppress PaddleOCR internal logging
            'det_limit_side_len': settings.ocr_det_limit_side_len,
            'det_limit_type': 'max',
        }

        # Add GPU parameter only if enabled
        if settings.ocr_use_gpu:
            ocr_params['use_gpu'] = True

        # Filter out unsupported parameters by checking PaddleOCR.__init__ signature
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
            # New version supports det/rec/cls parameters
            try:
                ocr_engine = PaddleOCR(det=det, rec=rec, cls=cls, **ocr_params)
            except Exception as e:
                logger.warning(
                    "Failed to initialize %s engine with det/rec/cls parameters: %s",
                    engine_type, e
                )
                # Fallback to standard initialization
                ocr_engine = PaddleOCR(**ocr_params)
        else:
            # Old version - use standard initialization (full engine only)
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
                    engine_type, e
                )
                logger.info("Falling back to minimal PaddleOCR initialization...")
                try:
                    ocr_engine = PaddleOCR(lang=settings.ocr_language)
                except Exception as e2:
                    logger.error(
                        "Failed to initialize %s engine even with minimal parameters: %s",
                        engine_type, e2
                    )
                    # Try with no parameters at all (will use defaults)
                    ocr_engine = PaddleOCR()
        
        logger.info("PaddleOCR %s engine initialized successfully", engine_type)
        return ocr_engine

