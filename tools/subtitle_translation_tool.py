"""
Screen OCR and subtitle translation helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SubtitleTranslatorConfig:
    """Configuration for screen OCR and Tencent Cloud translation."""

    tencent_secret_id: str = ""
    tencent_secret_key: str = ""
    tencent_region: str = "ap-guangzhou"
    tencent_project_id: int = 0
    ocr_language: str = "chi_sim+eng"
    stable_duration: float = 2.0
    capture_interval: float = 0.3
    source_lang: str = "auto"
    target_lang: str = "zh"
    tesseract_cmd: str | None = None

    @classmethod
    def from_env(cls) -> "SubtitleTranslatorConfig":
        """Build config from environment variables."""
        return cls(
            tencent_secret_id=os.getenv("TENCENT_SECRET_ID", ""),
            tencent_secret_key=os.getenv("TENCENT_SECRET_KEY", ""),
            tencent_region=os.getenv("TENCENT_REGION", "ap-guangzhou"),
            tencent_project_id=int(os.getenv("TENCENT_PROJECT_ID", "0")),
            ocr_language=os.getenv("OCR_LANGUAGE", "chi_sim+eng"),
            stable_duration=float(os.getenv("SUBTITLE_STABLE_DURATION", "2.0")),
            capture_interval=float(os.getenv("SUBTITLE_CAPTURE_INTERVAL", "0.3")),
            source_lang=os.getenv("SUBTITLE_SOURCE_LANG", "auto"),
            target_lang=os.getenv("SUBTITLE_TARGET_LANG", "zh"),
            tesseract_cmd=os.getenv("TESSERACT_CMD") or None,
        )


class TencentTranslator:
    """Tencent Cloud Machine Translation client."""

    def __init__(self, config: SubtitleTranslatorConfig | None = None) -> None:
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.tmt.v20180321 import tmt_client
        except ImportError as import_error:
            raise ImportError(
                "Tencent translation requires tencentcloud-sdk-python. "
                "Install with: pip install tencentcloud-sdk-python"
            ) from import_error

        self.config = config or SubtitleTranslatorConfig.from_env()
        if not self.config.tencent_secret_id or not self.config.tencent_secret_key:
            raise ValueError("Set TENCENT_SECRET_ID and TENCENT_SECRET_KEY before translating.")

        self._credential_module = credential
        self._tmt_client_module = tmt_client

        cloud_credential = credential.Credential(
            self.config.tencent_secret_id,
            self.config.tencent_secret_key,
        )
        http_profile = HttpProfile()
        http_profile.endpoint = "tmt.tencentcloudapi.com"
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        self.client = tmt_client.TmtClient(
            cloud_credential,
            self.config.tencent_region,
            client_profile,
        )

    def translate(
        self,
        text: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        untranslated_text: str | None = None,
    ) -> dict[str, Any]:
        """Translate text and return a normalized result dictionary."""
        try:
            from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
                TencentCloudSDKException,
            )
            from tencentcloud.tmt.v20180321 import models
        except ImportError as import_error:
            raise ImportError(
                "Tencent translation requires tencentcloud-sdk-python. "
                "Install with: pip install tencentcloud-sdk-python"
            ) from import_error

        source_text = text.strip()
        if not source_text:
            return {
                "target_text": "",
                "source": "",
                "target": "",
                "used_amount": 0,
                "error": "Text is empty",
            }

        try:
            request = models.TextTranslateRequest()
            params: dict[str, Any] = {
                "SourceText": source_text,
                "Source": source_lang or self.config.source_lang,
                "Target": target_lang or self.config.target_lang,
                "ProjectId": self.config.tencent_project_id,
            }
            if untranslated_text:
                params["UntranslatedText"] = untranslated_text
            request.from_json_string(json.dumps(params, ensure_ascii=False))
            response = self.client.TextTranslate(request)
            return {
                "target_text": response.TargetText,
                "source": response.Source,
                "target": response.Target,
                "used_amount": getattr(response, "UsedAmount", 0),
                "error": None,
            }
        except TencentCloudSDKException as exception:
            return {
                "target_text": "",
                "source": "",
                "target": "",
                "used_amount": 0,
                "error": f"Tencent API error: {exception}",
            }
        except Exception as exception:
            return {
                "target_text": "",
                "source": "",
                "target": "",
                "used_amount": 0,
                "error": f"Translation error: {exception}",
            }


class ScreenOCR:
    """OCR text from images or screen regions using Tesseract."""

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        default_lang: str = "chi_sim+eng",
    ) -> None:
        try:
            import pytesseract
        except ImportError as import_error:
            raise ImportError("Screen OCR requires pytesseract. Install with: pip install pytesseract") from import_error

        self.default_lang = default_lang
        self._pytesseract = pytesseract

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self._ensure_tesseract()

    def _ensure_tesseract(self) -> None:
        try:
            self._pytesseract.get_tesseract_version()
            return
        except Exception:
            pass

        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for tesseract_path in common_paths:
            try:
                self._pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self._pytesseract.get_tesseract_version()
                return
            except Exception:
                continue

    def capture_screen_region(self, bbox: tuple[int, int, int, int]) -> Any | None:
        """Capture a screen region as a PIL image."""
        try:
            from PIL import ImageGrab

            return ImageGrab.grab(bbox=bbox)
        except Exception as exception:
            print(f"Screen capture failed: {exception}")
            return None

    def preprocess_image(self, image: Any) -> Any:
        """Preprocess an image to improve OCR accuracy."""
        try:
            import cv2
            import numpy as numpy_module
            from PIL import Image
        except ImportError as import_error:
            raise ImportError(
                "OCR preprocessing requires opencv-python, numpy, and Pillow."
            ) from import_error

        image_array = numpy_module.array(image)
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        return Image.fromarray(denoised)

    def extract_text(
        self,
        image: Any,
        lang: str | None = None,
        preprocess: bool = True,
    ) -> str:
        """Extract OCR text from a PIL image."""
        if image is None:
            return ""
        try:
            processed_image = self.preprocess_image(image) if preprocess else image
            return self._pytesseract.image_to_string(
                processed_image,
                lang=lang or self.default_lang,
            ).strip()
        except Exception as exception:
            print(f"OCR failed: {exception}")
            return ""

    def capture_and_extract(
        self,
        bbox: tuple[int, int, int, int],
        lang: str | None = None,
        preprocess: bool = True,
    ) -> str:
        """Capture a screen region and extract OCR text."""
        return self.extract_text(
            self.capture_screen_region(bbox),
            lang=lang,
            preprocess=preprocess,
        )


class StabilityDetector:
    """Detect when OCR text in a screen region remains stable long enough."""

    def __init__(
        self,
        stable_duration: float = 2.0,
        capture_interval: float = 0.3,
        ocr: ScreenOCR | None = None,
    ) -> None:
        self.stable_duration = stable_duration
        self.capture_interval = capture_interval
        self.ocr = ocr or ScreenOCR()
        self.last_text = ""
        self.last_hash = ""
        self.stable_start_time: float | None = None
        self.is_stable = False

    @staticmethod
    def get_content_hash(text: str) -> str:
        """Return an MD5 hash for text content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def reset(self) -> None:
        """Reset tracked stability state."""
        self.last_text = ""
        self.last_hash = ""
        self.stable_start_time = None
        self.is_stable = False

    def check_stability(
        self,
        bbox: tuple[int, int, int, int],
        lang: str | None = None,
        preprocess: bool = True,
    ) -> tuple[bool, str]:
        """Check whether text in a region has stabilized."""
        current_text = self.ocr.capture_and_extract(bbox, lang=lang, preprocess=preprocess)
        current_hash = self.get_content_hash(current_text)
        current_time = time.time()

        if current_hash == self.last_hash and current_text.strip():
            if self.stable_start_time is None:
                self.stable_start_time = current_time
            if current_time - self.stable_start_time >= self.stable_duration:
                if not self.is_stable:
                    self.is_stable = True
                    return True, current_text
                return False, current_text
            return False, current_text

        self.last_text = current_text
        self.last_hash = current_hash
        self.stable_start_time = None
        self.is_stable = False
        return False, current_text

    def monitor_region(
        self,
        bbox: tuple[int, int, int, int],
        callback: Callable[[str], None],
        lang: str | None = None,
        preprocess: bool = True,
        stop_flag: Callable[[], bool] | None = None,
    ) -> None:
        """Monitor a region and call ``callback`` whenever stable text appears."""
        self.reset()
        while True:
            if stop_flag and stop_flag():
                break
            is_stable, text = self.check_stability(bbox, lang=lang, preprocess=preprocess)
            if is_stable:
                callback(text)
                self.reset()
            time.sleep(self.capture_interval)


def capture_translate_once(
    bbox: tuple[int, int, int, int],
    config: SubtitleTranslatorConfig | None = None,
    preprocess: bool = True,
) -> dict[str, Any]:
    """OCR one screen region and translate it once."""
    active_config = config or SubtitleTranslatorConfig.from_env()
    ocr = ScreenOCR(
        tesseract_cmd=active_config.tesseract_cmd,
        default_lang=active_config.ocr_language,
    )
    text = ocr.capture_and_extract(
        bbox,
        lang=active_config.ocr_language,
        preprocess=preprocess,
    )
    translator = TencentTranslator(active_config)
    translation = translator.translate(text)
    return {"source_text": text, "translation": translation}