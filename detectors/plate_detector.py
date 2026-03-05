"""
Best-effort license plate recognizer using EasyOCR.

Designed to be modular and optional: if easyocr is not installed,
PlateDetector degrades gracefully (returns None for all crops).

Multi-frame voting: accumulate OCR results over PLATE_VOTE_WINDOW frames
per track_id and return the most-common non-empty string.
"""
from __future__ import annotations
import re
import sys
from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np

try:
    from config import PLATE_CROP_EXPAND, PLATE_VOTE_WINDOW, PLATE_MIN_CONFIDENCE
except ImportError:
    PLATE_CROP_EXPAND = 0.15
    PLATE_VOTE_WINDOW = 30
    PLATE_MIN_CONFIDENCE = 0.30

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False


def _clean_plate(text: str) -> str:
    """공백 제거 후 한글·영문 대문자·숫자·하이픈만 유지. 3자 미만이면 빈 문자열 반환."""
    cleaned = re.sub(r"[^\uAC00-\uD7A3A-Z0-9\-]", "", text.upper().strip())
    return cleaned if len(cleaned) >= 4 else ""


def _normalize_plate(text: str) -> str:
    """한국 번호판 형태로 정규화(4가지 룰을 종합)."""
    raw = _clean_plate(text).replace("-", "")
    if not raw:
        return ""

    # 1) 원문 그대로 패턴 매칭
    m = re.findall(r"(?:[가-힣]{2})?(\d{2,3}[가-힣]\d{4})", raw)
    if m:
        return m[-1]

    # 2) OCR 오인식 보정(숫자 구간에서만): O/D/Q->0, I/L->1, Z->2, S->5, B->8
    trans = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"})
    parts = re.split(r"([가-힣])", raw)
    corrected_parts: list[str] = []
    for part in parts:
        if re.fullmatch(r"[가-힣]", part):
            corrected_parts.append(part)
        else:
            corrected_parts.append(part.translate(trans))
    corrected = "".join(corrected_parts)
    m = re.findall(r"(?:[가-힣]{2})?(\d{2,3}[가-힣]\d{4})", corrected)
    if m:
        return m[-1]

    # 3) 한글 1글자를 중심으로 앞 2~3자리, 뒤 4자리 복원 시도
    for m in re.finditer(r"([가-힣])", corrected):
        i = m.start()
        left = re.sub(r"\D", "", corrected[:i])
        right = re.sub(r"\D", "", corrected[i+1:])
        if len(right) >= 4 and len(left) >= 2:
            front = left[-3:] if len(left) >= 3 else left[-2:]
            if len(front) in (2, 3):
                return f"{front}{corrected[i]}{right[:4]}"

    # 4) 완전 패턴은 아니어도 길이가 충분하면 투표용 후보 유지
    return corrected if len(corrected) >= 7 else ""


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """번호판 OCR 정확도 향상을 위한 전처리.

    - 최소 높이 80px 보장 (EasyOCR는 큰 이미지에서 더 정확)
    - CLAHE로 대비 향상 (역광·야간 번호판에 효과적)
    """
    h, w = crop.shape[:2]
    # 높이 기준 최소 80px로 업스케일
    scale = max(1.0, 80.0 / h)
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    # CLAHE 대비 향상 (그레이스케일 채널)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class PlateDetector:
    """
    Per-vehicle plate text estimator with multi-frame voting.

    Usage:
        detector = PlateDetector()
        # each frame, for vehicles of interest:
        detector.update(track_id, frame, bbox_xyxy)
        # after enough frames:
        plate = detector.get_best(track_id)
    """

    def __init__(self, languages: list[str] | None = None) -> None:
        self._reader = None
        try:
            from config import PLATE_LANGUAGES
            default_langs = PLATE_LANGUAGES
        except (ImportError, AttributeError):
            default_langs = ["ko", "en"]
        self._languages = languages or default_langs
        self._history: dict[int, deque] = {}

    def _ensure_reader(self) -> bool:
        if not _EASYOCR_AVAILABLE:
            print("[PlateDetector] easyocr 미설치 — 번호판 인식 비활성화", file=sys.stderr)
            return False
        if self._reader is None:
            try:
                self._reader = easyocr.Reader(self._languages, gpu=self._gpu_available())
            except Exception as e:
                print(f"[PlateDetector] Reader 초기화 실패: {e}", file=sys.stderr)
                return False
        return True

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _plate_roi_crops(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> list[np.ndarray]:
        """번호판 위치 휴리스틱으로 후보 크롭 목록 반환.

        한국 번호판은 차량 전·후면 하단 중앙에 위치.
        대시캠에서 보이는 컷인 차량(측·전면)을 고려해 세 영역 탐색.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        crops: list[np.ndarray] = []

        # 1. 하단 40% 전체 너비 (번호판 주요 위치)
        py1 = max(0, y1 + int(bh * 0.55))
        py2 = min(h, y2 + int(bh * 0.05))
        px1 = max(0, x1 - int(bw * 0.05))
        px2 = min(w, x2 + int(bw * 0.05))
        c = frame[py1:py2, px1:px2]
        if c.size > 0:
            crops.append(c)

        # 2. 하단 40% 중앙 집중 (번호판 폭은 차체의 ~50%)
        cx = (x1 + x2) // 2
        hw = max(int(bw * 0.40), 60)
        c2 = frame[py1:py2, max(0, cx - hw):min(w, cx + hw)]
        if c2.size > 0:
            crops.append(c2)

        # 3. 전체 bbox 폴백
        exp = int(min(bw, bh) * 0.05)
        c3 = frame[max(0, y1 - exp):min(h, y2 + exp),
                   max(0, x1 - exp):min(w, x2 + exp)]
        if c3.size > 0:
            crops.append(c3)

        return crops

    @staticmethod
    def _ocr_variants(crop: np.ndarray) -> list[np.ndarray]:
        """OCR 입력 4종(raw/clahe/threshold/invert-threshold) 생성."""
        clahe_bgr = _preprocess_crop(crop)
        gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 7
        )
        thr_inv = cv2.bitwise_not(thr)
        return [
            crop,
            clahe_bgr,
            cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(thr_inv, cv2.COLOR_GRAY2BGR),
        ]

    def update(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> None:
        """Add one frame's OCR candidate for a vehicle."""
        if not self._ensure_reader():
            return
        crops = self._plate_roi_crops(frame, bbox_xyxy)
        hist = self._history.setdefault(track_id, deque(maxlen=PLATE_VOTE_WINDOW))
        for crop in crops:
            for variant in self._ocr_variants(crop):
                try:
                    results = self._reader.readtext(variant, detail=1, paragraph=False)
                except Exception:
                    continue
                for (_bbox, text, conf) in results:
                    if conf < PLATE_MIN_CONFIDENCE:
                        continue
                    cleaned = _normalize_plate(text)
                    if cleaned:
                        hist.append(cleaned)

    def get_best(self, track_id: int) -> Optional[str]:
        """Return the most-common plate string seen so far (min 2 votes), or None."""
        hist = self._history.get(track_id)
        if not hist:
            return None
        counter = Counter(hist)
        most_common, count = counter.most_common(1)[0]
        # 기본: 2회 이상 동일 문자열
        if count >= 2:
            return most_common

        # 이벤트가 짧은 경우(히스토리 짧음) 단 1표라도 패턴이 명확하면 반환
        if re.fullmatch(r"\d{2,3}[가-힣]\d{4}", most_common):
            return most_common
        return None

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)

    def reset(self) -> None:
        self._history.clear()
        self._reader = None
