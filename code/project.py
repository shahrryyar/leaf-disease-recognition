"""
================================================================================
 Leaf Disease Severity Analyzer - شیکەرەوەی ڕادەی نەخۆشی گەڵا
 Computer Vision Final Project - پڕۆژەی کۆتایی بینینی کۆمپیوتەری
 Author : Shahryar Sabir Muhammed - نوسەر: شەهریار سابیر محەمەد
================================================================================
"""

from __future__ import annotations # ڕێگەدان بە بەکارهێنانی جۆرەکان پێش پێناسەکردنیان

import os # بۆ کارکردن لەگەڵ فایل و بوخچەکان
import csv # بۆ خەزنکردنی داتا لە فایلی ئێکسڵ
import glob # بۆ گەڕان بەدوای فایلەکاندا
import json # بۆ کارکردن لەگەڵ فایلی جەیسۆن
import time # بۆ پێوانەکردنی کات
import argparse # بۆ وەرگرتنی فەرمان لە تێرمیناڵەوە
from dataclasses import dataclass, asdict, field # بۆ دروستکردنی کلاسی داتا بە ئاسانی
import logging # بۆ تۆمارکردنی زانیارییەکان و هەڵەکان

logger = logging.getLogger(__name__) # دروستکردنی لۆگەر بۆ ئەم فایلە

import cv2 # کتێبخانەی ئۆپن سی ڤی بۆ بینینی کۆمپیوتەری
import numpy as np # بۆ کارکردن لەگەڵ لیستە ژمارەییەکان و ماتریکس

# --------------------------------------------------------------------------- #
# Configuration - ڕێکخستنەکان
# --------------------------------------------------------------------------- #

PROCESS_WIDTH = 640          # پانی وێنەکە دەکرێت بە ٦٤٠ بۆ ئەوەی خێرا بێت
MIN_LESION_AREA = 25         # کەمترین قەبارەی برین کە حسابی بۆ بکرێت (پێکسڵ)
OVERLAY_ALPHA = 0.45         # ڕادەی شەفافی ڕەنگە سورەکە لەسەر گەڵاکە

# مەودای ڕەنگی گەڵای تەندروست لە سیستەمی HSV
HEALTHY_LOWER = np.array([30, 35, 30], np.uint8) # کەمترین ڕادەی ڕەنگی سەوز
HEALTHY_UPPER = np.array([90, 255, 255], np.uint8) # زۆرترین ڕادەی ڕەنگی سەوز

# دیاریکردنی زەردبوون و مردنی شانەکان لە ناوچە نەخۆشەکاندا
CHLOROSIS_LOWER = np.array([18, 40, 60], np.uint8)   # کەمترین ڕادەی ڕەنگی زەرد
CHLOROSIS_UPPER = np.array([34, 255, 255], np.uint8) # زۆرترین ڕادەی ڕەنگی زەرد

# ئاستەکانی سەختی نەخۆشییەکە بەپێی ڕێژەی سەدی
SEVERITY_BANDS = [
    (5.0,   "Healthy",  (0, 170, 0)),    # کەمتر لە ٥٪ - تەندروست
    (20.0,  "Mild",     (0, 215, 215)),  # کەمتر لە ٢٠٪ - کەم
    (40.0,  "Moderate", (0, 140, 255)),  # کەمتر لە ٤٠٪ - مامناوەند
    (101.0, "Severe",   (0, 0, 230)),    # زیاتر - سەخت
]

# ڕەنگەکان بۆ جوانکاری (BGR)
C_BG = (38, 38, 38) # ڕەنگی پاشبنەما
C_TEXT = (245, 245, 245) # ڕەنگی نوسین
C_LESION = (0, 0, 255) # ڕەنگی نیشانەکان (سور)
C_OUTLINE = (20, 20, 20) # ڕەنگی دەوری نیشانەکان


# --------------------------------------------------------------------------- #
# Result container - قابی دەرەنجامەکان
# --------------------------------------------------------------------------- #

@dataclass
class LeafMetrics:
    """کلاسێک بۆ هەڵگرتنی پێوەرە ژمارەییەکان بۆ هەر گەڵایەک"""
    name: str = "" # ناوی فایلەکە
    leaf_area_px: int = 0 # ڕووبەری گەڵاکە بە پێکسڵ
    diseased_px: int = 0 # ڕووبەری نەخۆش بە پێکسڵ
    diseased_percent: float = 0.0 # ڕێژەی سەدی نەخۆشی
    healthy_percent: float = 0.0 # ڕێژەی سەدی تەندروستی
    chlorosis_percent: float = 0.0 # ڕێژەی سەدی زەردبوون
    necrosis_percent: float = 0.0 # ڕێژەی سەدی مردنی شانەکان
    lesion_count: int = 0 # ژمارەی برینەکان
    largest_lesion_percent: float = 0.0 # ڕێژەی گەورەترین برین بەرامبەر گەڵاکە
    severity: str = "Unknown" # ئاستی نەخۆشی (ناو)
    severity_grade: int = 0 # ئاستی نەخۆشی (ژمارە ٠-٩)
    disease: str = "" # ناوی نەخۆشییەکە
    disease_confidence: float = 0.0 # ڕادەی دڵنیایی لە جۆری نەخۆشییەکە


@dataclass
class LeafResult:
    """کلاسێک بۆ هەڵگرتنی پێوەرەکان و وێنە جیاوازەکانی قۆناغەکانی شیکردنەوە"""
    metrics: LeafMetrics # پێوەرە ژمارەییەکان
    stages: dict = field(default_factory=dict) # وێنەی قۆناغەکان
    masks: dict = field(default_factory=dict) # ماسکە ڕەش و سپییەکان
    image: object = None # وێنە سەرەکییەکە دوای بچوککردنەوە


# --------------------------------------------------------------------------- #
# The analyzer - شیکەرەوەی سەرەکی
# --------------------------------------------------------------------------- #

class LeafDiseaseAnalyzer:
    """کلاسێک کە هەموو هەنگاوەکانی شیکردنەوەی نەخۆشی گەڵا لەخۆ دەگرێت"""

    def __init__(self,
                 healthy_lower=HEALTHY_LOWER, healthy_upper=HEALTHY_UPPER,
                 min_lesion_area=MIN_LESION_AREA, use_grabcut=True):
        # دیاریکردنی ڕێکخستنە سەرەتاییەکان لە کاتی دروستکردنی ئۆبجێکتەکەدا
        self.healthy_lower = np.asarray(healthy_lower, np.uint8)
        self.healthy_upper = np.asarray(healthy_upper, np.uint8)
        self.min_lesion_area = min_lesion_area
        self.use_grabcut = use_grabcut # بەکارهێنانی تەکنیکی GrabCut بۆ جیاکردنەوەی گەڵاکە
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # فیلتەری بچوک
        self._kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # فیلتەری گەورە

    # ---- individual stages - قۆناغە تاکەکەسییەکان --------------------------- #

    @staticmethod
    def _resize(image):
        """بچوککردنەوەی وێنەکە بۆ پانییەکی دیاریکراو بە پاراستنی ڕێژەی بەرزی"""
        h, w = image.shape[:2]
        scale = PROCESS_WIDTH / float(w)
        return cv2.resize(image, (PROCESS_WIDTH, max(1, int(h * scale))))

    @staticmethod
    def _normalize_illumination(bgr):
        """ڕێکخستنی ڕووناکی وێنەکە بۆ ئەوەی سێبەر و شوێنە زۆر ڕووناکەکان کاریگەری نەکەن"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) # گۆڕین بۆ سیستەمی HSV
        h, s, v = cv2.split(hsv) # جیاکردنەوەی کەناڵەکان
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # دروستکردنی فیلتەری ڕێکخستنی ڕووناکی
        v = clahe.apply(v) # جێبەجێکردنی لەسەر کەناڵی ڕووناکی
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR) # تێکەڵکردنەوە و گۆڕین بۆ BGR

    def _largest_filled(self, mask):
        """پاککردنەوەی ماسکەکە و تەنها هێشتنەوەی گەورەترین پارچە (کە گەڵاکەیە)"""
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_big, iterations=2) # پڕکردنەوەی کونە بچوکەکان
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_big, iterations=1) # سڕینەوەی خاڵە بچوکەکان
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE) # دۆزینەوەی دەورەبەری پارچەکان
        if contours:
            largest = max(contours, key=cv2.contourArea) # دۆزینەوەی گەورەترین پارچە
            mask = np.zeros_like(mask) # دروستکردنی وێنەیەکی ڕەشی نوێ
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED) # کێشان و پڕکردنەوەی گەورەترین پارچە
        return mask

    def _grabcut_fg(self, bgr):
        """جیاکردنەوەی گەڵاکە لە پاشبنەمایەکی ئاڵۆز بە بەکارهێنانی تەکنیکی GrabCut"""
        h, w = bgr.shape[:2]
        gc = np.full((h, w), cv2.GC_PR_FGD, np.uint8) # هەموو وێنەکە وەک گەڵای ئەگەری دادەنێین
        b = max(2, int(min(h, w) * 0.03)) # دیاریکردنی لێوارەکان
        gc[:b, :] = gc[-b:, :] = gc[:, :b] = gc[:, -b:] = cv2.GC_BGD # لێوارەکان بە پاشبنەمای دڵنیا دادەنێین
        cy, cx = h // 2, w // 2 # ناوەڕاستی وێنەکە
        ry, rx = int(h * 0.20), int(w * 0.20) # ناوچەیەکی ناوەڕاست
        gc[cy - ry:cy + ry, cx - rx:cx + rx] = cv2.GC_FGD # ناوەڕاست بە گەڵای دڵنیا دادەنێین
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64) # مۆدێلی ناوەکی GrabCut
        try:
            cv2.grabCut(bgr, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK) # جێبەجێکردنی کردارەکە
        except cv2.error:
            return np.zeros((h, w), np.uint8) # ئەگەر هەڵە هەبوو، وێنەیەکی بەتاڵ دەگەڕێنینەوە
        return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                        255, 0).astype(np.uint8) # گەڕاندنەوەی ئەو بەشانەی کە گەڵان

    def _segment_leaf(self, bgr, hsv):
        """دروستکردنی ماسکێکی ڕەش و سپی تەنها بۆ گەڵاکە"""
        if self.use_grabcut:
            fg = self._grabcut_fg(bgr) # جیاکردنەوە بە GrabCut
            if 0.04 * fg.size < cv2.countNonZero(fg) < 0.97 * fg.size: # پشکنینی ئەوەی کە ئایا ئەنجامەکە گونجاوە
                return self._largest_filled(fg)

        sat = hsv[:, :, 1] # بەکارهێنانی کەناڵی تێری ڕەنگ (Saturation)
        _, mask = cv2.threshold(sat, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU) # بەکارهێنانی شێوازی Otsu بۆ جیاکردنەوە
        return self._largest_filled(mask)

    def _split_tissue(self, hsv, leaf_mask):
        """جیاکردنەوەی بەشە تەندروستەکان و نەخۆشەکان لەناو گەڵاکەدا"""
        healthy = cv2.bitwise_and(
            cv2.inRange(hsv, self.healthy_lower, self.healthy_upper), leaf_mask) # دۆزینەوەی ڕەنگی سەوز لەناو گەڵاکەدا

        diseased = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(healthy)) # هەرچی گەڵایە و سەوز نییە، واتە نەخۆشە
        diseased = cv2.morphologyEx(diseased, cv2.MORPH_OPEN, self._kernel, iterations=1) # لابردنی خاڵە وردە زیادەکان
        diseased = cv2.morphologyEx(diseased, cv2.MORPH_CLOSE, self._kernel, iterations=1) # پڕکردنەوەی کونە وردەکان

        # جیاکردنەوەی جۆری نەخۆشییەکە
        chlorosis = cv2.bitwise_and(
            cv2.inRange(hsv, CHLOROSIS_LOWER, CHLOROSIS_UPPER), diseased) # دۆزینەوەی بەشە زەردەکان لەناو بەشە نەخۆشەکەدا
        necrosis = cv2.bitwise_and(diseased, cv2.bitwise_not(chlorosis)) # ئەوەی ماوەتەوە بە مردنی شانەکان دادەنرێت
        return healthy, diseased, chlorosis, necrosis

    @staticmethod
    def _severity(percent):
        """دیاریکردنی ئاستی سەختی نەخۆشییەکە و ڕەنگەکەی بەپێی ڕێژەی سەدی"""
        for limit, label, colour in SEVERITY_BANDS:
            if percent < limit:
                return label, colour
        return SEVERITY_BANDS[-1][1], SEVERITY_BANDS[-1][2]

    @staticmethod
    def _severity_grade(percent):
        """گۆڕینی ڕێژەی سەدی بۆ ئاستێکی ژمارەیی لە نێوان ٠ بۆ ٩"""
        cuts = [0, 1, 3, 6, 12, 25, 40, 60, 80]   # سنوورەکان بۆ پلەکان
        grade = 0
        for g, c in enumerate(cuts):
            if percent >= c:
                grade = g
        return min(grade + (1 if percent >= 80 else 0), 9)

    def _lesions(self, diseased_mask, leaf_area):
        """ژماردنی برینەکان و دۆزینەوەی گەورەترین برین"""
        contours, _ = cv2.findContours(diseased_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE) # دۆزینەوەی دەورەبەری برینەکان
        lesions = [c for c in contours if cv2.contourArea(c) >= self.min_lesion_area] # تەنها برینە گەورەکان وەردەگرین
        lesions.sort(key=cv2.contourArea, reverse=True) # ڕیزکردن لە گەورەوە بۆ بچوک
        largest_pct = 0.0
        if lesions and leaf_area > 0:
            largest_pct = cv2.contourArea(lesions[0]) / leaf_area * 100.0 # حسابی گەورەترین برین دەکەین
        return lesions, largest_pct

    # ---- public entry point - فەرمانی سەرەکی بۆ بەکارهێنەر ------------------ #

    def analyze(self, image, name="", classifier=None):
        """ئەمە فەرمانە سەرەکییەکەیە کە هەموو کردارەکان بەدوای یەکدا ئەنجام دەدات"""
        image = self._resize(image) # بچوککردنەوە
        norm = self._normalize_illumination(image) # ڕێکخستنی ڕووناکی
        blurred = cv2.GaussianBlur(norm, (5, 5), 0) # کەمکردنەوەی ژاوەژاو (Noise)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # گۆڕین بۆ HSV

        leaf_mask = self._segment_leaf(blurred, hsv) # جیاکردنەوەی گەڵا
        healthy, diseased, chlorosis, necrosis = self._split_tissue(hsv, leaf_mask) # جیاکردنەوەی شانەکان

        leaf_area = int(cv2.countNonZero(leaf_mask)) # ڕووبەری گشتی گەڵا
        diseased_px = int(cv2.countNonZero(diseased)) # ڕووبەری نەخۆش
        pct = (diseased_px / leaf_area * 100.0) if leaf_area else 0.0 # ڕێژەی سەدی نەخۆشی
        lesions, largest_pct = self._lesions(diseased, leaf_area) # شیکردنەوەی برینەکان
        label, _ = self._severity(pct) # دیاریکردنی ئاستی سەختی

        # کۆکردنەوەی هەموو داتاکان لە ناو ئۆبجێکتێکی LeafMetrics
        metrics = LeafMetrics(
            name=name,
            leaf_area_px=leaf_area,
            diseased_px=diseased_px,
            diseased_percent=round(pct, 2),
            healthy_percent=round(100.0 - pct, 2) if leaf_area else 0.0,
            chlorosis_percent=round(cv2.countNonZero(chlorosis) / leaf_area * 100.0, 2) if leaf_area else 0.0,
            necrosis_percent=round(cv2.countNonZero(necrosis) / leaf_area * 100.0, 2) if leaf_area else 0.0,
            lesion_count=len(lesions),
            largest_lesion_percent=round(largest_pct, 2),
            severity=label,
            severity_grade=self._severity_grade(pct),
        )

        masks = {"leaf": leaf_mask, "healthy": healthy, "diseased": diseased,
                 "chlorosis": chlorosis, "necrosis": necrosis}

        result = LeafResult(metrics=metrics, masks=masks, image=image)
        if classifier is not None:
            disease, conf = classifier.predict(result) # پێشبینیکردنی ناوی نەخۆشییەکە ئەگەر مۆدێلەکە هەبوو
            metrics.disease = disease
            metrics.disease_confidence = round(float(conf), 3)

        clean = self._annotate(image, diseased, lesions) # نیشانەکردنی وێنەکە
        result.stages = {
            "Original": image,
            "Illumination Normalized": norm,
            "Diagnosis": clean,                          # وێنەی نیشانەکراوی پاک
            "Result": self._banner(clean.copy(), metrics),  # وێنە لەگەڵ زانیارییەکان لە سەرەوەی
        }
        return result

    # ---- rendering - کێشانی ئەنجامەکان لەسەر وێنەکە ------------------------- #

    def _annotate(self, image, diseased_mask, lesions):
        """ڕەنگکردنی ناوچە نەخۆشەکان بە سور و ژمارەکردنی برینەکان"""
        out = image.copy()
        red = np.zeros_like(out)
        red[diseased_mask > 0] = C_LESION # دروستکردنی چینێکی سور
        out = cv2.addWeighted(out, 1.0, red, OVERLAY_ALPHA, 0) # تێکەڵکردنی سوری لەگەڵ وێنە سەرەکییەکە
        for i, c in enumerate(lesions, 1):
            cv2.drawContours(out, [c], -1, C_OUTLINE, 1) # کێشانی دەوری برینەکە
            x, y, _, _ = cv2.boundingRect(c)
            cv2.putText(out, str(i), (x, max(11, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA) # نوسینی ژمارەی برینەکە
        return out

    def _banner(self, img, m: LeafMetrics):
        """کێشانی شریتێکی زانیاری لە سەرەوەی وێنەکەدا"""
        h, w = img.shape[:2]
        bar_h = 86 # بەرزی شریتەکە
        bar = np.full((bar_h, w, 3), D_BG, np.uint8) # دروستکردنی شریتەکە
        out = np.vstack([bar, img])               # لکاندنی شریتەکە بە سەرەوەی وێنەکە

        sev_colour = self._severity(m.diseased_percent)[1] # ڕەنگی ئاستی سەختی
        verdict = f"{m.severity.upper()}  -  {m.diseased_percent:.1f}% AFFECTED" # نوسینی دەرەنجام
        cv2.putText(out, verdict, (12, 30), FONT, 0.74, sev_colour, 2, cv2.LINE_AA)

        if m.disease:
            dx = f"Likely: {m.disease} ({m.disease_confidence*100:.0f}%)" # نوسینی ناوی نەخۆشی ئەگەر هەبوو
            cv2.putText(out, dx, (w - 12 - _text_w(dx, 0.52), 28),
                        FONT, 0.52, D_ACCENT, 1, cv2.LINE_AA)

        # نوسینی پێوەرە وردەکان لە هێڵی دووەمدا
        line2 = (f"Grade {m.severity_grade}/9    Lesions {m.lesion_count}    "
                 f"Yellowing {m.chlorosis_percent:.1f}%    Necrosis {m.necrosis_percent:.1f}%"
                 f"    Largest {m.largest_lesion_percent:.1f}%")
        cv2.putText(out, line2, (12, 52), FONT, 0.44, D_SUB, 1, cv2.LINE_AA)

        _gauge(out, 12, 64, w - 24, 12, m.diseased_percent, sev_colour) # کێشانی پێوەرە ڕەنگاوڕەنگەکە
        return out


# --------------------------------------------------------------------------- #
# Output helpers - یاریدەدەرەکانی دەرەنجام
# --------------------------------------------------------------------------- #

# سیستەمی دیزاینی بینراو (BGR)
D_BG = (26, 28, 32)        # ڕەنگی پاشبنەمای تۆخ
D_CARD = (44, 48, 54)      # ڕەنگی کارتەکان
D_LINE = (72, 78, 86)      # ڕەنگی هێڵەکان
D_INK = (236, 238, 240)    # ڕەنگی نوسینی سەرەکی
D_SUB = (158, 164, 172)    # ڕەنگی نوسینی لاوەکی
D_ACCENT = (130, 220, 140) # ڕەنگی سەوزی جیاکەرەوە
FONT = cv2.FONT_HERSHEY_SIMPLEX # جۆری فۆنت


def _text(img, s, org, scale, color, thick=1):
    """فرمانێکی ئاسانکراو بۆ نوسینی تێکست لەسەر وێنە"""
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _text_w(s, scale, thick=1):
    """پێوانەکردنی پانی نوسینەکە بۆ ئەوەی بزانین چەند شوێن دەگرێت"""
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def _dim(img, f=0.32):
    """تاریککردنی وێنەکە"""
    return (img.astype(np.float32) * f).astype(np.uint8)


def _isolate(img, mask):
    """نیشاندانی تەنها گەڵاکە و تاریککردنی هەموو دەوروبەری"""
    out = _dim(img)
    out[mask > 0] = img[mask > 0]
    return out


def _highlight(img, mask, colour, alpha=0.45):
    """تاریککردنی وێنەکە و ڕەنگکردنی بەشێکی دیاریکراو بە ڕەنگێکی تر"""
    out = _dim(img, 0.30)
    tint = cv2.addWeighted(img, 1 - alpha, np.full_like(img, colour), alpha, 0)
    out[mask > 0] = tint[mask > 0]
    return out


def _tile(img, index, title, tw, th, cap_h=36):
    """دروستکردنی کارتێکی بچوک کە وێنەیەک و ناونیشانێکی تێدایە"""
    cell = np.full((th + cap_h, tw, 3), D_CARD, np.uint8)
    cell[:th] = cv2.resize(img, (tw, th))
    _text(cell, str(index), (12, th + 24), 0.62, D_ACCENT, 2)
    _text(cell, title, (34, th + 24), 0.55, D_INK, 1)
    cv2.rectangle(cell, (0, 0), (tw - 1, th + cap_h - 1), D_LINE, 1)
    return cell


def _gauge(img, x, y, w, h, percent, colour):
    """کێشانی شریتێکی پێوانەیی (Bar) بۆ نیشاندانی ڕێژەی سەدی نەخۆشییەکە"""
    cv2.rectangle(img, (x, y), (x + w, y + h), D_CARD, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), D_LINE, 1)
    fill = int(w * min(percent, 100.0) / 100.0) # بڕی پڕکراوە
    cv2.rectangle(img, (x, y), (x + fill, y + h), colour, -1) # پڕکردنەوە بە ڕەنگی سەختی
    for mark in (5, 20, 40): # کێشانی نیشانەکانی سەر شریتەکە
        mx = x + int(w * mark / 100.0)
        cv2.line(img, (mx, y), (mx, y + h), (110, 116, 124), 1)


def _chip(img, x, y, label, value, value_colour=D_INK):
    """دروستکردنی چوارگۆشەیەکی بچوک بۆ نیشاندانی زانیارییەک (وەک ناونیشان و نرخ)"""
    pad = 10
    w = max(_text_w(label, 0.42), _text_w(value, 0.6, 2)) + pad * 2
    cv2.rectangle(img, (x, y), (x + w, y + 44), D_CARD, -1)
    cv2.rectangle(img, (x, y), (x + w, y + 44), D_LINE, 1)
    _text(img, label.upper(), (x + pad, y + 16), 0.40, D_SUB, 1)
    _text(img, value, (x + pad, y + 37), 0.60, value_colour, 2)
    return w


def make_panel(result: LeafResult):
    """دروستکردنی وێنەیەکی ٣ پارچەیی: سەرەکی | نەخۆش | ئەنجامی کۆتایی"""
    img, mk = result.image, result.masks
    h0, w0 = img.shape[:2]
    tw = 360
    th = int(tw * h0 / w0)
    tiles = [
        _tile(img, 1, "Original", tw, th),
        _tile(_highlight(img, mk["diseased"], (60, 60, 235)), 2, "Diseased", tw, th),
        _tile(result.stages["Diagnosis"], 3, "Diagnosis", tw, th),
    ]
    gut = 14
    strip = np.full((th + 36, tw * 3 + gut * 2, 3), D_BG, np.uint8)
    for i, t in enumerate(tiles):
        x = i * (tw + gut)
        strip[0:t.shape[0], x:x + tw] = t
    return strip


def make_dashboard(result: LeafResult):
    """دروستکردنی تابلۆیەکی گەورە کە هەموو قۆناغەکانی تێدایە (٦ کارت + زانیارییەکان)"""
    img, mk, m = result.image, result.masks, result.metrics
    h0, w0 = img.shape[:2]
    tw = 392
    th = int(tw * h0 / w0)
    cap, gut, margin = 36, 18, 26
    head, foot = 104, 92

    sev_colour = LeafDiseaseAnalyzer._severity(m.diseased_percent)[1]

    # کارتەکانی شەش قۆناغەکە
    tiles = [
        _tile(img, 1, "Original", tw, th, cap),
        _tile(result.stages["Illumination Normalized"], 2, "Enhanced (CLAHE)", tw, th, cap),
        _tile(_isolate(img, mk["leaf"]), 3, "Leaf Isolated", tw, th, cap),
        _tile(_highlight(img, mk["healthy"], (90, 200, 90)), 4, "Healthy Tissue", tw, th, cap),
        _tile(_highlight(img, mk["diseased"], (60, 60, 235)), 5, "Diseased Regions", tw, th, cap),
        _tile(result.stages["Diagnosis"], 6, "Diagnosis", tw, th, cap),
    ]
    ch = th + cap
    width = margin * 2 + tw * 3 + gut * 2
    height = head + ch * 2 + gut + foot
    board = np.full((height, width, 3), D_BG, np.uint8)

    # --- header: بەشی سەرەوەی تابلۆکە ---
    _text(board, "LEAF DISEASE ANALYSIS", (margin, 42), 0.95, D_INK, 2)
    _text(board, m.name, (margin, 70), 0.52, D_SUB, 1)
    verdict = f"{m.severity.upper()}  -  {m.diseased_percent:.1f}% AFFECTED"
    vw = _text_w(verdict, 0.78, 2)
    _text(board, verdict, (width - margin - vw, 44), 0.78, sev_colour, 2)
    if m.disease:
        dx = f"likely: {m.disease} ({m.disease_confidence*100:.0f}%)"
        _text(board, dx, (width - margin - _text_w(dx, 0.5), 70), 0.5, D_SUB, 1)
    _gauge(board, margin, 84, width - margin * 2, 10, m.diseased_percent, sev_colour)

    # --- tile grid: ڕیزکردنی کارتەکان ---
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, 3)
        x = margin + c * (tw + gut)
        y = head + r * (ch + gut)
        board[y:y + ch, x:x + tw] = tile

    # --- footer: بەشی خوارەوەی تابلۆکە کە زانیارییە وردەکانی تێدایە ---
    fy = head + ch * 2 + gut + 12
    chips = [
        ("Diagnosis", f"{m.disease} {m.disease_confidence*100:.0f}%" if m.disease else "n/a", D_ACCENT),
        ("Severity", f"{m.severity} ({m.severity_grade}/9)", sev_colour),
        ("Diseased", f"{m.diseased_percent:.1f}%", D_INK),
        ("Healthy", f"{m.healthy_percent:.1f}%", D_INK),
        ("Yellowing", f"{m.chlorosis_percent:.1f}%", D_INK),
        ("Necrosis", f"{m.necrosis_percent:.1f}%", D_INK),
        ("Lesions", str(m.lesion_count), D_INK),
        ("Largest", f"{m.largest_lesion_percent:.1f}%", D_INK),
    ]
    x = margin
    for label, value, colour in chips:
        x += _chip(board, x, fy, label, value, colour) + 10
    return board


def print_metrics(m: LeafMetrics):
    """نوسینی ئەنجامەکان لە تێرمیناڵدا بە ڕێکوپێکی"""
    dx = (f"{m.disease} ({m.disease_confidence*100:.0f}%)  | "
          if m.disease else "")
    logger.info(f"  {m.name:>22}  ->  {dx}{m.diseased_percent:5.1f}% diseased | "
                f"{m.severity:<8} | {m.lesion_count:>3} lesions | "
                f"yellow {m.chlorosis_percent:4.1f}% / necrosis {m.necrosis_percent:4.1f}%")


# --------------------------------------------------------------------------- #
# Batch / single processing - شیکردنەوەی گروپ یان تەنیا یەک وێنە
# --------------------------------------------------------------------------- #

def analyze_image_file(analyzer, path, out_dir, classifier=None, save=True):
    """شیکردنەوەی یەک فایلی وێنە و خەزنکردنی ئەنجامەکانی ئەگەر پێویست بکات"""
    image = cv2.imread(path) # خوێندنەوەی وێنەکە لە فۆڵدەرەکەوە
    if image is None:
        return None
    name = os.path.splitext(os.path.basename(path))[0] # وەرگرتنی ناوی فایلەکە بێ پاشگر
    result = analyzer.analyze(image, name=os.path.basename(path),
                              classifier=classifier) # دەستپێکردنی شیکردنەوە
    if save:
        os.makedirs(out_dir, exist_ok=True) # دروستکردنی بوخچەی دەرەنجام
        cv2.imwrite(os.path.join(out_dir, f"{name}_result.jpg"), result.stages["Result"]) # خەزنکردنی وێنەی ئەنجام
        cv2.imwrite(os.path.join(out_dir, f"{name}_panel.jpg"), make_panel(result)) # خەزنکردنی تابلۆی ٣ بەشی
        cv2.imwrite(os.path.join(out_dir, f"{name}_dashboard.jpg"), make_dashboard(result)) # خەزنکردنی تابلۆی سەرەکی
    return result


def process_file(analyzer, path, out_dir, classifier=None):
    """فرمانێکی ئاسان بۆ یەک فایلی دیاریکراو کە ئەنجامەکەش چاپ دەکات"""
    result = analyze_image_file(analyzer, path, out_dir, classifier)
    if result is None:
        logger.warning(f"  [skip] cannot read {path}")
        return None
    print_metrics(result.metrics)
    return result.metrics


def export_metrics(metrics_list, out_dir):
    """ناردنی هەموو داتاکان بۆ ناو فایلی ئێکسڵ (CSV) و جەیسۆن"""
    if not metrics_list:
        return
    rows = [asdict(m) for m in metrics_list]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2) # خەزنکردن وەک جەیسۆن
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows) # خەزنکردن وەک ئێکسڵ
    logger.info(f"  metrics written: metrics.csv, metrics.json")


# --------------------------------------------------------------------------- #
# Real-time webcam mode - شیکردنەوەی ڕاستەوخۆ بە کامێرا
# --------------------------------------------------------------------------- #

def run_webcam(analyzer, out_dir, cam_index=0, classifier=None):
    """دەستپێکردنی کامێرا و شیکردنەوەی هەموو چرکەیەک بە شێوەی ڕاستەوخۆ"""
    cap = cv2.VideoCapture(cam_index) # کردنەوەی کامێرا
    if not cap.isOpened():
        print("Could not open the camera. Try a different --camera index.")
        return
    print("Webcam mode:  [s] save snapshot   [q] quit")
    saved = 0
    prev = time.time()
    while True:
        ok, frame = cap.read() # وەرگرتنی وێنەیەک لە کامێراوە
        if not ok:
            break
        result = analyzer.analyze(frame, name="webcam", classifier=classifier) # شیکردنەوەی وێنەکە
        view = result.stages["Result"]

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6) # حسابی خێرایی (چرکە/وێنە)
        prev = now
        cv2.putText(view, f"{fps:4.1f} FPS", (view.shape[1] - 90, view.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA) # نوسینی خێرایی

        cv2.imshow("Leaf Disease Analyzer - webcam", view) # نیشاندانی وێنەکە
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): # ئەگەر q داگیرا، دایدەخەین
            break
        if key == ord("s"): # ئەگەر s داگیرا، وێنەیەک خەزن دەکەین
            saved += 1
            p = os.path.join(out_dir, f"webcam_snapshot_{saved:02d}.jpg")
            cv2.imwrite(p, make_dashboard(result))
            print(f"  saved {p}")
    cap.release() # داخستنی کامێرا
    cv2.destroyAllWindows() # داخستنی پەنجەرەکان


# --------------------------------------------------------------------------- #
# Interactive tuner - شیکردنەوەی دەستی بۆ ڕێکخستنی ڕەنگەکان
# --------------------------------------------------------------------------- #

def run_tuner(path):
    """دروستکردنی پەنجەرەیەک بە شریتی دەستی بۆ ئەوەی ڕەنگەکان بە وردی ڕێکبخەیت"""
    image = cv2.imread(path)
    if image is None:
        print(f"Could not read {path}")
        return
    win = "HSV Tuner  (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    names = ["H low", "S low", "V low", "H high", "S high", "V high"]
    defaults = [30, 35, 30, 90, 255, 255] # بەها بنەڕەتییەکان
    maxes = [179, 255, 255, 179, 255, 255] # بەرزترین بەهاکان
    for n, d, mx in zip(names, defaults, maxes):
        cv2.createTrackbar(n, win, d, mx, lambda v: None) # دروستکردنی شریتەکان

    while True:
        lower = [cv2.getTrackbarPos(names[i], win) for i in range(3)] # وەرگرتنی بەها نزمەکان
        upper = [cv2.getTrackbarPos(names[i], win) for i in range(3, 6)] # وەرگرتنی بەها بەرزەکان
        analyzer = LeafDiseaseAnalyzer(healthy_lower=lower, healthy_upper=upper)
        result = analyzer.analyze(image, name="tuner")
        cv2.imshow(win, make_panel(result)) # نیشاندانی گۆڕانکارییەکان بە ڕاستەوخۆ
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# Entry point - دەسپێکی پڕۆگرامەکە
# --------------------------------------------------------------------------- #

def main():
    """فرمانە سەرەکییەکەی کە لێرەوە هەموو شتێک دەست پێ دەکات"""
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.normpath(os.path.join(here, "..", "input")) # بوخچەی سەرەکی وێنەکان
    default_out = os.path.normpath(os.path.join(here, "..", "output")) # بوخچەی سەرەکی ئەنجامەکان

    p = argparse.ArgumentParser(description="Leaf Disease Severity Analyzer")
    p.add_argument("--image", help="analyze a single image") # بۆ یەک وێنە
    p.add_argument("--input", default=default_in, help="input folder") # بوخچەی وێنەکان
    p.add_argument("--output", default=default_out, help="output folder") # بوخچەی ئەنجامەکان
    p.add_argument("--webcam", action="store_true", help="real-time camera mode") # کارکردن بە کامێرا
    p.add_argument("--camera", type=int, default=0, help="camera index") # ژمارەی کامێرا
    p.add_argument("--tune", action="store_true", help="interactive HSV tuner") # بۆ ڕێکخستنی ڕەنگ
    p.add_argument("--min-lesion", type=int, default=MIN_LESION_AREA,
                   help="ignore diseased blobs smaller than this (px)") # کەمترین قەبارەی برین
    p.add_argument("--no-classify", action="store_true",
                   help="skip disease identification (severity only)") # تەنها شیکردنەوە بێ ناونان
    p.add_argument("--fast", action="store_true",
                   help="disable GrabCut (faster; best for plain backgrounds)") # شێوازی خێرا
    args = p.parse_args()

    use_grabcut = not (args.fast or args.webcam) # ئەگەر خێرا بێت، GrabCut بەکارناهێنین
    analyzer = LeafDiseaseAnalyzer(min_lesion_area=args.min_lesion,
                                   use_grabcut=use_grabcut)

    if args.tune:
        run_tuner(args.image or os.path.join(args.input, "sample_leaf.jpg")) # دەستپێکردنی ڕێکخەر
        return

    classifier = None
    if not args.no_classify:
        from leaf_classifier import get_classifier
        classifier = get_classifier() # ئامادەکردنی جیاکەرەوەی نەخۆشی

    os.makedirs(args.output, exist_ok=True)

    if args.webcam:
        run_webcam(analyzer, args.output, args.camera, classifier=classifier) # کامێرا
        return

    if args.image:
        files = [args.image] # یەک وێنە
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files = sorted(f for e in exts
                       for f in glob.glob(os.path.join(args.input, e))) # هەموو وێنەکان
    if not files:
        print(f"No images in {args.input}. Add leaf photos and re-run.")
        return

    print(f"Analyzing {len(files)} image(s):")
    # شیکردنەوەی هەموو فایلەکان بەدوای یەکدا
    metrics = [m for m in (process_file(analyzer, f, args.output, classifier)
                           for f in files) if m is not None]
    export_metrics(metrics, args.output) # خەزنکردنی هەموو داتاکان
    print(f"\nDone. Results saved to: {args.output}")


if __name__ == "__main__":
    main() # بانگکردنی فرمانی سەرەکی بۆ دەستپێکردن
