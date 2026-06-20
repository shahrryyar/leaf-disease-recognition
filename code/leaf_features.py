"""
leaf_features.py — turns an analyzed leaf into a fixed numeric feature vector.
دەرهێنانی تایبەتمەندییەکان - وێنەی گەڵا دەگۆڕێت بۆ کۆمەڵێک ژمارە بۆ ژیری دەستکرد
"""

from __future__ import annotations
import numpy as np # بۆ کارکردن لەگەڵ لیستە ژمارەییەکان
import cv2 # بۆ کردارەکانی پڕۆسێسکردنی وێنە

# لیستی ناوی ئەو تایبەتمەندییانەی کە دەردەهێنرێن
FEATURE_NAMES = [
    "diseased_pct", "healthy_pct", "chlorosis_pct", "necrosis_pct",
    "lesion_density", "largest_lesion_pct", "mean_lesion_area", "std_lesion_area",
    "dis_h_mean", "dis_h_std", "dis_s_mean", "dis_s_std", "dis_v_mean", "dis_v_std",
    "leaf_h_mean", "leaf_s_mean", "leaf_v_mean",
    "orange_frac", "bright_lowsat_frac", "dark_frac",
    "texture_lap_var", "grad_mean",
]


def _masked_stats(channel, mask):
    """حسابکردنی تێکڕا و لادانی پێوانەیی بۆ ناوچەیەکی دیاریکراو لە وێنەکەدا"""
    vals = channel[mask > 0]
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std())


def extract(result) -> np.ndarray:
    """وەرگرتنی ڤێکتەرێکی ژمارەیی لە ئەنجامی شیکردنەوەی گەڵایەک"""
    image = result.image # وێنە سەرەکییەکە
    m = result.metrics # پێوەرەکان
    leaf = result.masks["leaf"] # ماسکی گەڵاکە
    diseased = result.masks["diseased"] # ماسکی بەشە نەخۆشەکان
    leaf_area = max(1, int(cv2.countNonZero(leaf))) # ڕووبەری گەڵا بە پێکسڵ

    # گۆڕین بۆ HSV و کەمکردنەوەی ژاوەژاو
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv) # جیاکردنەوەی ڕەنگ و تێری و ڕووناکی

    # ئامارەکانی قەبارەی برینەکان
    contours, _ = cv2.findContours(diseased, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= 10],
                     dtype=np.float32) # ڕووبەری برینەکان
    mean_les = float(areas.mean()) if areas.size else 0.0 # تێکڕای قەبارەی برین
    std_les = float(areas.std()) if areas.size else 0.0 # لادانی قەبارەی برینەکان
    lesion_density = len(areas) / (leaf_area / 1000.0)   # چڕی برینەکان لە هەر ١٠٠٠ پێکسڵدا

    # ئامارەکانی ڕەنگ بۆ بەشە نەخۆشەکان و هەموو گەڵاکە
    dis_h_m, dis_h_s = _masked_stats(H, diseased)
    dis_s_m, dis_s_s = _masked_stats(S, diseased)
    dis_v_m, dis_v_s = _masked_stats(V, diseased)
    leaf_h_m, _ = _masked_stats(H, leaf)
    leaf_s_m, _ = _masked_stats(S, leaf)
    leaf_v_m, _ = _masked_stats(V, leaf)

    # دیاریکردنی ڕەنگە تایبەتەکان بۆ نەخۆشییەکان (وەک پرتەقاڵی بۆ ژەنگ)
    orange = cv2.inRange(hsv, (5, 90, 90), (18, 255, 255))         # ژەنگی گەڵا
    bright_lowsat = cv2.inRange(hsv, (0, 0, 170), (179, 60, 255))  # کەڕووی سپی
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 70))             # مردنی شانەکان
    orange_frac = cv2.countNonZero(cv2.bitwise_and(orange, leaf)) / leaf_area
    bright_frac = cv2.countNonZero(cv2.bitwise_and(bright_lowsat, leaf)) / leaf_area
    dark_frac = cv2.countNonZero(cv2.bitwise_and(dark, leaf)) / leaf_area

    # شیکردنەوەی ململانی (Texture) و لێوارەکان (Edges)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) # ڕادەی ڕوونی وێنەکە
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mean = float(np.mean(np.sqrt(gx * gx + gy * gy))) # چڕی گۆڕانی ڕەنگەکان

    # کۆکردنەوەی هەموو تایبەتمەندییەکان لە ناو یەک ڤێکتەردا
    vec = [
        m.diseased_percent, m.healthy_percent, m.chlorosis_percent, m.necrosis_percent,
        lesion_density, m.largest_lesion_percent, mean_les, std_les,
        dis_h_m, dis_h_s, dis_s_m, dis_s_s, dis_v_m, dis_v_s,
        leaf_h_m, leaf_s_m, leaf_v_m,
        orange_frac, bright_frac, dark_frac,
        lap_var, grad_mean,
    ]
    return np.asarray(vec, dtype=np.float32)
