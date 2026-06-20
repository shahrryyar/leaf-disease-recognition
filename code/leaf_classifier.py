"""
leaf_classifier.py — names the likely disease for an analyzed leaf.
جیاکەرەوەی نەخۆشی گەڵا - ناوی نەخۆشییەکە دیاری دەکات
"""

from __future__ import annotations
import os # بۆ کارکردن لەگەڵ سیستەمی فایل
import logging # بۆ تۆمارکردنی زانیارییەکان
import leaf_features # هێنانە ناوەوەی تایبەتمەندییەکانی گەڵا

logger = logging.getLogger(__name__) # دروستکردنی لۆگەر

# ناونیشانی مۆدێلە فێربووەکە
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "leaf_clf.joblib")


class ModelClassifier:
    """کلاسێک بۆ بەکارهێنانی مۆدێلی ژیری دەستکردی فێربوو (Random Forest)"""
    backend = "trained-model"

    def __init__(self, bundle):
        self.model = bundle["model"] # مۆدێلە سەرەکییەکە
        self.labels = bundle["labels"] # ناوی نەخۆشییەکان

    def predict(self, result):
        """پێشبینیکردنی جۆری نەخۆشی بەپێی وێنەکە"""
        x = leaf_features.extract(result).reshape(1, -1) # دەرهێنانی تایبەتمەندییەکان
        proba = self.model.predict_proba(x)[0] # حسابی ئەگەری نەخۆشییەکان
        idx = int(proba.argmax()) # دۆزینەوەی بەرزترین ئەگەر
        return self.model.classes_[idx], float(proba[idx]) # گەڕاندنەوەی ناو و ڕێژەی دڵنیایی


class RuleBasedClassifier:
    """ئەمە جێگرەوەیە کاتێک مۆدێلە فێربووەکە بوونی نییە (بەکارهێنانی یاسای دەستی)"""
    backend = "rule-based"

    LABELS = ["Healthy", "Rust", "Early Blight", "Powdery Mildew", "Leaf Spot"]

    def predict(self, result):
        """دیاریکردنی نەخۆشی بەپێی یاساکانی ڕەنگ و شێوە"""
        m = result.metrics # پێوەرەکان
        f = {n: v for n, v in zip(leaf_features.FEATURE_NAMES,
                                  leaf_features.extract(result))} # تایبەتمەندییەکان
        scores = {}

        # Healthy: ئەگەر ڕێژەی نەخۆشی زۆر کەم بێت
        scores["Healthy"] = max(0.0, 1.0 - m.diseased_percent / 6.0)

        # Rust: ئەگەر خاڵی پرتەقاڵی زۆر بێت
        scores["Rust"] = (f["orange_frac"] * 6.0
                          + min(f["lesion_density"], 3.0) / 3.0 * 0.6)

        # Powdery Mildew: ئەگەر پەڵەی سپی و بێ ڕەنگ هەبێت
        scores["Powdery Mildew"] = f["bright_lowsat_frac"] * 7.0

        # Early Blight: پەڵەی گەورە و قاوەیی تۆخ
        big = 1.0 if m.largest_lesion_percent > 4 else m.largest_lesion_percent / 4.0
        scores["Early Blight"] = (f["dark_frac"] * 4.0 + big * 0.8
                                  + (m.necrosis_percent / 25.0))

        # Leaf Spot: خاڵی وردی زۆر و ڕەشبوونی شانەکان
        density = min(f["lesion_density"], 4.0) / 4.0
        scores["Leaf Spot"] = (density * 1.2 + m.necrosis_percent / 30.0
                               + (0.4 if m.lesion_count >= 6 else 0.0))

        # دیاریکردنی باشترین ئەنجام
        total = sum(scores.values()) or 1.0
        best = max(scores, key=scores.get)
        return best, scores[best] / total


def get_classifier():
    """ئەگەر فایلە خەزنکراوەکە هەبێت مۆدێلەکە وەردەگرێت، ئەگینا یاسا دەستییەکان بەکاردێنێت"""
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH) # خوێندنەوەی مۆدێلەکە
            logger.info(f"Classifier: trained model ({len(bundle['labels'])} classes) loaded")
            return ModelClassifier(bundle)
        except Exception as e:   # ئەگەر کێشەیەک هەبێت لە مۆدێلەکە
            logger.warning(f"Classifier: model load failed ({e}); using rule-based")
    else:
        logger.info("Classifier: no trained model found; using rule-based fallback (run train.py for real predictions)")
    return RuleBasedClassifier()
