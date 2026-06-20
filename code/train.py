"""
train.py — trains the leaf-disease classifier on a labelled image folder.

Expected layout (one sub-folder per disease):
    input/dataset/
        Healthy/         *.jpg
        Rust/            *.jpg
        Early Blight/    *.jpg
        ...

For each image it runs the full CV pipeline (project.LeafDiseaseAnalyzer),
extracts the interpretable feature vector (leaf_features.extract), then trains a
Random Forest and reports held-out accuracy + a confusion matrix. The model is
saved to code/model/leaf_clf.joblib and is picked up automatically by
project.py for real predictions.

Usage:  python train.py                 # uses ../input/dataset
        python train.py --data PATH
"""

from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from project import LeafDiseaseAnalyzer
import leaf_features

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "leaf_clf.joblib")
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def load_dataset(data_dir, analyzer):
    X, y = [], []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    if not classes:
        raise SystemExit(f"No class sub-folders found in {data_dir}")
    print("Extracting features:")
    for klass in classes:
        files = sorted(f for e in EXTS
                       for f in glob.glob(os.path.join(data_dir, klass, e)))
        for path in files:
            img = __import__("cv2").imread(path)
            if img is None:
                continue
            result = analyzer.analyze(img)
            X.append(leaf_features.extract(result))
            y.append(klass)
        print(f"  {klass:<16} {len(files)} images")
    return np.array(X, np.float32), np.array(y), classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.normpath(
        os.path.join(HERE, "..", "input", "dataset")))
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()

    analyzer = LeafDiseaseAnalyzer()
    X, y, classes = load_dataset(args.data, analyzer)
    print(f"\nDataset: {len(X)} samples, {len(classes)} classes, "
          f"{X.shape[1]} features each")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=args.trees, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"\nHeld-out accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_te, clf.predict(X_te), zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print("  " + "  ".join(f"{c[:6]:>6}" for c in clf.classes_))
    for row, c in zip(confusion_matrix(y_te, clf.predict(X_te), labels=clf.classes_),
                      clf.classes_):
        print(f"{c[:14]:<14} " + "  ".join(f"{v:>6}" for v in row))

    # Top features (interpretability for the viva).
    order = np.argsort(clf.feature_importances_)[::-1][:6]
    print("\nMost important features:")
    for i in order:
        print(f"  {leaf_features.FEATURE_NAMES[i]:<20} {clf.feature_importances_[i]:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "labels": list(clf.classes_),
                 "features": leaf_features.FEATURE_NAMES}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
