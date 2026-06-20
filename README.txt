Leaf Disease Severity Analyzer — Computer Vision Final Project
=============================================================

A classic-CV + machine-learning system that, from a single leaf photo:
  - SEGMENTS the leaf and measures DISEASE SEVERITY as a real percentage of
    leaf area affected (the standard plant-pathology metric) + a 0-9 grade.
  - IDENTIFIES the likely DISEASE with a confidence score, using a Random
    Forest trained on labelled images (rule-based fallback if no model yet).
  - Splits damage into yellowing (chlorosis) vs dead tissue (necrosis),
    counts and measures individual lesions, and exports metrics to CSV/JSON.

Folder layout
-------------
  code/
    app.py             web GUI dashboard (Flask) - the front end
    templates/         index.html for the web GUI
    project.py         main pipeline + CLI (segmentation, severity, diagnosis)
    leaf_features.py   interpretable feature extractor (shared by train+predict)
    leaf_classifier.py loads the trained model, or rule-based fallback
    train.py           (optional) trains the classifier on input/dataset/
    model/             saved classifier (leaf_clf.joblib) - created by train.py
  input/   YOUR leaf photos to analyze (+ optional dataset/<Disease>/ to train)
  output/  annotated results, dashboards, metrics.csv / metrics.json
  requirements.txt
  run_all.sh   one command: venv + install + (train) + launch the web GUI

RUN EVERYTHING (one command)
----------------------------
From the project root (the folder that contains this file):

  bash run_all.sh

That creates a virtual environment, installs dependencies, (optionally trains
the classifier if input/dataset/ exists), and launches the WEB GUI at
http://127.0.0.1:5000/ in your browser. In the GUI you can upload a leaf photo
or click one from input/, toggle Fast mode, and view the dashboard, the
annotated leaf, the panel, and all the metrics.

MANUAL / STEP BY STEP
---------------------
  # 1. Create + ACTIVATE a virtual environment (once)
  python3 -m venv .venv
  source .venv/bin/activate            # Linux / macOS
  #   .venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)

  # 2. Install dependencies
  pip install -r requirements.txt

  # 3. (OPTIONAL) train disease ID on a labelled dataset you provide
  #    input/dataset/<Disease>/*.jpg  (e.g. PlantVillage). Without this,
  #    disease names come from the rule-based fallback.
  cd code
  python train.py                 # trains + prints accuracy, saves model/

  # 4a. Web GUI (recommended)
  python app.py                           # opens http://127.0.0.1:5000/
  python app.py --port 8000 --no-browser  # custom port / don't auto-open

  # 4b. Or the command line
  python project.py                       # batch ../input  (+ metrics.csv/json)
  python project.py --image ../input/leaf.jpg
  python project.py --webcam              # real-time camera diagnosis
  python project.py --tune                # interactive HSV tuner (viva demo)
  python project.py --no-classify         # severity only, skip disease ID
  python project.py --fast                # skip GrabCut (plain backgrounds)

  # (when finished:  deactivate )

NOTE ON BACKGROUNDS
-------------------
By default the leaf is isolated with GrabCut, so photos with a CLUTTERED
background (e.g. other leaves behind) still work. This is a little slower
(~a few seconds per image). For leaves on a PLAIN background, --fast skips
GrabCut and runs almost instantly. The webcam always uses the fast path.

WHAT IS REAL
------------
  REAL & trustworthy: the leaf segmentation, the SEVERITY measurement (measured
  % of leaf area affected), the chlorosis/necrosis split, and the lesion
  analysis. These work on the real photos in input/.

  DISEASE NAME: currently from the rule-based fallback (a heuristic guess), so
  treat it as indicative only. For real disease identification, add a labelled
  dataset at input/dataset/<Disease>/ (e.g. PlantVillage) and run train.py -
  the SAME feature code then powers a trained classifier. No code changes.

Viva story: GrabCut/Otsu leaf segmentation -> HSV + CLAHE -> healthy green
  range -> diseased = leaf minus healthy -> % area = severity (+ 0-9 grade);
  engineered features (colour / texture / lesion signatures) -> Random Forest
  -> disease (when a labelled dataset is provided).
