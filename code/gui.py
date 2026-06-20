"""
gui.py - Professional PyQt5 GUI for the Leaf Disease Severity Analyzer.
ڕووکاری گرافیکی پرۆفیشناڵ بۆ بەرنامەی شیکەرەوەی نەخۆشی گەڵا
"""

import os # بۆ کارکردن لەگەڵ فایلەکان
import sys # بۆ کارکردن لەگەڵ سیستەم
import threading # بۆ جێبەجێکردنی کارەکان لە پاشبنەما (بۆ ئەوەی بەرنامەکە نەوەستێت)
import logging # بۆ تۆمارکردنی لۆگ
from PyQt5 import QtWidgets, QtCore, QtGui # کتێبخانەی سەرەکی بۆ دروستکردنی ڕووکار
import cv2 # بۆ کارکردن لەگەڵ وێنەکان
from PIL import Image # بۆ گۆڕینی جۆری وێنەکان

# ڕێکخستنی تۆمارکردنی زانیارییەکان
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# هێنانە ناوەوەی لۆژیکی پڕۆژەکە
from project import LeafDiseaseAnalyzer, analyze_image_file, LeafMetrics, make_dashboard, make_panel
from leaf_classifier import get_classifier

# دیاریکردنی ناونیشانی بوخچەکان
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
INPUT_DIR = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")

# ئامادەکردنی جیاکەرەوەی نەخۆشییەکان
CLASSIFIER = get_classifier()

class LeafApp(QtWidgets.QMainWindow):
    """کلاسی سەرەکی بۆ پەنجەرەی بەرنامەکە"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leaf Disease Analyzer Pro - شیکەرەوەی نەخۆشی گەڵا") # ناونیشانی پەنجەرە
        self.resize(1200, 800) # قەبارەی پەنجەرە
        
        # بارودۆخی ناوەکی
        self.selected_file = None # فایلی هەڵبژێردراو
        self.result_images = {} # وێنە دەرەنجامەکان
        self.cap = None # ئۆبجێکتی کامێرا
        self.webcam_timer = QtCore.QTimer() # تایمەر بۆ نوێکردنەوەی وێنەی کامێرا
        self.webcam_timer.timeout.connect(self.update_webcam_frame)
        
        self.setup_ui() # دروستکردنی پێکهاتەکانی ڕووکار
        self.apply_styles() # جێبەجێکردنی ستایلی ڕەنگەکان

    def setup_ui(self):
        """دروستکردنی هەموو دوگمە و بەشەکانی ڕووکارەکە"""
        main_widget = QtWidgets.QWidget() # وێجێتی سەرەکی
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget) # ڕیزکردنی ئاسۆیی
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Sidebar - بەشی تەنیشت ---
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(300) # پانی بەشی تەنیشت
        sidebar.setObjectName("sidebar")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar) # ڕیزکردنی ستوونی
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)

        # ناونیشانی ناو بەرنامەکە
        title = QtWidgets.QLabel("🌿 Leaf Analyzer")
        title.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
        sidebar_layout.addWidget(title)

        backend_info = QtWidgets.QLabel(f"Backend: {CLASSIFIER.backend}") # جۆری مۆدێلەکە
        backend_info.setStyleSheet("color: #888; font-size: 10px;")
        sidebar_layout.addWidget(backend_info)

        sidebar_layout.addSpacing(10)

        # بەشی کردارەکان
        self.btn_load = QtWidgets.QPushButton("Select Leaf Photo") # دوگمەی هەڵبژاردنی وێنە
        self.btn_load.clicked.connect(self.on_select_image) # بەستنەوە بە فەنکشنەکە
        sidebar_layout.addWidget(self.btn_load)

        self.btn_webcam = QtWidgets.QPushButton("Start Live Webcam") # دوگمەی کامێرا
        self.btn_webcam.setObjectName("btn_webcam")
        self.btn_webcam.clicked.connect(self.on_toggle_webcam)
        sidebar_layout.addWidget(self.btn_webcam)

        self.lbl_filename = QtWidgets.QLabel("No image selected") # ناوی وێنە هەڵبژێردراوەکە
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet("color: #aaa; font-style: italic;")
        sidebar_layout.addWidget(self.lbl_filename)

        self.chk_fast = QtWidgets.QCheckBox("Fast Mode (No GrabCut)") # هەڵبژاردنی شێوازی خێرا
        sidebar_layout.addWidget(self.chk_fast)

        self.btn_analyze = QtWidgets.QPushButton("Run Analysis") # دوگمەی دەستپێکردنی شیکردنەوە
        self.btn_analyze.setEnabled(False) # سەرەتا ناچالاکە
        self.btn_analyze.setFixedHeight(45)
        self.btn_analyze.setObjectName("btn_analyze")
        self.btn_analyze.clicked.connect(self.on_start_analysis)
        sidebar_layout.addWidget(self.btn_analyze)

        self.lbl_status = QtWidgets.QLabel("Ready.") # نیشاندەری دۆخی بەرنامە
        self.lbl_status.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 11px;")
        sidebar_layout.addWidget(self.lbl_status)

        # نیشاندەری پێشکەوتن (Loading)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        sidebar_layout.addWidget(self.progress)

        sidebar_layout.addSpacing(20)

        # بەشی ئەنجامە ژمارەییەکان
        self.metrics_group = QtWidgets.QGroupBox("Diagnostic Metrics") # گروپی ئەنجامەکان
        self.metrics_vbox = QtWidgets.QVBoxLayout(self.metrics_group)
        self.metrics_vbox.setContentsMargins(10, 15, 10, 10)
        self.metrics_vbox.setSpacing(8)
        sidebar_layout.addWidget(self.metrics_group)

        sidebar_layout.addStretch() # پڕکردنەوەی بۆشاییەکان

        # --- Main View Area - بەشی نیشاندانی سەرەکی ---
        view_area = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(view_area)
        view_layout.setContentsMargins(20, 20, 20, 20)

        # تابلۆکانی سەرەوە بۆ گۆڕینی جۆری نیشاندان
        self.tabs = QtWidgets.QTabBar()
        self.tabs.addTab("Dashboard")
        self.tabs.addTab("Annotated Result")
        self.tabs.addTab("3-Panel Evidence")
        self.tabs.currentChanged.connect(self.on_tab_changed) # کاتی گۆڕینی تابلۆ
        view_layout.addWidget(self.tabs)

        # شوێنی نیشاندانی وێنەکە (بە شێوەی سکرۆڵ بۆ ئەوەی قەبارە تێکنەچێت)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(QtCore.Qt.AlignCenter)
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_area.setStyleSheet("background-color: #000; border: 1px solid #333;")
        
        self.img_display = QtWidgets.QLabel("Select an image to begin analysis")
        self.img_display.setAlignment(QtCore.Qt.AlignCenter)
        self.img_display.setObjectName("img_display")
        
        self.scroll_area.setWidget(self.img_display)
        view_layout.addWidget(self.scroll_area)

        layout.addWidget(sidebar)
        layout.addWidget(view_area)

    def apply_styles(self):
        """جێبەجێکردنی ستایلی CSS بۆ ڕەنگەکان و دیزاینەکە"""
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            #sidebar { background-color: #1e1e1e; border-right: 1px solid #333; }
            QLabel { color: #e0e0e0; }
            QGroupBox { color: #888; border: 1px solid #333; margin-top: 15px; padding-top: 10px; font-weight: bold; }
            QPushButton { 
                background-color: #333; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #444; }
            QPushButton:pressed { background-color: #222; }
            #btn_analyze { background-color: #2e7d32; }
            #btn_analyze:hover { background-color: #388e3c; }
            #btn_analyze:disabled { background-color: #222; color: #555; }
            #btn_webcam { background-color: #1565c0; }
            #btn_webcam:hover { background-color: #1976d2; }
            QProgressBar { background-color: #111; border: 1px solid #333; height: 10px; border-radius: 5px; }
            QProgressBar::chunk { background-color: #4caf50; border-radius: 5px; }
            QCheckBox { color: #aaa; }
            QTabBar::tab {
                background: #222; color: #888; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #333; color: #4caf50; border-bottom: 2px solid #4caf50; }
            #img_display { background-color: #000; color: #555; font-size: 16px; }
        """)

    def on_toggle_webcam(self):
        """کردنەوە یان داخستنی کامێرا بە شێوەیەکی ڕاستەوخۆ"""
        if self.cap is None:
            # کردنەوەی کامێرا
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, "Error", "Could not open webcam.")
                self.cap = None
                return
            
            self.btn_webcam.setText("Stop Webcam")
            self.btn_webcam.setStyleSheet("background-color: #c62828;") # گۆڕینی ڕەنگ بۆ سور
            self.btn_load.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.webcam_timer.start(30) # دەستپێکردنی نوێکردنەوە (٣٠ میڵی چرکە)
            self.lbl_status.setText("Live Webcam Active.")
        else:
            # داخستنی کامێرا
            self.webcam_timer.stop()
            self.cap.release()
            self.cap = None
            self.btn_webcam.setText("Start Live Webcam")
            self.btn_webcam.setStyleSheet("") # گەڕانەوە بۆ ڕەنگی شین
            self.btn_load.setEnabled(True)
            if self.selected_file:
                self.btn_analyze.setEnabled(True)
                self.show_image(self.selected_file)
            self.lbl_status.setText("Webcam stopped.")

    def update_webcam_frame(self):
        """وەرگرتنی وێنە لە کامێرا و نیشاندانی بە شێوەی ڕاستەوخۆ"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # شیکردنەوەی خێرا لە کاتی ڕاستەوخۆدا (ئەگەر بەکارهێنەر بییەوێت)
                # لێرەدا دەتوانیت تەنها وێنەکە نیشان بدەیت یان شیکردنەوەکەشی بۆ بکەیت
                fast = self.chk_fast.isChecked()
                analyzer = LeafDiseaseAnalyzer(use_grabcut=False) # هەمیشە GrabCut لادەبەین بۆ خێرایی
                
                # ئەنجامدانی شیکردنەوەیەکی خێرا
                result = analyzer.analyze(frame, name="Live View", classifier=CLASSIFIER)
                
                # دیاریکردنی کام وێنە نیشان بدرێت بەپێی تابلۆ هەڵبژێردراوەکە
                keys = ["dashboard", "result", "panel"]
                view_key = keys[self.tabs.currentIndex()]
                
                if view_key == "dashboard":
                    display_frame = make_dashboard(result)
                elif view_key == "result":
                    display_frame = result.stages["Result"]
                else:
                    display_frame = make_panel(result)
                
                self.show_image(display_frame)
                # نوێکردنەوەی ژمارەکان بە شێوەی ڕاستەوخۆ
                self.display_metrics(result.metrics)

    def on_select_image(self):
        """کردنەوەی پەنجەرەی هەڵبژاردنی وێنە لە ناو کۆمپیوتەر"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Leaf Image", INPUT_DIR, "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self.selected_file = path
            self.lbl_filename.setText(os.path.basename(path)) # نیشاندانی ناوی فایلەکە
            self.btn_analyze.setEnabled(True) # چالاککردنی دوگمەی شیکردنەوە
            self.show_image(path) # نیشاندانی وێنە هەڵبژێردراوەکە
            self.result_images = {} # سڕینەوەی دەرەنجامە کۆنەکان
            self.clear_metrics() # پاککردنەوەی پێوەرە کۆنەکان

    def show_image(self, data):
        """نیشاندانی وێنە (چ وەک ناونیشانی فایل یان وەک وێنەی ئۆپن سی ڤی)"""
        if isinstance(data, str):
            pixmap = QtGui.QPixmap(data) # ئەگەر ناونیشانی فایل بێت
        else:
            # ئەگەر وێنەی ئۆپن سی ڤی بێت، دەبێت بیگۆڕین بۆ جۆرێک کە Qt بتوانێت نیشانی بدات
            rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) # گۆڕین بۆ RGB
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)

        if pixmap.isNull():
            return

        # بچوککردنەوەی وێنەکە بە پێی قەبارەی شاشەکە بێ ئەوەی شێوەکەی تێکبچێت
        view_size = self.scroll_area.viewport().size()
        scaled_pixmap = pixmap.scaled(
            view_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.img_display.setPixmap(scaled_pixmap) # دانانی وێنەکە لە شوێنی خۆی

    def resizeEvent(self, event):
        """ئەم فەنکشنە بانگ دەکرێت کاتێک بەکارهێنەر قەبارەی پەنجەرەکە دەگۆڕێت"""
        if self.tabs.currentIndex() >= 0 and self.result_images:
            self.on_tab_changed(self.tabs.currentIndex()) # دووبارە ڕێکخستنەوەی قەبارەی وێنەی ئەنجام
        elif self.selected_file and not self.result_images:
            self.show_image(self.selected_file) # دووبارە ڕێکخستنەوەی وێنە سەرەکییەکە
        super().resizeEvent(event)

    def on_tab_changed(self, index):
        """گۆڕینی وێنەی نیشاندراو کاتێک تابلۆکان دەگۆڕدرێن"""
        keys = ["dashboard", "result", "panel"]
        key = keys[index]
        if key in self.result_images:
            self.show_image(self.result_images[key]) # نیشاندانی وێنەی تابلۆ هەڵبژێردراوەکە

    def on_start_analysis(self):
        """دەستپێکردنی کرداری شیکردنەوە"""
        self.btn_analyze.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0) # نیشاندانی جوڵەی لۆدینگ
        
        fast = self.chk_fast.isChecked()
        # دەستپێکردنی شیکردنەوە لەناو ثرێدێکی جیاواز بۆ ئەوەی ڕووکارەکە نەوەستێت
        threading.Thread(target=self.do_analysis, args=(self.selected_file, fast), daemon=True).start()

    def do_analysis(self, path, fast):
        """ئەنجامدانی کردارە قورسەکانی بینینی کۆمپیوتەری"""
        try:
            analyzer = LeafDiseaseAnalyzer(use_grabcut=not fast)
            result = analyze_image_file(analyzer, path, OUTPUT_DIR, CLASSIFIER, save=True)
            if result:
                # کۆکردنەوەی وێنە دەرەنجامەکان
                self.result_images = {
                    "dashboard": make_dashboard(result),
                    "result": result.stages["Result"],
                    "panel": make_panel(result)
                }
                # ناردنی ئەنجامەکان بۆ ڕووکاری سەرەکی (چونکە تەنها ثرێدی سەرەکی دەتوانێت ڕووکار بگۆڕێت)
                QtCore.QMetaObject.invokeMethod(self, "on_analysis_done", 
                                               QtCore.Qt.QueuedConnection,
                                               QtCore.Q_ARG(object, result.metrics))
            else:
                QtCore.QMetaObject.invokeMethod(self, "on_analysis_failed", 
                                               QtCore.Qt.QueuedConnection,
                                               QtCore.Q_ARG(str, "Could not process image"))
        except Exception as e:
            logger.exception("Analysis failed")
            QtCore.QMetaObject.invokeMethod(self, "on_analysis_failed", 
                                           QtCore.Qt.QueuedConnection,
                                           QtCore.Q_ARG(str, str(e)))

    @QtCore.pyqtSlot(object)
    def on_analysis_done(self, metrics):
        """ئەمە بانگ دەکرێت کاتێک شیکردنەوە بە سەرکەوتوویی تەواو دەبێت"""
        self.progress.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.on_tab_changed(self.tabs.currentIndex()) # نیشاندانی وێنەی دەرەنجام
        self.display_metrics(metrics) # نیشاندانی ژمارەکان

    @QtCore.pyqtSlot(str)
    def on_analysis_failed(self, msg):
        """ئەمە بانگ دەکرێت ئەگەر هەڵەیەک ڕوو بدات"""
        self.progress.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.btn_load.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Error", f"Analysis failed:\n{msg}") # نیشاندانی هەڵە بە نامە

    def clear_metrics(self):
        """سڕینەوەی هەموو نوسینەکانی ناو بەشی ئەنجامەکان"""
        while self.metrics_vbox.count():
            item = self.metrics_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def display_metrics(self, m: LeafMetrics):
        """نیشاندانی دەرەنجامە ژمارەییەکان بە شێوەیەکی جوان لە تەنیشت"""
        self.clear_metrics()
        
        def add_row(label, value, color=None, bold=False):
            """فرمانێکی ناوەکی بۆ زیادکردنی هێڵێکی زانیاری"""
            w = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet("color: #9aa0a8; font-size: 11px; text-transform: uppercase;")
            val = QtWidgets.QLabel(str(value))
            val_style = "font-size: 13px; color: #eceef0;"
            if bold: val_style += " font-weight: bold;"
            if color: val_style += f" color: {color}; font-weight: bold;"
            val.setStyleSheet(val_style)
            l.addWidget(lbl)
            l.addStretch()
            l.addWidget(val)
            self.metrics_vbox.addWidget(w)

        # ڕەنگەکان بە پێی ئاستی سەختی
        sev_colors = {"Healthy": "#4fb15a", "Mild": "#d8d23c", "Moderate": "#e6973a", "Severe": "#e23b3b"}
        color = sev_colors.get(m.severity, "#eceef0")

        # زیادکردنی زانیارییە سەرەکییەکان
        add_row("OVERALL SEVERITY", m.severity.upper(), color=color)
        add_row("AFFECTED AREA", f"{m.diseased_percent:.1f}%", bold=True)
        
        if m.disease:
            add_row("DIAGNOSIS", m.disease, color="#81c784")
            add_row("CONFIDENCE", f"{m.disease_confidence*100:.0f}%")

        # کێشانی هێڵێکی جیاکەرەوە
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("background-color: #333;")
        self.metrics_vbox.addWidget(line)

        # زیادکردنی زانیارییە وردەکان
        add_row("Healthy Tissue", f"{m.healthy_percent:.1f}%")
        add_row("Yellowing (Chlorosis)", f"{m.chlorosis_percent:.1f}%")
        add_row("Dead Tissue (Necrosis)", f"{m.necrosis_percent:.1f}%")
        add_row("Lesion Count", m.lesion_count)
        add_row("Largest Lesion", f"{m.largest_lesion_percent:.1f}%")
        
        self.metrics_vbox.addStretch()

if __name__ == "__main__":
    # دەستپێکردنی بەرنامەکە
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion") # بەکارهێنانی ستایلی مۆدێرن
    window = LeafApp()
    window.show() # نیشاندانی پەنجەرەکە
    sys.exit(app.exec_()) # مانەوە لە ناو پڕۆسەکە تا داخستن
