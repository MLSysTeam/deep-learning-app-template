import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QTextEdit, QTableWidget, 
    QTableWidgetItem, QHeaderView, QGraphicsView, QGraphicsScene,
    QSplitter, QGroupBox, QMessageBox, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QPixmap, QImage
import os


class ImageView(QGraphicsView):
    """自定义图像视图，支持缩放和平移"""
    def __init__(self):
        super().__init__()
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        
    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = self.sceneRect()
        if not rect.isNull():
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1)).width()
            self.scale(1 / unity, 1 / unity)
            view_rect = self.viewport().rect()
            scene_rect = self.transform().mapRect(rect)
            factor = min(view_rect.width() / scene_rect.width(),
                         view_rect.height() / scene_rect.height())
            self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()


class PredictionWorker(QThread):
    """处理预测的后台线程"""
    finished = Signal(str, float)  # 预测完成信号
    error = Signal(str)  # 错误信号

    def __init__(self, model_handler, image_path):
        super().__init__()
        self.model_handler = model_handler
        self.image_path = image_path

    def run(self):
        try:
            predicted_class, confidence = self.model_handler.predict(self.image_path)
            self.finished.emit(predicted_class, confidence)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, model_handler, db_handler):
        super().__init__()
        self.model_handler = model_handler
        self.db_handler = db_handler
        self.current_image_path = None
        
        self.setWindowTitle("Deep Learning Image Classification")
        self.setGeometry(100, 100, 1200, 800)
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧区域 - 上传和图像显示
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 上传组
        upload_group = QGroupBox("Upload Image")
        upload_layout = QVBoxLayout(upload_group)
        
        self.upload_btn = QPushButton("Select Image")
        upload_layout.addWidget(self.upload_btn)
        
        self.classify_btn = QPushButton("Classify Image")
        self.classify_btn.setEnabled(False)
        upload_layout.addWidget(self.classify_btn)
        
        # 图像显示区域
        self.image_view = ImageView()
        upload_layout.addWidget(self.image_view)
        
        left_layout.addWidget(upload_group)
        
        # 右侧区域 - 结果显示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 预测结果组
        result_group = QGroupBox("Classification Result")
        result_layout = QVBoxLayout(result_group)
        
        self.result_label = QLabel("No prediction yet")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        result_layout.addWidget(self.result_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setStyleSheet("font-size: 14px;")
        result_layout.addWidget(self.confidence_label)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-size: 12px;")
        result_layout.addWidget(self.status_label)
        
        right_layout.addWidget(result_group)
        
        # 历史记录组
        history_group = QGroupBox("Recent Classifications")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["ID", "Class", "Confidence", "Date"])
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)
        
        refresh_btn = QPushButton("Refresh History")
        history_layout.addWidget(refresh_btn)
        refresh_btn.clicked.connect(self.load_history)
        
        right_layout.addWidget(history_group)
        
        # 添加左右两个部分到分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])  # 初始大小分配
        
        # 加载历史记录
        self.load_history()
        
    def setup_connections(self):
        self.upload_btn.clicked.connect(self.select_image)
        self.classify_btn.clicked.connect(self.classify_image)
        self.history_table.cellClicked.connect(self.on_history_item_clicked)
        
        # 初始化时创建worker，但不启动
        self.worker = None
        
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.show_image(file_path)
            self.classify_btn.setEnabled(True)
            self.status_label.setText(f"Selected: {os.path.basename(file_path)}")
    
    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_view.setPhoto(pixmap)
        else:
            QMessageBox.critical(self, "Error", "Could not load image file")
    
    def classify_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
            
        self.status_label.setText("Processing image...")
        self.classify_btn.setEnabled(False)
        
        # 创建新的工作线程
        self.worker = PredictionWorker(self.model_handler, self.current_image_path)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.on_prediction_error)
        self.worker.start()
    
    def on_prediction_finished(self, predicted_class, confidence):
        # 更新结果显示
        self.result_label.setText(f"Prediction: {predicted_class}")
        
        # 确保confidence是数值类型再格式化
        try:
            conf_value = float(confidence)
            self.confidence_label.setText(f"Confidence: {conf_value:.4f}")
        except (ValueError, TypeError):
            # 如果无法转换为float，则显示原始值
            self.confidence_label.setText(f"Confidence: {confidence}")
        
        self.status_label.setText("Prediction completed")
        
        # 保存到数据库
        try:
            self.db_handler.save_classification(
                image_path=self.current_image_path,
                predicted_class=predicted_class,
                confidence=confidence
            )
            self.load_history()  # 刷新历史记录
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not save to database: {str(e)}")
        
        self.classify_btn.setEnabled(True)
    
    def on_prediction_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Prediction failed: {error_msg}")
        self.status_label.setText("Prediction failed")
        self.classify_btn.setEnabled(True)
    
    def load_history(self):
        try:
            records = self.db_handler.get_recent_classifications(limit=10)
            
            self.history_table.setRowCount(len(records))
            
            for row, record in enumerate(records):
                self.history_table.setItem(row, 0, QTableWidgetItem(str(record.id)))
                self.history_table.setItem(row, 1, QTableWidgetItem(record.predicted_class))
                
                # 确保confidence转换为float后再格式化
                try:
                    # 检查是否已经是float类型
                    if isinstance(record.confidence, float):
                        conf_str = f"{record.confidence:.4f}"
                    else:
                        # 尝试转换为float
                        conf_value = float(record.confidence)
                        conf_str = f"{conf_value:.4f}"
                except (ValueError, TypeError):
                    # 如果无法转换为float，则使用原始值
                    conf_str = str(record.confidence)
                
                self.history_table.setItem(row, 2, QTableWidgetItem(conf_str))
                self.history_table.setItem(row, 3, QTableWidgetItem(str(record.created_at)))
        except Exception as e:
            # 更详细的错误报告，有助于调试
            print(f"Error loading history: {str(e)}")
            QMessageBox.warning(self, "Warning", f"Could not load history: {str(e)}")
    
    def on_history_item_clicked(self, row, column):
        # 当点击历史记录时，在图像视图中显示对应的图片
        item_id = self.history_table.item(row, 0)
        if item_id:
            try:
                records = self.db_handler.get_recent_classifications(limit=10)
                if row < len(records):
                    record = records[row]
                    image_path = record.image_path
                    if os.path.exists(image_path):
                        self.show_image(image_path)
                        self.current_image_path = image_path
                        self.status_label.setText(f"Showing history item: {os.path.basename(image_path)}")
                    else:
                        QMessageBox.warning(self, "Warning", f"Image file not found: {image_path}")
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not load image from history: {str(e)}")