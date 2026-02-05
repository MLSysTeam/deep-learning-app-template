import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QTextEdit, QTableWidget, 
    QTableWidgetItem, QHeaderView, QGraphicsView, QGraphicsScene,
    QSplitter, QGroupBox, QMessageBox, QGraphicsPixmapItem, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QPixmap, QImage
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


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


class SegmentationWorker(QThread):
    """处理分割的后台线程"""
    finished = Signal(dict)  # 分割完成信号
    error = Signal(str)  # 错误信号

    def __init__(self, model_handler, image_path, text_prompt):
        super().__init__()
        self.model_handler = model_handler
        self.image_path = image_path
        self.text_prompt = text_prompt

    def run(self):
        try:
            results = self.model_handler.segment(self.image_path, self.text_prompt)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, model_handler, db_handler):
        super().__init__()
        self.model_handler = model_handler
        self.db_handler = db_handler
        self.current_image_path = None
        
        self.setWindowTitle("Deep Learning Image Segmentation")
        self.setGeometry(100, 100, 1400, 800)
        
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
        
        # 文本提示输入框
        text_prompt_layout = QHBoxLayout()
        text_prompt_layout.addWidget(QLabel("Text Prompt:"))
        self.text_prompt_input = QLineEdit("object")  # 默认提示
        text_prompt_layout.addWidget(self.text_prompt_input)
        upload_layout.addLayout(text_prompt_layout)
        
        self.segment_btn = QPushButton("Segment Image")
        self.segment_btn.setEnabled(False)
        upload_layout.addWidget(self.segment_btn)
        
        # 图像显示区域
        self.image_view = ImageView()
        upload_layout.addWidget(self.image_view)
        
        left_layout.addWidget(upload_group)
        
        # 右侧区域 - 结果显示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 分割结果组
        result_group = QGroupBox("Segmentation Result")
        result_layout = QVBoxLayout(result_group)
        
        # 分割结果显示
        self.result_image_view = ImageView()
        result_layout.addWidget(self.result_image_view)
        
        # 结果标签
        self.result_label = QLabel("No segmentation yet")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        result_layout.addWidget(self.result_label)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-size: 12px;")
        result_layout.addWidget(self.status_label)
        
        right_layout.addWidget(result_group)
        
        # 历史记录组
        history_group = QGroupBox("Recent Segmentations")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["ID", "Prompt", "Objects", "Confidence", "Date"])
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
        splitter.setSizes([700, 700])  # 初始大小分配
        
        # 加载历史记录
        self.load_history()
        
    def setup_connections(self):
        self.upload_btn.clicked.connect(self.select_image)
        self.segment_btn.clicked.connect(self.segment_image)
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
            self.segment_btn.setEnabled(True)
            self.status_label.setText(f"Selected: {os.path.basename(file_path)}")
    
    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_view.setPhoto(pixmap)
        else:
            QMessageBox.critical(self, "Error", "Could not load image file")
    
    def show_result_image(self, original_path, results):
        """优化后的显示分割结果图片函数"""
        try:
            # 1. 加载原始图像
            image = cv2.imread(original_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # 备份一份用于叠加
            output_image = image.copy()
            
            raw_masks = results.get("raw_masks")
            raw_boxes = results.get("raw_boxes")
            phrases = results.get("phrases", [])

            # 2. 绘制掩码 (Masks)
            if raw_masks is not None:
                # 确保转为 numpy 格式
                masks = raw_masks.cpu().numpy() if hasattr(raw_masks, 'cpu') else np.array(raw_masks)
                
                for mask in masks:
                    if len(mask.shape) > 2:
                        mask = mask.squeeze()
                    
                    # 创建随机颜色 (B, G, R)
                    color = np.random.randint(0, 255, (3,)).tolist()
                    
                    # 创建彩色掩码层
                    mask_overlay = np.zeros_like(image, dtype=np.uint8)
                    mask_overlay[mask > 0.5] = color  # 假设阈值0.5
                    
                    # 混合原图与掩码层 (alpha=0.4)
                    cv2.addWeighted(mask_overlay, 0.4, output_image, 1.0, 0, output_image)

            # 3. 绘制边界框 (Boxes) 和 标签 (Labels)
            if raw_boxes is not None:
                boxes = raw_boxes.cpu().numpy() if hasattr(raw_boxes, 'cpu') else np.array(raw_boxes)
                for i, (box, label) in enumerate(zip(boxes, phrases)):
                    x0, y0, x1, y1 = map(int, box)
                    
                    # 画矩形框
                    cv2.rectangle(output_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    
                    # 画标签背景和文字
                    # label_txt = f"{label}"
                    # cv2.putText(output_image, label_txt, (x0, y0 - 10), 
                                # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 4. 将 BGR 转换为 RGB 并显示到 PySide6
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            h, w, ch = output_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(output_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            self.result_image_view.setPhoto(pixmap)

        except Exception as e:
            print(f"Error rendering result: {e}")
            self.result_image_view.setPhoto(QPixmap(original_path))
    
    # def show_result_image(self, original_path, results):
    #     """显示分割结果图片"""
    #     try:
    #         # 加载原始图像
    #         original_image = cv2.imread(original_path)
    #         if original_image is None:
    #             raise ValueError("Could not load image")
    #         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
    #         # 创建可视化结果
    #         fig, ax = plt.subplots(figsize=(10, 10))
    #         ax.imshow(original_image)
            
    #         # 获取分割结果
    #         raw_masks = results.get("raw_masks")
    #         raw_boxes = results.get("raw_boxes")
    #         phrases = results.get("phrases", [])
            
    #         # 检查是否存在分割数据
    #         if raw_masks is not None and hasattr(raw_masks, 'cpu'):
    #             # 处理Grounded-SAM模型输出的掩码
    #             masks = raw_masks.cpu().numpy() if hasattr(raw_masks, 'cpu') else raw_masks.numpy()
                
    #             # 如果有掩码结果，则绘制
    #             for i, mask in enumerate(masks):
    #                 # 确保mask是二维数组
    #                 if len(mask.shape) > 2:
    #                     mask = mask.squeeze(0)  # 去掉单维度
                    
    #                 # 创建随机颜色
    #                 color = np.random.rand(3,)
    #                 # 创建掩码显示
    #                 colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    #                 for j in range(3):
    #                     colored_mask[:, :, j] = mask * color[j]
                    
    #                 # 设置透明度
    #                 alpha = 0.5
    #                 ax.imshow(colored_mask, alpha=alpha)
            
    #         # 绘制边界框
    #         if raw_boxes is not None:
    #             boxes = raw_boxes.cpu().numpy() if hasattr(raw_boxes, 'cpu') else raw_boxes
    #             for i, (box, label) in enumerate(zip(boxes, phrases)):
    #                 x0, y0, x1, y1 = box
    #                 width = x1 - x0
    #                 height = y1 - y0
                    
    #                 rect = plt.Rectangle((x0, y0), width, height, 
    #                                    linewidth=2, edgecolor='red', facecolor='none')
    #                 ax.add_patch(rect)
                    
    #                 # 添加标签
    #                 ax.text(x0, y0, label, bbox=dict(facecolor='red', alpha=0.5), fontsize=10, color='white')
            
    #         ax.axis('off')
    #         plt.tight_layout()
            
    #         # 保存到临时文件
    #         temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    #         plt.savefig(temp_file.name, bbox_inches='tight', pad_inches=0.0, dpi=100)
    #         plt.close(fig)
            
    #         # 显示结果图片
    #         result_pixmap = QPixmap(temp_file.name)
    #         self.result_image_view.setPhoto(result_pixmap)
            
    #         # 删除临时文件
    #         os.unlink(temp_file.name)
            
    #     except Exception as e:
    #         print(f"Error showing result image: {e}")
    #         # 如果无法生成可视化结果，至少显示原始图片
    #         self.result_image_view.setPhoto(QPixmap(self.current_image_path))
    
    def segment_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
            
        text_prompt = self.text_prompt_input.text().strip()
        if not text_prompt:
            text_prompt = "object"  # 默认提示
            
        self.status_label.setText("Processing image...")
        self.segment_btn.setEnabled(False)
        
        # 创建新的工作线程
        self.worker = SegmentationWorker(self.model_handler, self.current_image_path, text_prompt)
        self.worker.finished.connect(self.on_segmentation_finished)
        self.worker.error.connect(self.on_segmentation_error)
        self.worker.start()
    
    def on_segmentation_finished(self, results):
        # 更新结果显示
        num_objects = results.get("num_objects_detected", 0)
        self.result_label.setText(f"Segmentation completed: {num_objects} objects detected")
        
        # 显示分割结果图片
        self.show_result_image(self.current_image_path, results)
        
        self.status_label.setText("Segmentation completed")
        
        # 保存到数据库
        try:
            text_prompt = self.text_prompt_input.text().strip()
            if not text_prompt:
                text_prompt = "object"
                
            self.db_handler.save_segmentation(
                image_path=self.current_image_path,
                text_prompt=text_prompt,
                num_objects=num_objects,
                confidence_avg=sum(obj.get("confidence", 0) for obj in results.get("objects", [])) / max(len(results.get("objects", [])), 1) if results.get("objects", []) else 0
            )
            self.load_history()  # 刷新历史记录
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not save to database: {str(e)}")
        
        self.segment_btn.setEnabled(True)
    
    def on_segmentation_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Segmentation failed: {error_msg}")
        self.status_label.setText("Segmentation failed")
        self.segment_btn.setEnabled(True)
    
    def load_history(self):
        try:
            records = self.db_handler.get_recent_segmentations(limit=10)
            
            self.history_table.setRowCount(len(records))
            
            for row, record in enumerate(records):
                self.history_table.setItem(row, 0, QTableWidgetItem(str(record.id)))
                self.history_table.setItem(row, 1, QTableWidgetItem(record.text_prompt))
                self.history_table.setItem(row, 2, QTableWidgetItem(str(record.num_objects)))
                self.history_table.setItem(row, 3, QTableWidgetItem(f"{record.confidence_avg:.2f}"))
                self.history_table.setItem(row, 4, QTableWidgetItem(str(record.created_at)))
        except Exception as e:
            # 更详细的错误报告，有助于调试
            print(f"Error loading history: {str(e)}")
            QMessageBox.warning(self, "Warning", f"Could not load history: {str(e)}")
    
    def on_history_item_clicked(self, row, column):
        # 当点击历史记录时，在图像视图中显示对应的图片
        item_id = self.history_table.item(row, 0)
        if item_id:
            try:
                records = self.db_handler.get_recent_segmentations(limit=10)
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