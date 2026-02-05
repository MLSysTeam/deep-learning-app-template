# 深度学习应用程序模板

一个使用PySide6桌面客户端部署PyTorch模型的模板。此项目演示了如何创建一个集成了深度学习模型的图形用户界面桌面应用程序。

## 功能特点

- 基于PySide6的桌面应用程序
- 集成PyTorch模型处理
- SQLite/MySQL数据库存储分类结果
- 线程预测防止UI阻塞
- 支持缩放的响应式图像查看器
- 历史记录跟踪以前的分类

## 架构

应用程序分为四个主要模块：

- [main.py](file:///home/jason/Code/ai4fish_sdu_nyu/deep-learning-app-template/app/main.py)：应用程序入口点
- [ui.py](file:///home/jason/Code/ai4fish_sdu_nyu/deep-learning-app-template/app/ui.py)：PySide6 GUI组件和布局
- [model.py](file:///home/jason/Code/ai4fish_sdu_nyu/deep-learning-app-template/app/model.py)：模型加载和预测逻辑
- [database.py](file:///home/jason/Code/ai4fish_sdu_nyu/deep-learning-app-template/app/database.py)：数据库操作和ORM模型

```
┌─────────────────────────────────────────────────────────────┐
│                    桌面应用程序                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌──────────────────────────────┐ │
│  │   UI模块        │◄────►│         主模块              │ │
│  │   (ui.py)       │      │     (main.py)              │ │
│  │                 │      │                            │ │
│  │ - 图像查看器    │      │ - 初始化所有模块           │ │
│  │ - 控件          │      │ - 处理应用生命周期        │ │
│  │ - 历史记录表    │      │ - 事件循环                │ │
│  └─────────────────┘      └──────────────────────────────┘ │
│             │                           │                  │
│             ▼                           ▼                  │
│  ┌─────────────────┐      ┌──────────────────────────────┐ │
│  │  模型模块       │      │    数据库模块              │ │
│  │   (model.py)    │      │    (database.py)           │ │
│  │                 │      │                            │ │
│  │ - PyTorch模型   │      │ - 存储分类结果             │ │
│  │ - 预测          │      │ - 支持SQLite/MySQL        │ │
│  │ - 预处理        │      │                            │ │
│  └─────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 要求

- Python 3.8+
- PyTorch
- PySide6
- SQLAlchemy
- Pillow

## 安装

1. 克隆仓库：
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. 安装依赖：
   ```bash
   uv sync
   ```

## 使用方法

通过以下任一方式运行应用程序：

1. 使用运行脚本：
   ```bash
   ./run_app.sh
   ```

2. 直接使用Python：
   ```bash
   python run_app.py
   ```

应用程序将启动一个包含以下内容的窗口：
- 左侧的图像上传区域
- 右侧的结果显示区域
- 显示以前分类的历史表格

## 配置

### 数据库设置

应用程序首先尝试连接到MySQL，如果MySQL不可用则回退到SQLite。您可以通过在`.env`文件中设置环境变量来配置MySQL连接：

```
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_PORT=3336
DB_NAME=image_classification
```

## 项目结构

```
app/
├── main.py          # 应用程序入口点
├── ui.py           # 用户界面定义
├── model.py        # 模型处理逻辑
└── database.py     # 数据库操作
```

## 自定义

要集成您自己的模型：

1. 修改[model.py](file:///home/jason/Code/ai4fish_sdu_nyu/deep-learning-app-template/app/model.py)中的`ImageClassifier`类以加载您的训练模型
2. 更新预处理管道以匹配您的模型要求
3. 根据需要调整预测逻辑以适应您的特定用例

## 许可证

此项目根据MIT许可证授权 - 详情请参见LICENSE文件。