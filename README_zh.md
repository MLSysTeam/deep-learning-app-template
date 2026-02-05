# 深度学习桌面应用模板

一个完整的深度学习桌面应用程序模板，使用PyTorch和PySide6构建，集成了MySQL/SQLite数据库，用于图像分割任务。此示例使用 **[Grounded-SAM模型](https://github.com/IDEA-Research/Grounded-Segment-Anything)** 进行图像分割，但可以轻松适配到其他模型。

该应用程序结构为单一桌面客户端，通过不同文件分离了各种功能：
- **模型推理** 由 [model.py](./app/model.py) 处理
- **用户界面** 由 [ui.py](./app/ui.py) 管理
- **数据库操作** 由 [database.py](./app/database.py) 实现

## 🏗️ 系统架构

此应用程序遵循模块化的桌面应用程序架构，具有清晰的关注点分离：

### 桌面UI层
- **PySide6**: 提供用于上传图像并查看分割结果的桌面GUI
- 处理图像显示和预测可视化
- 管理用户交互和事件

### 应用逻辑层
- **主控制器** ([main.py](./app/main.py)): 协调UI、模型和数据库层之间的通信
- 处理应用程序启动和生命周期
- 协调组件间的数据流

### 机器学习层
- **PyTorch/TorchVision**: 流行的深度学习框架
- 执行图像预处理和模型推理
- 集成Grounded-SAM进行图像分割功能
- 包含基于文本提示的对象检测功能

### 数据层
- **MySQL/SQLite**: 存储分割结果，包括：
  - 图像文件路径
  - 用于分割的文本提示
  - 检测到的对象数量
  - 平均置信度分数
  - 检测时间戳

## 🚀 快速开始

### 先决条件

- Python 3.9+
- MySQL服务器（可选，有SQLite回退选项）
- [UV](https://github.com/astral-sh/uv) 包管理器

### 安装

1. 克隆此仓库：
   ```bash
   git clone https://github.com/MLSysTeam/deep-learning-app-template.git
   cd deep-learning-app-template
   ```

2. 使用UV安装依赖：
   ```bash
   uv sync # 等价于 uv pip install -r requirements.txt
   ```
   安装完成后，你会看到项目根目录中创建了一个`.venv`文件夹。

3. 安装Grounded-Segment-Anything依赖（用于完整分割功能）：
   ```bash
   # 首先导航到Grounded-Segment-Anything目录
   cd app/3rd_party/Grounded-Segment-Anything
   
   # 安装GroundingDINO依赖
   uv pip install -e GroundingDINO
   
   # 安装Segment Anything依赖
   uv pip install -e segment_anything
   
   # 返回项目根目录
   cd ../../..
   ```

4. 下载所需的模型权重：
   ```bash
   # 下载GroundingDINO SwinT-OVC模型 (~694MB)
   cd app/3rd_party/Grounded-Segment-Anything
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   
   # 下载SAM ViT-H模型 (~2.5GB)
   cd app/3rd_party/Grounded-Segment-Anything
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

   _关于模型权重及其使用的更多详细信息可在 [Grounded-Segment-Anything 文档](https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#running_man-grounded-sam-detect-and-segment-everything-with-text-prompt) 中找到。_

### 设置MySQL数据库（可选）

应用程序现在包含带有容错机制的自动数据库创建功能，简化了设置过程：

### 简单设置（推荐）

对于快速测试和开发，应用程序将会自动：

1. 尝试连接到配置的MySQL数据库
2. 如果MySQL不可用或访问被拒绝，则回退到使用本地SQLite数据库
3. 无论使用哪个数据库，都会自动创建所需的表

只需运行应用程序，它就会自动处理数据库初始化！

### 完整MySQL设置（生产环境）

如果你想在生产环境中使用MySQL：

1. **安装MySQL服务器**（一次性设置）
   - 在Ubuntu/Debian上：`sudo apt-get install mysql-server`
   - 在CentOS/RHEL上：`sudo yum install mysql-server`
   - 在macOS上：`brew install mysql`
   - 或从官方MySQL网站下载

2. **启动MySQL服务**
   ```bash
   # 在Ubuntu/Debian上
   sudo systemctl start mysql
   
   # 在macOS上
   brew services start mysql
   ```

3. **创建具有权限的MySQL用户**（如果不使用root）
   ```sql
   CREATE USER 'dl_app_user'@'localhost' IDENTIFIED BY 'secure_password';
   GRANT ALL PRIVILEGES ON *.* TO 'dl_app_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

### 环境配置

更新你的环境变量：

1. 复制`.env.example`到`.env`：
   ```bash
   cp .env.example .env
   ```

2. 使用你的MySQL凭据编辑`.env`：
   ```bash
   DB_USER=your_mysql_username
   DB_PASSWORD=your_mysql_password
   DB_HOST=localhost
   DB_PORT=3306
   DB_NAME=image_segmentation
   ```

> **注意**：如果应用程序无法连接到MySQL（由于凭据错误、MySQL未运行等原因），它将自动回退到使用本地SQLite数据库（`image_segmentations.db`）进行开发和测试。

### 替代方案：使用Docker运行MySQL

对于专用的MySQL设置，您可以使用Docker：

```bash
docker run --name dl-mysql-container \
  -e MYSQL_ROOT_PASSWORD=rootpassword \
  -p 3306:3306 \
  -d mysql:8.0
```

然后相应地更新您的`.env`文件：
```bash
DB_USER=root
DB_PASSWORD=rootpassword
DB_HOST=localhost
DB_PORT=3306
DB_NAME=image_segmentation
```

4. 复制环境示例文件：
   ```bash
   cp .env.example .env
   ```

   然后在`.env`中更新您的数据库凭据。

### 运行应用程序

使用提供的脚本运行应用程序：
   ```bash
   ./run_app.sh
   ```

或者直接运行：
   ```bash
   uv run python app/main.py
   ```

### 使用方法

1. 使用上面的命令启动应用程序
2. PySide6桌面应用程序将启动
3. 使用GUI上传图像文件（JPG、PNG等）
4. 输入描述您想要分割的内容的文本提示
5. 点击"Segment Image"执行分割
6. 查看带掩码覆盖在原始图像上的分割结果
7. 结果存储在数据库中并在历史面板中显示

## 📁 项目结构

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # PySide6应用程序入口点和控制器
│   ├── model.py         # ML模型处理逻辑和分割
│   ├── database.py      # 数据库模型和连接
│   └── ui.py            # PySide6 GUI实现
│   └── 3rd_party/       # 第三方模型代码
│       └── Grounded-Segment-Anything/  # Grounded-SAM模型代码
├── uploads/             # 存储上传图像的目录
├── pyproject.toml       # 项目依赖和元数据
├── requirements.txt     # 依赖列表
├── .env.example         # 环境变量示例
├── run_app.sh           # 启动桌面应用程序的脚本
├── README.md            # 此文件
└── README_zh.md         # README的中文版本
```

## 🔧 自定义

### 添加您自己的模型

要集成您自己的PyTorch模型：

1. 修改[app/model.py](app/model.py)以加载您的模型：
   - 更新`__init__`方法以加载您的特定模型
   - 调整`segment`方法以处理您的模型的输入/输出格式
   - 如果您的模型需要不同的预处理，请修改`preprocess_image`方法

2. 如有需要，请更新分割类：
   - 用您的特定模型需求替换示例分割逻辑

### 数据库模式

应用程序自动创建以下表：

```sql
CREATE TABLE image_segmentations (
    id INTEGER AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    text_prompt VARCHAR(255),  -- 用于分割的文本提示的新字段
    num_objects INTEGER,       -- 检测到的对象数量
    confidence_avg FLOAT,      -- 平均置信度分数
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🛠️ 使用的技术

- **前端**: PySide6
- **后端**: Python
- **数据库**: MySQL/SQLite
- **机器学习框架**: PyTorch, TorchVision
- **包管理**: UV, pip
- **图像处理**: OpenCV, Pillow, Matplotlib

## 🤝 贡献

欢迎贡献！随时提交拉取请求或开启问题以改进此模板。

## 📄 许可证

此项目根据MIT许可证授权 - 详情请见LICENSE文件。