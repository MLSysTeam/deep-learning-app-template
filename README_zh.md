# 深度学习应用模板

一个完整的深度学习应用模板，包含Streamlit前端、FastAPI后端和MySQL数据库，用于图像分类任务。

## 🏗️ 系统架构

此应用程序采用现代、可扩展的架构：

### 前端层
- **Streamlit**: 提供交互式UI用于上传图像并查看分类结果
- 处理图像显示和预测可视化
- 通过REST API与后端通信

### 后端层
- **FastAPI**: 用于创建REST API的高性能Web框架
- 处理图像预处理和模型推理
- 管理与数据库的通信
- 实现异步请求处理

### 机器学习层
- **PyTorch/TorchVision**: 用于实现和运行图像分类模型
- 包含预处理管道和预测逻辑
- 设计为与标准图像分类架构配合使用

### 数据层
- **MySQL**: 存储分类结果，包括：
  - 图像文件路径
  - 预测类别标签
  - 置信度分数
  - 预测时间戳

## 🚀 快速开始

### 先决条件

- Python 3.8+
- MySQL服务器
- [UV](https://github.com/astral-sh/uv) 包管理器

### 安装

1. 克隆此仓库：
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. 使用UV安装依赖：
   ```bash
   uv sync
   # 或者如果您更喜欢pip：
   pip install -r requirements.txt
   ```

3. 设置MySQL数据库：
   - 创建用于图像分类的数据库
   - 在`.env`中更新数据库凭据（参见`.env.example`）

4. 复制环境示例文件：
   ```bash
   cp .env.example .env
   ```
   
   然后在`.env`中更新您的数据库凭据。

### 运行应用程序

#### 方法1：分终端运行

1. 启动后端（在终端1中）：
   ```bash
   ./start_backend.sh
   ```
   或直接运行：
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. 启动前端（在终端2中）：
   ```bash
   ./start_frontend.sh
   ```
   或直接运行：
   ```bash
   streamlit run app/frontend.py
   ```

#### 方法2：使用进程管理器

或者，您可以使用进程管理器如`pm2`启动两个服务：

1. 安装pm2：
   ```bash
   npm install -g pm2
   ```

2. 启动两个服务：
   ```bash
   pm2 start ecosystem.config.js
   ```

> 注意：如果使用PM2，请创建`ecosystem.config.js`文件（此模板中未包含）。

### 使用方法

1. 在`http://localhost:8501`访问Streamlit前端
2. 上传图像文件（JPG、PNG等）
3. 单击"Classify Image"将图像发送到后端
4. 在前端查看分类结果
5. 结果存储在MySQL数据库中

## 📁 项目结构

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI后端应用程序
│   ├── frontend.py      # Streamlit前端应用程序
│   ├── database.py      # 数据库模型和连接
│   └── model_handler.py # 机器学习模型处理逻辑
├── uploads/             # 存储上传图像的目录
├── pyproject.toml       # 项目依赖和元数据
├── requirements.txt     # 依赖列表
├── .env.example         # 环境变量示例
├── start_backend.sh     # 启动后端服务的脚本
├── start_frontend.sh    # 启动前端服务的脚本
├── README.md            # 此文件
└── README_zh.md         # README的中文版本
```

## 🔧 自定义

### 添加您自己的模型

要集成您自己的PyTorch模型：

1. 修改[app/model_handler.py](app/model_handler.py)以加载您的模型：
   - 更新`__init__`方法以加载您的特定模型
   - 调整`predict`方法以处理您的模型的输入/输出格式
   - 如果您的模型需要不同的预处理，请修改`preprocess_image`方法

2. 如有需要，请更新分类类：
   - 用您的特定类替换示例ImageNet类

### 数据库模式

应用程序自动创建以下表：

```sql
CREATE TABLE image_classifications (
    id INTEGER AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    predicted_class VARCHAR(100),
    confidence VARCHAR(10),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🛠️ 使用的技术

- **前端**: Streamlit
- **后端**: FastAPI
- **数据库**: MySQL
- **机器学习框架**: PyTorch, TorchVision
- **包管理**: UV, pip
- **图像处理**: Pillow

## 🤝 贡献

欢迎贡献！随时提交拉取请求或开启问题以改进此模板。

## 📄 许可证

此项目根据MIT许可证授权 - 详情请见LICENSE文件。