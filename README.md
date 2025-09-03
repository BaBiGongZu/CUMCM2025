# CUMCM2025
This is a repository belonging to a group from ZJU for CUMCM2025

以下为 AI 生成的 README 内容：

# 2025年全国大学生数学建模竞赛 - [题目编号/名称，例如：A题]

本项目是为备战2025年全国大学生数学建模竞赛（CUMCM）而创建的代码仓库。

## 团队成员

* [姓名1] - 负责：[例如：建模与论文]
* [姓名2] - 负责：[例如：编程与求解]
* [姓名3] - 负责：[例如：数据处理与可视化]

---

## 🚀 快速开始 (Quick Start)

### 1. 环境配置

本项目使用 Python 3.x。为了保证团队成员环境一致，请使用虚拟环境。

```bash
# 1. 克隆仓库
git clone [你的仓库SSH或HTTPS链接]
cd [仓库名]

# 2. 创建并激活虚拟环境
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. 安装所有依赖
pip install -r requirements.txt
```

### 2. 运行代码

我们约定所有分析脚本都应能独立运行。核心脚本位于 `scripts/` 目录下，并按数字顺序组织。

```bash
# 运行单个脚本（例如，第一问的求解）
python scripts/02_solve_question_one.py

# (推荐) 运行主程序，它将按顺序执行所有步骤
python scripts/main.py
```

---

## 📁 项目结构说明 (Project Structure)

为了保持代码的清晰和团队协作的顺畅，我们采用以下统一的目录结构：

```
.
├── data/               # 存放所有数据文件
│   ├── raw/            # 存放从官网下载的原始数据（只读）
│   └── processed/      # 存放经过预处理、清洗后的数据
│
├── scripts/            # 存放可独立运行的分析脚本，按解题顺序命名
│   ├── 01_data_preprocessing.py
│   ├── 02_solve_question_one.py
│   └── main.py
│
├── src/                # 存放我们自己编写的、被多处调用的公用函数
│   ├── data_utils.py   # 数据处理的公用函数
│   └── plot_utils.py   # 统一绘图风格的函数
│
├── third_party/                # 存放从外部（如Github）复用的、非pip安装的第三方算法
│   └── particle_swarm.py
│
├── results/            # 存放所有代码生成的最终结果，用于上传到Overleaf
│   ├── figures/        # 所有论文中要使用的图片
│   └── tables/         # 所有论文中要使用的表格数据（.csv格式）
│
├── .gitignore          # Git忽略文件配置
├── README.md           # 项目说明文件（就是你正在看的这个）
└── requirements.txt    # 项目Python依赖库清单
```

### 各文件夹用途详解

* **`data/`**: 数据中心。`raw` 子目录下的原始数据下载后**不应被任何代码修改**，以保证数据源的纯净。所有数据处理脚本都应从 `raw` 读取，并将处理后的结果存入 `processed`。
* **`scripts/`**: 核心工作区。每个文件都应聚焦于解决问题的一个特定部分，并按 `01_`、`02_`... 的顺序命名，清晰地反映我们的解题思路。
* **`src/`**: 我们自己的“工具箱”。当你发现一段代码（比如一个特定的数据清洗逻辑）可能在多个脚本里都会用到时，就应该把它封装成一个函数，放到这里。
* **`third_party/`**: “外援”工具箱。专门用来存放那些我们从网上找到的、又不能通过 `pip` 安装的优秀算法代码。
* **`results/`**: 成果输出地。**所有脚本生成的、需要放入最终论文的图、表等，都必须保存到这个文件夹**，方便负责论文的同学查找和上传。
* **`requirements.txt`**: 环境的“快照”。当团队中有人添加了新的库时，应及时更新此文件 (`pip freeze > requirements.txt`) 并提交，以保持团队环境同步。

---

## ✍️ 论文撰写

我们的论文在 Overleaf 上进行协作撰写。

* **Overleaf项目链接**: [在此处粘贴你们的Overleaf项目链接]

所有需要插入论文的图表，请从本项目的 `results/` 文件夹中获取最新版本。