# 电影推荐系统

## 目录

- [简介](#简介)
- [功能](#功能)
- [安装](#安装)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [正在实现的功能](#正在实现的功能)
- [未来会加入的功能](#未来会加入的功能)
- [可能加入的功能](#可能加入的功能)
- [贡献](#贡献)
- [许可证](#许可证)

## 简介

本项目是一个使用Flask、Pandas和Scikit-learn构建的电影推荐系统。它包括基于用户和基于物品的协同过滤算法，并使用交互式图表可视化推荐结果和评估指标。

## 功能

- 用户和物品的协同过滤推荐算法
- 使用Chart.js可视化推荐结果
- 显示不同推荐算法的RMSE（均方根误差）
- 支持用户评分的实时更新和反馈

## 安装

### 前提条件

- Python 3.7或更高版本
- MySQL数据库

### 步骤

1. 克隆仓库：
    ```bash
    git clone https://github.com/yeliccc/ece4010project.git
    cd ece4010project
    ```

2. 创建虚拟环境并激活：
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # 对于Windows用户，使用 `venv\Scripts\activate`
    ```

3. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

4. 配置数据库：
    - 确保MySQL数据库已安装并运行。
    - 在MySQL中创建一个名为`movielens`的数据库，并导入数据。可以使用create_movielens.sql脚本，需要重启数据库可以使用reload_movielens
    - 在项目根目录创建一个名为`config.py`的文件，添加数据库配置信息：

    ```python
   DB_CONFIG = {
        'host': 'localhost',
        'user': 'your_username',
        'password': 'your_password',
        'database': 'movielens'
    }
    ```

5. 运行应用程序：
    ```bash
    python app.py
    ```

## 使用方法

1. 启动应用后，在浏览器中访问`http://127.0.0.1:5000/`。
2. 在主页上输入用户ID和推荐数量，选择推荐算法，点击“Get Recommendations”获取推荐结果。
3. 推荐结果页面将显示推荐的电影列表及其评分，同时展示推荐结果的柱状图。
4. 新用户可以进行测试来获得推荐的电影
## 项目结构

```plaintext
movierecsys/
├── app.py
├── requirements.txt
├── config.py
├── data/
│   ├── __init__.py
│   ├── load_data.py
├── models/
│   ├── __init__.py
│   ├── collaborative_filtering.py
├── templates/
│   ├── index.html
│   ├── recommendations.html
│   ├── feedback_thankyou.html
└── utils/
    ├── __init__.py
    ├── database.py
    ├── evaluation.py
```

- `app.py`：主应用程序文件，定义了Flask路由和逻辑。
- `requirements.txt`：项目依赖列表。
- `config.py`：数据库配置文件。
- `data/`：包含数据加载和预处理脚本。
- `models/`：包含推荐算法实现。
- `templates/`：包含HTML模板文件。
- `utils/`：包含数据库连接和评估函数。

## 正在实现的功能

- **深度学习推荐系统**：使用TensorFlow或PyTorch构建深度学习模型，提供更准确的推荐结果。

## 未来会加入的功能

- **矩阵分解方法**：引入如ALS（交替最小二乘）或NMF（非负矩阵分解）等矩阵分解方法，以提高推荐精度。
- **实时推荐**：实现实时推荐系统，能够在用户交互时实时更新推荐结果。

## 可能加入的功能

- **社交推荐**：结合社交网络数据，提供基于社交关系的推荐服务。
- **多模态推荐**：结合用户的浏览历史、评分、评论等多种数据源，提供更全面的推荐服务。
- **移动应用支持**：开发移动端应用，使用户可以随时随地获取电影推荐。

## 贡献

欢迎贡献者！如果有任何改进建议或发现了问题，请提交issue或创建pull request。

## 许可证

本项目基于MIT许可证进行分发。详情请参见LICENSE文件。

