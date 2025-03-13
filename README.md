# PhaseUnwrappingNet

## Project Structure
- `data/`: Contains training and testing datasets.数据集（训练集及测试集）
- `models/`: Contains model definitions.模型定义
- `utils/`: Contains utility functions. #数据加载器+评估指标计算+训练过程中使用的工具函数
- `configure.py`: Configuration file.配置文件
- `main.py`: Main program for training and testing.对训练及测试的程序文件
- `requirements.txt`: List of dependencies.依赖的库
- `README.md`: Project documentation.主要说明
- `change_num`: 批量改导入图像的编号
- `residue.py`: 残点生成
- `results`: 训练生成指标

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.

## Usage
- Run the main program: `python main.py`.

# (- 将你的训练图像文件放入 data/train/images/ 文件夹中。如果训练数据还包括标签（例如，用于监督学习的标签图像）)#
# (- 则将标签文件放入 data/train/labels/ 文件夹中。)
# (- 确保文件名能够正确匹配，以便在加载数据时能够找到对应的图像和标签)