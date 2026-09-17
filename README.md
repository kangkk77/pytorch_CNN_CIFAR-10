# CIFAR-10 CNN

使用 PyTorch 和 torchvision 训练一个简单的卷积神经网络，对 CIFAR-10 的 10 类图片进行分类。

## 目录

```text
cifar-10/
├── README.md
├── .gitignore
├── .vscode/          # 编辑器配置
├── .venv/            # 本地 Python 环境
├── train.py          # 训练与评估入口
├── data/             # CIFAR-10 数据和原始压缩包
├── checkpoints/      # 模型权重（现有 cifar_net.pt）
├── results/          # 预留的图表、实验结果目录
├── notes/            # 学习笔记
└── .archive/         # 整理前保留的重复数据与空入口文件，不提交 Git
```

## 运行

在项目根目录的 PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe .\train.py
```

脚本会自动选择 CUDA 或 CPU，训练 20 轮，并输出整体和各类别测试准确率。数据路径固定为脚本所在目录下的 `data/`，不受终端当前目录影响；数据缺失时 torchvision 会尝试下载。

依赖包括 `torch`、`torchvision`、`matplotlib`、`numpy`，已安装在本地虚拟环境中。

当前脚本不自动保存权重或图表，`checkpoints/cifar_net.pt` 是整理前已有的模型文件。

学习笔记见 [notes/note.md](notes/note.md)。数据、虚拟环境、权重和整理备份由根目录 `.gitignore` 忽略。
