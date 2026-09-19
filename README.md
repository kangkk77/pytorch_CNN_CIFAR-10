| 层 | 操作 | 输出尺寸 |
| --- | --- | --- |
| conv1 | Conv2d(3→6, 5×5) + ReLU + MaxPool(2, 2) | 6×14×14 |
| conv2 | Conv2d(6→16, 5×5) + ReLU + MaxPool(2, 2) | 16×5×5 |
| flatten | 展平 | 400 |
| fc1 | Linear(400→120) + ReLU | 120 |
| fc2 | Linear(120→84) + ReLU | 84 |
| fc3 | Linear(84→10) | 10（各类 logits） |

- 卷积层默认 stride=1、无 padding，特征图经 5×5 卷积后每边减 4，再经 2×2 最大池化边长减半。
- 最后一层不接 softmax，直接输出 logits，由 `nn.CrossEntropyLoss` 内部完成 log-softmax。
- 全网络约 62,006 个可训练参数（含偏置）。
## 训练配置
- 损失函数：`nn.CrossEntropyLoss`（交叉熵）
- 优化器：SGD，`lr=0.01`，`momentum=0.9`
- batch size：128，共训练 20 个 epoch
- 预处理：`ToTensor` 后用 `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` 归一化到 [-1, 1]
- 设备：自动选择 CUDA（`cuda:0`）或 CPU
- 评估：训练结束后输出测试集整体准确率，以及 10 个类别各自的准确率