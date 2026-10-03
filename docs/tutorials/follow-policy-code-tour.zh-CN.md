# 跟随策略代码逐行讲解：从 YOLO 观测到行为克隆

本文按当前源码行号解释三个文件。每个按行号讲解的小节都会先贴出对应源码，再在代码下方逐行解释；你可以直接在网页里对照阅读。完整源码：[follow_policy.py](../../scripts/follow_policy.py)、[train_follow_policy.py](../../scripts/train_follow_policy.py)、[duck_vision_follow.py](../../scripts/duck_vision_follow.py)。行号以这版工程为准；空行只负责排版，注释和文档字符串用于说明，不参与 Python 计算。

先记住这一句：**YOLO 看图找人；跟随策略不看图片，它只接收 YOLO 算出的 7 个数字，再输出两个移动命令。训练时，神经网络模仿的是规则控制器给出的命令。**

## 1. 整条数据流

```text
MuJoCo 摄像头图像
    ↓
YOLO 找出 person 边界框
    ↓ 计算 bearing、box_area、box_height 等 7 个特征
规则控制器计算 teacher_vx、teacher_vyaw
    ↓ 把“7 个输入 + 2 个老师答案”写进 CSV
train_follow_policy.py 读取 CSV，训练 FollowActor
    ↓
follow-policy.ts（可执行的 TorchScript 模型）
    ↓
duck_vision_follow.py 再用 YOLO 算 7 个特征，模型预测 vx、vyaw
    ↓
robot.move → robotd → 已有的鸭子步态策略
```

这里有两个不同的神经网络，容易混淆：YOLO 是“识别人”的视觉模型；`FollowActor` 是“看 7 个检测数字后决定怎么移动”的小策略网络。`follow-policy.ts` 不接收原始图片，也没有重新训练 YOLO。

## 2. 一条训练数据是什么

现有 CSV 的一行长这样：

```text
bearing=0.728, box_area=0.0239, box_height=0.204,
confidence=0.849, bearing_rate=0, centered=0, holding_distance=0,
teacher_vx=0.30, teacher_vyaw=-0.692
```

前七项是输入 `x`，后两项是老师给的答案 `y`。`timestamp` 只用于记录样本何时产生，不在 `FEATURE_COLUMNS` 中，所以网络看不到时间戳。这个 CSV 也不是 YOLO 标注数据：它不保存图片或人工框，而是保存 YOLO 已经算出的数字和规则控制器答案。这个例子表示：人出现在画面右侧，规则控制器要求鸭子以前进速度 0.30 m/s 向前走，并用负 yaw 向右转。训练的目标是让网络在相似输入下预测出相似命令。

`bearing` 是归一化图像横向偏差：左负右正，约在 -1 到 +1；`box_area` 是 YOLO 框面积占整张图的比例；`box_height` 是目标框高度占画面高度的比例；`confidence` 是 YOLO 对分类的置信度；`bearing_rate` 是 bearing 随时间的变化速度；`centered` 记录规则控制器进入“居中”状态没有；`holding_distance` 记录它此前是否处于距离保持状态。最后两个状态位让网络知道控制器的滞回状态，避免只看当前一帧时无法复现“为什么这次继续转、或者为什么这次停车”。

## 3. `follow_policy.py`：定义输入输出契约和网络

### 第 1–7 行：模块说明与依赖

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
"""Shared observation contract and small policy for visual following."""

from __future__ import annotations

import torch
from torch import nn

~~~

第 1 行说明这个文件同时定义视觉跟随的观测格式和小策略模型。第 3 行启用 Python 的延迟注解解析，方便类型标注。第 5 行导入 PyTorch；第 6 行单独导入神经网络组件 `torch.nn`，后面用它声明线性层、激活函数和模块基类。空行把模块说明和代码分开。

### 第 8–19 行：模型两端必须遵守的顺序和单位

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
FEATURE_COLUMNS = (
    "bearing",
    "box_area",
    "box_height",
    "confidence",
    "bearing_rate",
    "centered",
    "holding_distance",
)
ACTION_COLUMNS = ("vx", "vyaw")
MAX_FORWARD_MPS = 0.40
MAX_YAW_RAD_S = 0.95
~~~

第 8–16 行的 `FEATURE_COLUMNS` 是 7 个输入的固定顺序。这不只是列名清单：收集 CSV、训练张量、模型输入必须完全同序。比如若把 `bearing` 和 `box_area` 调换，程序不会报错，但网络会把面积当方位使用，行为就会错。

第 17 行的 `ACTION_COLUMNS` 固定两个输出的顺序：先 `vx`，再 `vyaw`。第 18 行规定最大前进输出为 0.40 m/s；第 19 行规定最大转向输出绝对值为 0.95 rad/s。它们既限制网络的输出范围，也让数据读取器可以拒绝超范围标签。

### 第 22–35 行：网络结构和保存的数据

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
class FollowActor(nn.Module):
    """Map normalized YOLO target features to bounded body velocity commands."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, -1).float())
        self.register_buffer("scale", scale.reshape(1, -1).float().clamp_min(1e-4))
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_COLUMNS), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, len(ACTION_COLUMNS)),
        )
~~~

第 22 行声明 `FollowActor` 是 PyTorch 神经网络模块；第 23 行说明任务：将归一化的目标观测映射为有边界的速度命令。

第 25 行的构造函数接收训练数据计算出的 `mean` 和 `scale`。第 26 行调用父类初始化，这是 PyTorch 模块的标准步骤。第 27 行把 7 个均值改成形状 `[1, 7]` 的浮点张量，并用 `register_buffer` 注册；buffer 会随模型保存、加载和迁移设备，但它不是需要梯度更新的权重。第 28 行对标准差也做相同处理，并用 `clamp_min(1e-4)` 避免某列标准差为零时除零。

第 29–35 行定义多层感知机：第一个 `Linear(7, 64)` 把 7 个输入变成 64 个隐藏数；第 31 行 `Tanh` 增加非线性；第二个 `Linear(64, 64)` 再组合这些信息；第 33 行再次使用 `Tanh`；最后 `Linear(64, 2)` 产生两个未限制的原始输出。它不是卷积网络，因为输入已经是 7 个数而不是像素。

### 第 37–42 行：前向计算和安全输出范围

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.mean) / self.scale
        raw = self.network(normalized)
        vx = torch.sigmoid(raw[..., 0]) * MAX_FORWARD_MPS
        vyaw = torch.tanh(raw[..., 1]) * MAX_YAW_RAD_S
        return torch.stack((vx, vyaw), dim=-1)
~~~

第 37 行定义模型每次预测所调用的 `forward`。第 38 行用 `(输入 - 均值) / 标准差` 标准化特征，让不同量纲的列更容易一起训练。第 39 行将标准化特征送入 MLP，得到原始输出。第 40 行对前进分量用 sigmoid，把任意实数压进 0 到 1，再乘 0.40，所以网络不能输出倒车，也不能要求超过 0.40 m/s。第 41 行对 yaw 用 tanh，把数压进 -1 到 +1，再乘 0.95，所以转向范围是 ±0.95 rad/s。第 42 行把两个分量重新叠成最后一维为 2 的张量；单条输入形状为 `[1, 7]` 时，输出形状为 `[1, 2]`。

## 4. `train_follow_policy.py`：把 CSV 变成 TorchScript

### 第 1–22 行：入口说明和依赖

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
#!/usr/bin/env python3
"""Behaviour-clone a small visual-follow policy from simulator demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from follow_policy import (
    ACTION_COLUMNS,
    FEATURE_COLUMNS,
    FollowActor,
    MAX_FORWARD_MPS,
    MAX_YAW_RAD_S,
)
~~~

第 1 行声明脚本可以由 Unix 环境直接执行；第 2 行指出这是行为克隆。第 4 行启用延迟注解。第 6–9 行分别导入命令行解析、CSV、JSON 和路径处理。第 11 行导入 NumPy，负责数值数组和有限值检查；第 12 行导入 PyTorch；第 13 行引入损失函数所在的神经网络命名空间；第 14 行导入小批次数据工具。

第 16–22 行从 `follow_policy.py` 导入动作列、特征列、模型和两个动作上限。复用这些常量可防止训练脚本另写一套列顺序或上限。

### 第 25–38 行：训练命令的参数

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
def parse_args() -> argparse.Namespace:
    default_output = Path.home() / ".cache/duck-sim/follow-training/follow-policy.ts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="+", type=Path, help="one or more --record-csv demonstration files")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch size, and patience must be positive")
    return args
~~~

第 25 行定义参数解析函数。第 26 行将默认模型保存到用户缓存目录 `~/.cache/duck-sim/follow-training/follow-policy.ts`。第 27 行创建参数解析器。第 28 行要求至少给一个 CSV；`nargs="+"` 让多个演示文件可以合并训练。第 29 行允许覆盖模型输出路径。第 30 行设定最多训练 400 轮；第 31 行每批最多取 128 行；第 32 行如果连续 50 轮验证集不再改善，就提前停止；第 33 行设置随机种子；第 34 行允许自动选设备、CPU 或 Apple MPS。

第 35 行真正解析命令行。第 36–37 行拒绝非正的轮数、批大小或耐心轮数。第 38 行返回已检查的参数对象。

### 先采集老师示范

采集要分两个终端：终端一启动带人物和相机的仿真，终端二启动视觉跟随程序。采集时不要传 `--policy`，这样程序用规则控制器发命令，并把同一时刻的特征和规则动作保存到 CSV。

终端一：

```sh
cd /Users/suhao/github/robotic_ai/microduck
export DUCK_SIM_RL=/Users/suhao/github/robotic_ai/microduck_rl
export DUCK_SIM_SCENE=approach
export DUCK_SIM_CAMERAS=a
export DUCK_SIM_KEYFRAME=HOME
export DUCK_SIM_MOVE_TARGET=follow_person
scripts/duck-sim up
```

终端二：

```sh
cd /Users/suhao/github/robotic_ai/microduck
export DUCK_SIM_RL=/Users/suhao/github/robotic_ai/microduck_rl
"$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py \
  --target person --drive --seconds 90 \
  --max-forward 0.40 --stop-area 0.06 --stop-height 0.45 \
  --record-csv "$HOME/.cache/duck-sim/follow-training/run-01.csv"
```

`--seconds 90` 表示采集约 90 秒；`--record-csv` 指定输出文件。脚本只在 YOLO 找到目标的检测帧写入样本，所以 90 秒不等于 450 条有效记录。CSV 以追加方式打开；每次单独实验请换一个文件名，避免把不同控制参数的数据悄悄混在一起。

### 再训练学生策略

训练示例：

```sh
cd /Users/suhao/github/robotic_ai/microduck
export DUCK_SIM_RL=/Users/suhao/github/robotic_ai/microduck_rl
"$DUCK_SIM_RL/.venv/bin/python" scripts/train_follow_policy.py \
  /Users/suhao/.cache/duck-sim/follow-training/seed-17.csv
```

默认输出路径就是上述缓存里的 `follow-policy.ts`。传多个 CSV 路径可以合并多个场景的数据。

### 第 41–71 行：读取和清理监督样本

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
def read_demonstrations(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    actions: list[list[float]] = []
    required = (*FEATURE_COLUMNS, "teacher_vx", "teacher_vyaw")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = set(required) - set(reader.fieldnames or ())
            if missing:
                raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    x = [float(row[name]) for name in FEATURE_COLUMNS]
                    y = [float(row["teacher_vx"]), float(row["teacher_vyaw"])]
                except (TypeError, ValueError) as error:
                    raise SystemExit(f"invalid number in {path}:{line_number}: {error}") from error
                if np.isfinite(x).all() and np.isfinite(y).all():
                    features.append(x)
                    actions.append(y)
    if len(features) < 100:
        raise SystemExit(
            f"need at least 100 visible-target samples; found {len(features)}. "
            "Collect a longer run or add demonstrations from another target-motion seed."
        )
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32)
    if np.any(y[:, 0] < -1e-5) or np.any(y[:, 0] > MAX_FORWARD_MPS + 1e-5):
        raise SystemExit(f"forward labels must be within 0..{MAX_FORWARD_MPS} m/s")
    if np.any(np.abs(y[:, 1]) > MAX_YAW_RAD_S + 1e-5):
        raise SystemExit(f"yaw labels must be within ±{MAX_YAW_RAD_S} rad/s")
    return x, y
~~~

第 41 行声明读取函数，返回输入矩阵 `x` 和标签矩阵 `y`。第 42–43 行创建两个列表：一个存特征，一个存老师动作。第 44 行列出每个 CSV 必须包含的 9 列。

第 45 行逐个读取给定的 CSV；第 46 行用 UTF-8 文本模式打开，并让 CSV 模块负责换行处理；第 47 行将每行解析成“列名到字符串值”的字典。第 48 行找出缺少的列；第 49–50 行若缺列就指出具体文件和列名后停止。

第 51 行从第 2 行开始记录行号，因为第 1 行是表头。第 52–56 行把 7 个输入列和 2 个老师动作转成浮点数；若空值或文本不能转为数值，就报出文件和行号。第 57 行只接受所有值都是有限数的记录，排除 `NaN` 和无穷大；第 58–59 行分别把输入和标签添加到列表。

第 60–64 行要求至少 100 条有效的目标可见记录；样本过少就提示继续采集或加入其他演示。第 65 行把特征转为 `float32` 矩阵，行是样本、列是特征；第 66 行用同样格式保存动作标签。第 67–70 行检查 vx 在 0–0.40 内、yaw 在 ±0.95 内，确保监督答案与模型设计范围一致。第 71 行返回两个矩阵。

### 第 74–108 行：固定随机性、划分数据、创建训练对象

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y = read_demonstrations(args.data)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested, but this machine does not expose an Apple GPU")
    print(f"Loaded {len(x)} samples from {len(args.data)} demonstration file(s); training on {device}.")

    permutation = torch.randperm(len(x))
    validation_count = max(1, int(round(len(x) * 0.2)))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    x_train = x_tensor[train_indices]
    y_train = y_tensor[train_indices]
    x_validation = x_tensor[validation_indices]
    y_validation = y_tensor[validation_indices]

    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-3)
    model = FollowActor(mean, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    action_scale = torch.tensor([MAX_FORWARD_MPS, MAX_YAW_RAD_S], device=device)
    dataset = TensorDataset(x_train.to(device), y_train.to(device))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validation_x = x_validation.to(device)
    validation_y = y_validation.to(device)
~~~

第 74 行是训练主函数，第 75 行解析命令行。第 76–77 行分别固定 PyTorch 和 NumPy 的随机种子，让切分和初始化在相同条件下尽量复现。第 78 行读取全部 CSV。

第 80–83 行决定计算设备：`auto` 优先用 Apple MPS，否则用 CPU；明确指定时就使用用户选择。第 84–85 行检查用户要求 MPS 但系统不可用的情况。第 86 行打印样本数、CSV 文件数和设备。

第 88 行生成样本行号的随机排列。第 89 行取约 20% 作验证集，至少保留一行；第 90–91 行把排列前段当验证索引，其余当训练索引。第 92–93 行把 NumPy 数组转为 PyTorch 张量；第 94–97 行根据索引拆出训练输入、训练标签、验证输入和验证标签。

第 99 行只用训练集计算每个输入列的均值；第 100 行计算总体标准差，并把低于 `1e-3` 的值抬高，避免标准化时除以极小数。验证集没有参与均值和标准差估算，可避免把验证数据统计泄漏进训练预处理。第 101 行用这些统计量构造模型并放到目标设备。第 102 行创建 AdamW 优化器，学习率为 `0.002`，权重衰减为 `0.0001`。第 103 行选择 Smooth L1 损失，对大误差比纯平方误差更稳健。第 104 行创建 `[0.40, 0.95]` 动作缩放向量。第 105–106 行把训练张量包装为数据集，并创建每批 128 条、每轮打乱顺序的数据加载器。第 107–108 行把验证数据放到计算设备上。

### 第 110–145 行：训练、验证和选出最佳轮次

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
    best_loss = float("inf")
    best_weights = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            prediction = model(batch_x)
            loss = loss_fn(prediction / action_scale, batch_y / action_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_x)
            validation_loss = loss_fn(
                validation_prediction / action_scale,
                validation_y / action_scale,
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d} · validation scaled loss {validation_loss:.5f}", flush=True)
        if stale_epochs >= args.patience:
            break

    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    with torch.inference_mode():
        prediction = model(x_validation.to(device)).cpu().numpy()
    mae = np.abs(prediction - y_validation.numpy()).mean(axis=0)
~~~

第 110 行把最佳验证损失设为无穷大；第 111 行还没有保存过最佳权重；第 112 行连续未改善的轮数从零开始。第 113 行从第 0 轮开始，最多到参数指定的轮数。

第 114 行切换到训练模式。第 115 行逐小批处理训练数据。第 116 行让模型根据输入预测动作。第 117 行先分别除以 vx、yaw 的上限再算 Smooth L1，让两个不同单位的动作在损失里有可比尺度；这一步只改变训练误差的衡量尺度，不改变模型实际输出的单位。第 118 行清空旧梯度；第 119 行反向传播计算每个权重对损失的梯度；第 120 行由 AdamW 根据梯度更新权重。

第 122 行切换到评估模式。第 123 行关闭梯度记录，节省验证计算。第 124 行对验证集预测。第 125–128 行以相同的动作缩放方式计算验证损失并转为普通 Python 数字。第 129 行判断验证损失是否创下新低；第 130 行保存新低；第 131 行逐项复制网络状态到 CPU，作为最佳快照；第 132 行把“连续未改善轮数”清零。第 133–134 行若没有改进，就将计数加一。

第 135–136 行在第一轮和每 50 轮打印验证损失。第 137–138 行若连续未改善达到 patience，就结束训练。第 140 行确保至少有一轮产生过权重；第 141 行恢复验证表现最好的那一轮，而不是盲目使用最后一轮；第 142 行设为评估模式。第 143–145 行再次预测验证集，并计算 vx 和 yaw 的平均绝对误差（MAE）。

### 第 147–174 行：导出模型和写指标

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_model = model.to("cpu").eval()
    example = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32)
    torch.jit.trace(export_model, example).save(str(args.output))
    metadata = {
        "algorithm": "behaviour cloning from the existing rule-based simulator controller",
        "features": list(FEATURE_COLUMNS),
        "actions": list(ACTION_COLUMNS),
        "samples": int(len(x)),
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "validation_mae": {"vx_mps": float(mae[0]), "vyaw_rad_s": float(mae[1])},
        "validation_split": "random sample split; adjacent simulator frames are correlated",
        "source_files": [str(path) for path in args.data],
        "seed": args.seed,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved TorchScript policy to {args.output}\n"
        f"Saved metadata to {metadata_path}\n"
        f"Validation MAE · vx {mae[0]:.3f} m/s · vyaw {mae[1]:.3f} rad/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
~~~

第 147 行确保输出目录存在。第 148 行把模型移到 CPU，TorchScript 部署时不依赖训练用的 MPS 设备。第 149 行构造一个形状 `[1, 7]` 的示例输入。第 150 行用 `torch.jit.trace` 记录模型前向计算，并将模型保存为指定 `.ts` 文件。

第 151–162 行构造 JSON 元数据：算法名称、输入列、输出列、总样本数、训练/验证样本数、验证 MAE、验证切分方式、来源 CSV 和随机种子。第 163 行把指标文件命名为 `模型文件名.ts.json`；第 164 行以缩进 JSON 写入 UTF-8 文件。第 165–169 行打印两个输出路径及人类可读的 MAE。第 170 行表示正常返回。第 173–174 行保证文件作为脚本直接运行时会调用 `main()`，并把返回码交给操作系统。

## 5. `duck_vision_follow.py`：采集老师示范，或加载学生模型

这个文件近 500 行，除了训练接口，还包括相机协议、YOLO 推理、跟随状态机、搜索和安全退出。下面先贴对应源码片段，再按数据流和源文件行号逐行解释。

### 第 1–42 行：程序说明、导入和协议常量

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
#!/usr/bin/env python3
"""Detect a named YOLO class in the MuJoCo duck camera and optionally approach it.

The first run is perception-only:

    "$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py --target person

Add --drive to enable a deliberately slow, finite simulation-only follow:

    "$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py --target person --drive

The body server sends length-prefixed 640x360 UYVY frames on port 7901. The simulator renders an
upright landscape view, so `--rotate` defaults to 0; the physical camera's mount correction remains
90 degrees. Commands go to the local robotd Unix socket and use its normal deadman: if this process
stops refreshing a command, the duck stops within 500 ms. This script never reaches a physical
robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import struct
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from follow_policy import FEATURE_COLUMNS

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
FRAME_BYTES = CAMERA_WIDTH * CAMERA_HEIGHT * 2  # UYVY: two bytes per pixel.
DEFAULT_ROBOT_SOCKET = Path.home() / ".cache/duck-sim/duck-a.sock"
DEFAULT_MODEL = Path.home() / ".cache/duck-sim/vision/yolo26n.pt"
DEFAULT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
~~~

第 1–17 行是可直接运行脚本的声明和模块文档。第 2 行说明任务；第 4–10 行给出“只检测”和“检测加移动”示例；第 12–16 行说明相机数据是 640×360 UYVY、仿真图像无需旋转、命令走本机 Unix socket，意图停止刷新后有 500 ms deadman，而且程序不会控制实体鸭子。

第 19 行启用延迟注解。第 21–29 行导入参数解析、CSV、JSON、socket、二进制结构解析、系统退出、时间、路径和模型下载模块。第 31 行 OpenCV 用于颜色格式转换和旋转；第 32 行 NumPy 把相机字节变成数组；第 33 行 PyTorch 执行策略；第 34 行 Ultralytics 提供 YOLO；第 35 行导入训练与推理共同遵守的特征顺序。

第 37–39 行固定相机宽高，并计算 UYVY 每像素 2 字节，所以一帧共 `640×360×2` 字节。第 40 行是默认 robotd socket；第 41 行是 YOLO 权重缓存路径；第 42 行是权重首次下载地址。

### 第 45–79 行：接收图像、解码、发移动命令

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
def read_exact(stream: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.recv(size - len(data))
        if not chunk:
            raise ConnectionError("camera stream closed")
        data.extend(chunk)
    return bytes(data)


def read_camera_frame(stream: socket.socket, rotate: int) -> np.ndarray:
    size = struct.unpack("<I", read_exact(stream, 4))[0]
    if size != FRAME_BYTES:
        raise ValueError(f"expected a 640x360 UYVY frame ({FRAME_BYTES} bytes), got {size}")
    uyvy = np.frombuffer(read_exact(stream, size), dtype=np.uint8).reshape(
        CAMERA_HEIGHT, CAMERA_WIDTH, 2
    )
    bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if rotate == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rotate == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def send_move(stream: socket.socket, vx: float, vyaw: float) -> None:
    message = {
        "jsonrpc": "2.0",
        "method": "robot.move",
        "params": {"vx": vx, "vy": 0.0, "vyaw": vyaw},
    }
    stream.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())

~~~

第 45 行定义“读满指定字节数”的函数，因为一次 `recv` 不保证返回整帧。第 46 行创建接收缓存；第 47 行一直读到长度够；第 48–50 行如果连接关闭就报错；第 51 行累计字节；第 52 行返回不可变 bytes。

第 55 行定义读一帧图像。第 56 行先读 4 字节小端整数，得到帧长度；第 57–58 行要求长度正好匹配 UYVY 一帧。第 59–61 行把字节转为 `uint8` 数组并整理成 `[360, 640, 2]`。第 62 行将 UYVY 转成 OpenCV 使用的 BGR。第 63–68 行按用户选择旋转 90、180 或 270 度；第 69 行返回图像。

第 72 行定义发移动指令。第 73–77 行构造 JSON-RPC 的 `robot.move` 请求，前进量是 `vx`，横移固定为 0，转向是 `vyaw`。第 78 行把 JSON 转成紧凑文本，加换行后发送；第 79 行结束函数。

### 第 81–105 行：让相机先看向人物躯干

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
def send_look(stream: socket.socket, height: float) -> dict:
    """Aim the head camera at a forward point around the person's torso."""
    message = {
        "jsonrpc": "2.0",
        "id": "vision-follow-look",
        "method": "robot.look",
        "params": {"x": 1.0, "y": 0.0, "z": height},
    }
    previous_timeout = stream.gettimeout()
    stream.settimeout(3.0)
    try:
        stream.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())
        response = bytearray()
        while b"\n" not in response:
            chunk = stream.recv(4096)
            if not chunk:
                raise ConnectionError("robotd closed while setting the camera gaze")
            response.extend(chunk)
        result = json.loads(bytes(response).split(b"\n", 1)[0])
        if "error" in result:
            raise RuntimeError(f"robot.look failed: {result['error']}")
        return result.get("result", {})
    finally:
        stream.settimeout(previous_timeout)

~~~

第 81–88 行构造 `robot.look` 请求，目标点在机器前方 x=1、横向 y=0、高度由参数决定。第 89–90 行暂时把 socket 超时设为 3 秒。第 91–102 行发送请求、逐块读到 JSON-RPC 响应完整的一行、解析结果；如果 socket 关闭或服务返回 error，就抛出异常。第 103–104 行无论成功或失败都会把原来的超时恢复。第 105 行结束。

### 第 107–232 行：命令行参数和参数范围保护

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use YOLO to find a named class in the simulated duck camera."
    )
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=7901)
    parser.add_argument("--robot-socket", type=Path, default=DEFAULT_ROBOT_SOCKET)
    parser.add_argument(
        "--look-height",
        type=float,
        default=0.15,
        help="aim above the trunk so a standing person's body stays in the camera frame",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Ultralytics model or weights path")
    parser.add_argument(
        "--target",
        nargs="+",
        default=["person"],
        metavar="CLASS",
        help='one or more exact model class names, for example --target person bottle "sports ball"',
    )
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--hz", type=float, default=5.0, help="inference and command refresh rate")
    parser.add_argument("--seconds", type=float, default=15.0, help="run-time limit; 0 follows until Ctrl-C")
    parser.add_argument("--drive", action="store_true", help="enable the simulation-only follow controller")
    parser.add_argument(
        "--record-csv",
        type=Path,
        help="record visible-target observations and rule-controller actions as training demonstrations",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="TorchScript high-level follow policy; target-loss search remains the existing safe state machine",
    )
    parser.add_argument("--max-forward", type=float, default=0.40, help="forward speed, capped at 0.40 m/s")
    parser.add_argument(
        "--policy-forward-scale",
        type=float,
        default=1.0,
        help="multiply learned forward output before the max-forward safety cap",
    )
    parser.add_argument("--max-yaw", type=float, default=0.95, help="turn command, capped at 0.95 rad/s")
    parser.add_argument("--min-yaw", type=float, default=0.40, help="minimum turn command outside center tolerance")
    parser.add_argument(
        "--center-tolerance",
        type=float,
        default=0.30,
        help="normalized image offset inside which the duck walks forward",
    )
    parser.add_argument(
        "--center-hysteresis",
        type=float,
        default=0.10,
        help="extra offset needed to leave the centered state, to avoid turn/walk chatter",
    )
    parser.add_argument(
        "--stop-area",
        type=float,
        default=0.09,
        help="hold forward motion when the target box covers this fraction of the image",
    )
    parser.add_argument(
        "--stop-height",
        type=float,
        default=0.58,
        help="hold forward motion when the target box reaches this fraction of image height",
    )
    parser.add_argument(
        "--search-delay",
        type=float,
        default=0.6,
        help="wait this long after losing the target before scanning, in seconds",
    )
    parser.add_argument(
        "--search-yaw",
        type=float,
        default=0.22,
        help="in-place scan speed, capped by --max-yaw, in rad/s",
    )
    parser.add_argument(
        "--search-leg-seconds",
        type=float,
        default=3.0,
        help="duration of each left/right scan sweep, in seconds",
    )
    parser.add_argument(
        "--search-pause-seconds",
        type=float,
        default=0.8,
        help="pause between scan sweeps, in seconds",
    )
    args = parser.parse_args()
    if args.policy_forward_scale <= 0.0:
        parser.error("--policy-forward-scale must be positive")
    if args.hz <= 0 or args.seconds < 0:
        parser.error("--hz must be positive and --seconds cannot be negative")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")
    if not 0.0 < args.center_tolerance < 1.0:
        parser.error("--center-tolerance must be in (0, 1)")
    if not 0.0 <= args.center_hysteresis < 1.0 - args.center_tolerance:
        parser.error("--center-hysteresis must be non-negative and leave room below 1")
    if not 0.0 < args.stop_area < 1.0:
        parser.error("--stop-area must be in (0, 1)")
    if not 0.0 < args.stop_height < 1.0:
        parser.error("--stop-height must be in (0, 1)")
    if args.max_forward < 0 or args.max_yaw < 0 or args.min_yaw < 0:
        parser.error("speed limits cannot be negative")
    if args.search_delay < 0 or args.search_yaw < 0 or args.search_pause_seconds < 0:
        parser.error("search delay, yaw, and pause cannot be negative")
    if args.search_leg_seconds <= 0:
        parser.error("--search-leg-seconds must be positive")
    if args.record_csv is not None and not args.drive:
        parser.error("--record-csv requires --drive so each sample has a teacher action")
    if args.policy is not None and not args.drive:
        parser.error("--policy requires --drive")
    if args.record_csv is not None and args.policy is not None:
        parser.error("record teacher demonstrations without --policy")
    args.max_forward = min(args.max_forward, 0.40)
    args.max_yaw = min(args.max_yaw, 0.95)
    args.min_yaw = min(args.min_yaw, args.max_yaw)
    args.search_yaw = min(args.search_yaw, args.max_yaw)
    return args
~~~

第 107–110 行定义参数解析函数和命令说明。第 111–113 行设置相机地址、端口和 robotd socket。第 114–119 行设置机头看向高度，默认 0.15 m。第 120 行设置 YOLO 模型；第 121–127 行允许选择一个或多个模型类别，默认 `person`。第 128 行是最低 YOLO 置信度；第 129 行是推理图片尺寸；第 130 行是画面旋转；第 131 行是推理和控制更新频率，默认 5 Hz；第 132 行是运行秒数，0 表示一直运行；第 133 行只有带 `--drive` 才会移动鸭子。

第 134–138 行的 `--record-csv` 用于记录训练示范，要求规则控制器给每一帧提供老师动作。第 139–143 行的 `--policy` 指定训练好的 TorchScript 学生模型。第 144 行设置前进命令上限；第 145–150 行为已学习模型的前进输出提供额外缩放，默认 1.0。第 151–152 行设置转向速度范围。

第 153–164 行设置目标居中阈值和滞回量。第 165–176 行设置近距离停止阈值：目标框面积或高度超过阈值就停止前进。第 177–181 行设置丢失后等待多久才开始搜索；第 183–199 行设置左右搜索速度、每次扫描时长和扫描间隔。

第 201 行真正解析参数。第 202–203 行要求策略速度缩放为正。第 204–221 行检查推理频率、运行时长、置信度、居中阈值、距离阈值、速度和搜索参数。第 222–223 行规定采集 CSV 必须同时开移动模式，否则没有老师动作；第 224–225 行规定加载策略也必须开移动；第 226–227 行禁止一边用学生策略控制、一边把学生动作误记成老师示范。第 228–231 行将速度再次截到机器人控制的上限内，并保证 min yaw、搜索 yaw 不超过 max yaw。第 232 行返回参数。

### 第 235–294 行：加载 YOLO、设备、策略、CSV 和 robotd

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
def main() -> int:
    args = parse_args()
    if args.model == str(DEFAULT_MODEL) and not DEFAULT_MODEL.exists():
        DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pretrained weights to {DEFAULT_MODEL}.", flush=True)
        with urlopen(DEFAULT_MODEL_URL, timeout=30) as response, DEFAULT_MODEL.open("wb") as weights:
            while chunk := response.read(1024 * 1024):
                weights.write(chunk)
    print(f"Loading {args.model}.", flush=True)
    model = YOLO(args.model)
    names = model.names
    class_ids = {
        class_id
        for class_id, name in names.items()
        if name.casefold() in {target.casefold() for target in args.target}
    }
    missing = [target for target in args.target if target.casefold() not in {n.casefold() for n in names.values()}]
    if missing:
        available = ", ".join(str(name) for name in names.values())
        raise SystemExit(f"unknown target class {missing}; model classes: {available}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    follow_policy = None
    if args.policy is not None:
        # Load TorchScript on CPU first; some PyTorch/MPS builds attempt to
        # materialize serialized scalar tensors as float64 during map_location.
        follow_policy = torch.jit.load(str(args.policy)).to(device).eval()
    print(
        f"Model ready · device {device} · targets {', '.join(args.target)} · "
        f"mode {'LEARNED FOLLOW' if follow_policy is not None else 'FOLLOW' if args.drive else 'DETECT ONLY'}",
        flush=True,
    )

    robot: socket.socket | None = None
    camera: socket.socket | None = None
    records = None
    writer = None
    if args.record_csv is not None:
        args.record_csv.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not args.record_csv.exists() or args.record_csv.stat().st_size == 0
        records = args.record_csv.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            records,
            fieldnames=("timestamp", *FEATURE_COLUMNS, "teacher_vx", "teacher_vyaw"),
        )
        if needs_header:
            writer.writeheader()
    if args.drive:
        if not args.robot_socket.exists():
            raise SystemExit(f"robot socket not found: {args.robot_socket}; start scripts/duck-sim first")
        robot = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        robot.connect(str(args.robot_socket))
        gaze = send_look(robot, args.look_height)
        head = gaze.get("head", {})
        print(
            f"Camera gaze set · head pitch {float(head.get('head_pitch', 0.0)):+.2f} rad",
            flush=True,
        )
        send_move(robot, 0.0, 0.0)

~~~

第 235 行是主程序。第 236 行解析参数。第 237–242 行仅在使用默认模型且缓存不存在时下载 YOLO 权重，并边读网络响应边写文件。第 243–244 行加载 YOLO。第 245 行取模型类别表。第 246–250 行把用户输入的类名映射到模型类别编号，忽略大小写。第 251–254 行如果类名不存在，就列出所有可用类后退出。

第 256 行优先选择苹果 MPS，否则 CPU。第 257 行先把 `follow_policy` 设为空，代表只用规则控制。第 258–261 行若提供 `--policy`，用 TorchScript 加载它，再迁移到 MPS/CPU 并切到评估模式。这里载入的是动作策略，不是 YOLO 检测权重。第 262–266 行打印模型、设备和当前模式。

第 268–271 行初始化 socket、CSV 文件和写入器引用。第 272–281 行若指定 CSV，就创建父目录；文件新建或为空时写表头；`DictWriter` 的字段顺序包含 timestamp、7 个特征和 2 个老师动作；第 281 行把列名写入文件。第 282–294 行仅在 `--drive` 时检查 robotd socket、连接机器人、调用 `robot.look` 抬头，并先发送一次零速度，避免刚连上就继承旧意图。

### 第 295–356 行：循环取图，YOLO 找人并计算输入特征

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
    try:
        camera = socket.create_connection((args.camera_host, args.camera_port), timeout=5.0)
        camera.settimeout(5.0)
        print(
            f"Camera {args.camera_host}:{args.camera_port} · {CAMERA_WIDTH}x{CAMERA_HEIGHT} UYVY · "
            f"rotate {args.rotate}° clockwise · press Ctrl-C to stop",
            flush=True,
        )
        start = last_inference = last_report = 0.0
        period = 1.0 / args.hz
        stop_reason = "time limit" if args.seconds > 0 else "Ctrl-C"
        centered = False
        holding_distance = False
        lost_since: float | None = None
        last_seen_bearing = 0.0
        previous_bearing: float | None = None
        previous_detection_at: float | None = None

        while True:
            frame = read_camera_frame(camera, args.rotate)
            now = time.monotonic()
            if start == 0.0:
                start = now
            if args.seconds > 0 and now - start >= args.seconds:
                break
            if now - last_inference < period:
                continue
            last_inference = now

            result = model.predict(
                source=frame,
                imgsz=args.image_size,
                conf=args.confidence,
                device=device,
                verbose=False,
            )[0]
            height, width = frame.shape[:2]
            found: list[tuple[float, float, float, float, float, float, int]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if class_id not in class_ids:
                        continue
                    x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
                    confidence = float(box.conf.item())
                    area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (width * height)
                    found.append((area, confidence, x0, y0, x1, y1, class_id))

            if found:
                area, confidence, x0, y0, x1, y1, class_id = max(found, key=lambda item: item[0])
                center_x = (x0 + x1) / 2.0
                bearing = 2.0 * center_x / width - 1.0  # negative = left; positive = right.
                height_fraction = max(0.0, y1 - y0) / height
                if previous_bearing is None or previous_detection_at is None:
                    bearing_rate = 0.0
                else:
                    elapsed = max(1e-3, now - previous_detection_at)
                    bearing_rate = max(-4.0, min(4.0, (bearing - previous_bearing) / elapsed))
                previous_bearing = bearing
                previous_detection_at = now
                lost_since = None
                last_seen_bearing = bearing
~~~

第 295 行开始主循环保护区。第 296 行连接相机帧服务；第 297 行设置读取超时；第 298–302 行打印图像规格和旋转角。第 303 行初始化计时器；第 304 行将 Hz 转为每次推理之间的秒数；第 305 行准备退出原因文本；第 306–311 行初始化控制状态：是否已居中、是否在保持距离、目标丢失开始时间、上次目标方位，以及计算方位变化率所需的历史数据。

第 313 行进入无限循环。第 314 行读取摄像头帧。第 315 行取单调时钟，避免系统时间校正影响间隔计算。第 316–317 行记录开始时刻。第 318–319 行若达到 `--seconds` 就结束。第 320–321 行如果推理间隔还没到，就继续取图但跳过 YOLO；第 322 行更新时间戳。

第 324–330 行对当前图像运行 YOLO，输入图像、尺寸、置信度和设备，取第一张结果。第 331 行获取图像宽高。第 332 行创建候选目标列表。第 333–341 行逐个看 YOLO 框：取类别编号，只留下用户指定的类别；读框坐标与置信度；计算框面积占全图的比例；将结果放入列表。

第 343–347 行如果找到了目标，就选择面积最大的那个，作为本帧跟随对象。第 345 行算水平中心点；第 346 行将像素中心映射到约 -1 到 +1 的 bearing；第 347 行将框高度也换成画面比例。第 348–352 行计算 bearing_rate：当前方位减上次方位，再除以两次检测时间间隔，并限幅在 ±4，避免一帧抖动产生异常大值。第 353–356 行保存当前历史值、清除丢失状态，并记住最后一次看到人的 bearing。

### 第 357–433 行：规则老师、记录示范、学生预测

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
                vx = vyaw = 0.0
                if args.drive:
                    # These are the controller's internal states before this observation. Giving
                    # them to the learner makes the rule teacher's hysteresis observable.
                    centered_before = centered
                    holding_before = holding_distance
                    # A running person's box changes shape from frame to frame. Keep the
                    # standoff state until the box is clearly smaller, instead of alternating
                    # between walking and stopping around one threshold.
                    close = area >= args.stop_area or height_fraction >= args.stop_height
                    resume = (
                        area >= args.stop_area * 0.75
                        or height_fraction >= args.stop_height * 0.85
                    )
                    if holding_distance:
                        holding_distance = resume
                    else:
                        holding_distance = close
                    if holding_distance:
                        centered = abs(bearing) <= args.center_tolerance
                        if not centered:
                            # Hold distance while continuing to keep the person in view.
                            turn_rate = max(args.min_yaw, args.max_yaw * min(1.0, abs(bearing)))
                            vyaw = -turn_rate * (1.0 if bearing > 0 else -1.0)
                    else:
                        # Enter the forward state near the centre, then tolerate modest box jitter
                        # until the error grows past the wider exit boundary.
                        if centered:
                            centered = abs(bearing) <= args.center_tolerance + args.center_hysteresis
                        else:
                            centered = abs(bearing) <= args.center_tolerance
                    if not holding_distance:
                        # Keep walking while steering toward the target. The walking
                        # policy already turns and translates together; stopping forward
                        # motion whenever the person is off-centre made it spin in place.
                        vx = args.max_forward
                        if not centered:
                            offset = min(1.0, abs(bearing))
                            # robot.move defines positive vyaw as turning left.
                            turn_rate = max(args.min_yaw, args.max_yaw * offset)
                            vyaw = -turn_rate * (1.0 if bearing > 0 else -1.0)
                    teacher_vx, teacher_vyaw = vx, vyaw
                    features = [
                        bearing,
                        area,
                        height_fraction,
                        confidence,
                        bearing_rate,
                        float(centered_before),
                        float(holding_before),
                    ]
                    if writer is not None:
                        writer.writerow(
                            {
                                "timestamp": f"{now:.6f}",
                                **dict(zip(FEATURE_COLUMNS, features)),
                                "teacher_vx": f"{teacher_vx:.6f}",
                                "teacher_vyaw": f"{teacher_vyaw:.6f}",
                            }
                        )
                        assert records is not None
                        records.flush()
                    if follow_policy is not None:
                        with torch.inference_mode():
                            prediction = follow_policy(
                                torch.tensor([features], dtype=torch.float32, device=device)
                            )[0]
                        scaled_vx = float(prediction[0].item()) * args.policy_forward_scale
                        vx = max(0.0, min(args.max_forward, scaled_vx))
                        vyaw = max(-args.max_yaw, min(args.max_yaw, float(prediction[1].item())))
                        # The range gate is a safety backstop while this first learned policy is
                        # only a behaviour-cloned baseline. It cannot command forward motion
                        # after the teacher's conservative visual standoff threshold is crossed.
                        if holding_distance:
                            vx = 0.0
                    assert robot is not None
                    send_move(robot, vx, vyaw)
~~~

第 357 行先把本帧移动量设为零。第 358 行只有启用移动时才计算控制。第 359–360 行说明下面先保存更新前的状态，给训练样本提供控制器的滞回上下文。

第 361–365 行保存 `centered` 和 `holding_distance` 的旧状态，并说明距离保持使用滞回，避免人体框轻微抖动导致走走停停。第 366 行只要面积或高度达到“靠近”阈值就判为 close。第 367–370 行计算较低的恢复阈值；目标必须缩小到恢复线以下，控制器才解除保持。第 371–374 行更新保持状态：原本在保持就看是否可以恢复；原本没保持则看是否已靠近。

第 375–380 行若正在保持距离，就不往前走；若人物偏离中心，就按偏差大小转向，并根据图像左右决定 yaw 正负。第 381–387 行否则更新居中状态：使用不同的进入/退出阈值形成滞回，避免临界点来回抖动。第 388–397 行若无需保持距离，就前进到 max-forward；目标未居中时同时转向。规则控制器就是这里的“老师”。

第 398 行将老师给出的动作暂存为 `teacher_vx` 和 `teacher_vyaw`。第 399–407 行按 `FEATURE_COLUMNS` 的顺序组装 7 个输入数字；注意状态特征用的是第 361–362 行保存的旧状态。第 408–418 行若开启了 CSV 记录，就将时间、特征和老师动作写成一行并立即 flush，避免进程中断时丢失全部缓冲。

因为只有 `if found:` 分支会写 CSV，目标不可见时不会生成训练行。因此这个模型没有学过“人丢了该怎样搜索”；第 448–471 行的停止和搜索仍由 Python 状态机负责。

第 419–426 行只有加载了学生策略才执行神经网络预测：`torch.inference_mode()` 关闭梯度；第 421–423 行把 7 个特征做成 `[1, 7]` 张量，得到两个预测动作；第 424 行按设置放大学生的前进速度；第 425 行把它截到 0 和 max-forward 之间；第 426 行把 yaw 截到 ±max-yaw。第 427–431 行保留硬安全门：进入距离保持后，无论学生网络预测什么，前进量都强制为零。第 432–433 行确认 robotd 已连接，再发送最终动作。

关键点：同一段代码既能“用规则控制并记老师答案”，也能“用已训练策略控制”。CSV 写入的是 `teacher_*`，而不是模型 `prediction`；参数检查也禁止二者同时启用，避免把学生的输出冒充老师标签。

### 第 435–493 行：日志、丢失搜索和安全退出

下面先贴这一段在源码中的原文，紧接着按源文件行号解释。

~~~python
                if now - last_report >= 0.5:
                    name = names[class_id]
                    action = f"vx={vx:.2f} vyaw={vyaw:.2f}" if args.drive else "detect only"
                    if follow_policy is not None:
                        action = f"learned {action}"
                    if args.drive and holding_distance:
                        action = f"holding distance · {action}"
                    print(
                        f"{name} conf={confidence:.2f} bearing={bearing:+.2f} "
                        f"box-area={area:.1%} box-height={height_fraction:.1%} · {action}",
                        flush=True,
                    )
                    last_report = now
            else:
                if args.drive:
                    centered = False
                    assert robot is not None
                    if lost_since is None:
                        lost_since = now

                    lost_for = now - lost_since
                    turn = 0.0
                    action = "lost · pausing"
                    if lost_for >= args.search_delay:
                        scan_time = lost_for - args.search_delay
                        cycle = args.search_leg_seconds + args.search_pause_seconds
                        leg = int(scan_time // cycle)
                        leg_time = scan_time - leg * cycle
                        if leg_time < args.search_leg_seconds:
                            # Positive image bearing means the target was to the right, so
                            # begin by turning right (negative robot yaw) to reacquire it.
                            first_direction = -1.0 if last_seen_bearing > 0 else 1.0
                            turn = first_direction * (-1.0 if leg % 2 else 1.0) * args.search_yaw
                            action = f"searching {'left' if turn > 0 else 'right'}"
                        else:
                            action = "search sweep pause"
                    send_move(robot, 0.0, turn)
                if now - last_report >= 0.5:
                    message = f"target not detected · {action}" if args.drive else "target not detected"
                    print(message, flush=True)
                    last_report = now

        if args.drive:
            print(f"Stopping: {stop_reason}.", flush=True)
    except KeyboardInterrupt:
        print("\nStopping: Ctrl-C.", flush=True)
    finally:
        if robot is not None:
            send_move(robot, 0.0, 0.0)
            robot.close()
        if camera is not None:
            camera.close()
        if records is not None:
            records.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
~~~

第 435–447 行每 0.5 秒打印一次类别、置信度、bearing、框面积、框高度和动作；第 438–439 行若动作来自模型，就在日志中标注 learned；第 440–441 行若距离门生效，就打印 holding distance。

第 448–471 行处理本帧没有目标。第 449–451 行在跟随模式把居中状态清零并确认 robot socket 存在。第 452–454 行第一次丢失时记录时刻。第 455 行计算丢失持续时间。第 456–457 行默认不前进，只暂停。第 458–470 行等超过 search delay 后左右交替扫描：按丢失时长计算当前扫描轮次和扫描段；一侧转动若干秒，另一侧反向；间隔期原地暂停。第 471 行发送“前进 0，只转向”的搜索指令，避免看不到人时盲目前进。

第 472–475 行定期打印未检测到目标及当前搜索阶段。第 477–478 行正常结束时打印原因。第 479–480 行捕获 Ctrl-C 并打印停止说明。第 481–488 行的 `finally` 无论正常结束还是异常都会执行：先发零速度，再关闭 robot、camera 和 CSV 文件。第 489 行返回成功状态；第 492–493 行保证直接运行脚本时调用 `main()` 并向操作系统返回退出码。

## 6. 这次模型的真实训练结果怎么读

当前 `follow-policy.ts.json` 记录：353 条总样本，282 条训练、71 条验证；前进 MAE 为 0.0132 m/s，转向 MAE 为 0.0555 rad/s。MAE 是预测与老师动作的平均绝对差，例如 yaw 误差 0.0555 rad/s 大约是 3.2°/s。CSV 中 `teacher_vx` 最大值为 0.30 m/s，说明这批示范来自前进上限 0.30 m/s 的旧控制配置；现在 `scripts/duck-sim follow` 使用 0.40 m/s 上限、不同的距离阈值，并对策略前进输出乘 1.25。因此当前缓存模型与现行规则的训练设置并不完全相同，重新采集时应使用现行参数。元数据没有保存规则阈值，单靠 JSON 看不出这项差异。

这些数字说明模型在抽出来的验证行上能模仿老师，不说明鸭子真的更会跟人。JSON 的 `seed: 17` 是训练程序的随机种子；`seed-17.csv` 只是数据文件名，两者不是人物路线或场景种子的证明。

验证集是随机抽行，而不是按整段运行或不同路线留出。相邻视频帧高度相似，一个场景的相邻行可能同时出现在训练集和验证集，所以这个误差可能显得过于乐观。样本只有 353 条，也没有证明它能适应陌生人物、遮挡、不同光照或新路线。推理时使用的 YOLO 仍是预训练检测器，训练这个跟随策略没有更新 YOLO。

## 7. 行为克隆和 PPO 的区别

当前训练没有 reward、episode、critic、优势估计或环境交互更新。训练程序只读已有 CSV，最小化“网络动作和老师动作的差距”，这叫监督学习/行为克隆。

PPO 则要把策略放进仿真里运行，记录动作后环境如何变化，例如人物是否一直在视野、距离是否合适、鸭子是否摔倒；根据这些成败信号形成 reward，再用 rollout 和策略优化更新网络。现在这三个文件没有实现这条 PPO 流程。因此更准确地说：YOLO 负责检测，行为克隆策略负责模仿规则跟随器，鸭子的步态仍由 robotd 已有的运动策略执行。
