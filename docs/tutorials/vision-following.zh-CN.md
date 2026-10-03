# 视觉识别后跟随目标：第一条闭环

跟随模型的特征、CSV 样本、训练循环与每个脚本的逐段讲解见[跟随策略代码导读](follow-policy-code-tour.zh-CN.md)。

这个练习把“看到目标”和“朝目标移动”连起来：

```text
仿真相机帧 → YOLO 检测框 → 选定类别 → 算水平偏差 → 转向或前进 → 下一帧再看
```

这里用 MuJoCo 场景中的全尺寸人形模型做目标，高约 1.73 米。它从鸭子前方约 6.5 米处出发，持续沿镜头正前方直线奔跑，不设活动边界，也不会绕圈或停下来转向。模型用几何体和摆动的四肢组成，不是照片级的人体；检测来自通用 YOLO 模型的 `person` 类。脚本默认只检测，添加 `--drive` 后才会向本机的仿真 `robotd` 发送移动意图。

## 1. 准备 Python 依赖

在 microduck 仓库根目录执行。依赖装进 `microduck_rl` 已有的虚拟环境，不会改它的训练配置或 `uv.lock`：

```sh
export DUCK_SIM_RL=/Users/suhao/github/robotic_ai/microduck_rl
uv pip install --python "$DUCK_SIM_RL/.venv/bin/python" -r scripts/vision-requirements.txt
```

第一次运行会下载约 5 MB 的 YOLO 权重，保存在 `~/.cache/duck-sim/vision/`；Mac 上会优先使用 MPS。权重不放进 Git。

## 2. 启动带相机和移动人形目标的仿真

若仿真尚未启动，在另一个终端执行：

```sh
export DUCK_SIM_RL=/Users/suhao/github/robotic_ai/microduck_rl
export DUCK_SIM_SCENE=approach
export DUCK_SIM_CAMERAS=a
export DUCK_SIM_KEYFRAME=HOME
export DUCK_SIM_MOVE_TARGET=follow_person
scripts/duck-sim
```

相机帧由 `duck-body` 从 TCP 7901 端口发送，控制意图通过 `~/.cache/duck-sim/duck-a.sock` 到达现有的 `robotd`。相机画面也可在 <http://127.0.0.1:8080> 查看。

## 3. 先只检测

```sh
"$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py --target person
```

屏幕会周期性打印类别、置信度和 `bearing`。`bearing` 是目标框中心相对画面中心的归一化偏差：负数在左边，正数在右边，接近 0 表示居中。YOLO 的 COCO 预训练模型有 80 个常见类别；这里默认选择 `person` 来识别仿真里跑动和跳跃的人形目标。

YOLO 会输出 `person` 类的检测框。指定其它单个类别，例如：

```sh
"$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py --target bottle
```

也可以同时指定多个类别，例如 `--target person bottle`。类别名必须存在于所选模型中。

## 4. 限时跟随

确认日志检测到了目标后，再运行：

```sh
"$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py --target person --drive --seconds 30 --stop-area 0.04 --stop-height 0.40
```

脚本根据目标框偏差同时调整前进和转向：目标居中时直行，目标偏离时一边转向一边以前进速度继续走，避免只在原地转。跟随演示中，人体框高度占画面 45% 或面积占 6% 时暂停前进；只有高度降到约 38% 以下、面积也低于约 4.5% 才继续靠近。目标丢失后先停 0.6 秒，再按上次看到目标的方向开始原地左右扫描；每次扫描 2.5 秒，中间短暂停顿，并持续重复直到重新识别到目标。扫描时前进速度为 0，避免盲目前进。重新识别后立即恢复跟随；时间到或按 Ctrl-C 退出时也发送零速度。`robotd` 自己还有 500 ms deadman，停止收到新意图后会清零速度。

只运行 `scripts/duck-sim` 会启动仿真，但不会启动 YOLO 跟随程序。要一条命令启动完整演示，在 microduck 仓库根目录运行 `scripts/duck-sim follow`；它会重启到人物场景并持续跟随，按 Ctrl-C 停止。

`scripts/duck-sim follow` 会自动使用 `~/.cache/duck-sim/follow-training/follow-policy.ts` 中的已训练跟随策略（若文件存在），并把其前进输出放大 1.25 倍、限制在 0.40 m/s 内；找不到策略时会使用规则控制器。人物从约 6.5 米外以默认 0.10 m/s 持续向前跑，鸭子的前进命令最高为 0.40 m/s。实测鸭子收到前进命令时的位移速度约为 0.11–0.15 m/s，转向时会更低，因此人物默认速度调低以便鸭子能跟上并保持视距。这里的速度是仿真控制速度，不是对真实儿童冲刺速度的复现；可用 `DUCK_SIM_TARGET_MOTION_SPEED` 调整人物速度，若设得高于鸭子的实际速度，目标最终会离开视野。更改后重启跟随演示：

```sh
scripts/duck-sim down
export DUCK_SIM_TARGET_MOTION_SPEED=0.10
scripts/duck-sim follow
```

缩短跟随练习时间的命令如下：

```sh
"$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py \
  --target person --drive --seconds 8 --max-forward 0.40
```

## 5. 它识别很多物体吗？

YOLO26n 的 COCO 预训练模型有 80 类，例如 person、bicycle、car、dog、cat、bottle、chair 和 sports ball。对已训练的类别，换 `--target` 就可以选择跟随对象。

如果目标不在模型类别里，或者是特定的物品、颜色/外观、你自己的玩具或鸭子，需要采集相机视角的数据并标注目标框，然后用预训练模型微调，再把输出模型路径传给 `--model /path/to/weights.pt`。摄像头角度、距离和光照要尽量覆盖真实使用情况。识别不等于测距：本练习只用目标框大小作粗略“接近了”停止条件；要稳定控制距离，应再融合 ToF 或深度。

当前工程的 Rust `duck-detect` 是单类“鸭子”检测器，不是通用 YOLO 前端。此脚本作为 Mac 仿真教学原型单独运行；后续若要让真实机载行为复用多类别结果，需要扩展检测输出协议，并在机器人端适配硬件推理后端。
