# 语音唤醒、寻找主人与跟随

本实验把主机麦克风、语音唤醒、鸭子相机里的 YOLO 人物检测、动作控制和 Sub2API Responses API Agent 连起来。Mac 上的 Whisper 在本机识别唤醒词和随后的命令，只有转写文字会发给 Agent，命令音频不离开本机。

## 启动场景

在 `microduck` 仓库根目录运行：

```sh
export DUCK_SIM_RL=/Users/suhao/github/robotic_ai/microduck_rl
export DUCK_SIM_AGENT_MODEL=gpt-6-luna
scripts/duck-sim find
```

首次使用时，脚本会在 `~/.cache/duck-sim/speech/` 下载约 181 MiB 的 Whisper 模型，用于本机唤醒词和命令识别。随后仿真会启动人物、鸭子相机和视觉控制器；人物初始位置在鸭子正后方约 3.8 米。为降低 Mac 上的窗口渲染负载，这个场景默认使用 15 fps 的相机流与 MuJoCo 可视刷新，媒体配置也同步为 360p15；YOLO 控制器以 5 Hz 检测，机器人控制仍为 50 Hz。设置 `DUCK_SIM_CAMERA_FPS=30` 和 `DUCK_SIM_VIEWER_FPS=30` 可恢复较高刷新率。窗口左上角显示麦克风状态，终端会显示唤醒词、转写文本和 Agent 选择的 Tool。当前 Sub2API 账号需选 `/models` 返回的模型；本机验证 `gpt-5.5` 不可用，`gpt-6-luna` 可用。

| 按键 | 行为 |
|---|---|
| `↑` / `↓` | 人物沿世界坐标 `+X` / `-X` 移动 |
| `←` / `→` | 人物沿世界坐标 `+Y` / `-Y` 移动 |
| `Home` | 人物回到鸭子正后方的初始位置 |
| `Enter` | 静音麦克风；再次按下恢复监听 |
| `Ctrl-C` | 退出语音跟随控制器 |

鸭子正前方是世界 `+X`，因此正后方是 `-X`。方向键可以把人物带到任意位置；每次按键改变一个行走目标，人物会平滑跑向该位置。

## 说什么、鸭子做什么

Mac 上的 Whisper 会持续监听。可以先单独叫名字，再在 8 秒内说命令；也可以一次说完：

| 说法 | 意图和行为 |
|---|---|
| `小鸭` | 只唤醒，鸭子保持不动，等待下一句命令 |
| `小鸭，过来` | 寻找主人；相机看到人后靠近，并在合适距离停下 |
| `小鸭，跟随我` | 寻找并持续跟随，在视觉范围内保持距离 |
| `小鸭，停止` | 停止移动并回到待机 |

程序还接受 `小鸭子` 等常见唤醒词变体。只叫名字不会让鸭子移动；之后 8 秒内本机 Whisper 识别到的下一句会作为命令交给 Agent。一次说完“小鸭过来”时，本机 Whisper 识别整句并交给 Agent。“停止”“停下”等安全词在本机直接生效，不等待云端。完成一条命令后，程序回到唤醒监听状态。

运行 `scripts/duck-sim find` 的终端会显示 Agent 选择的 Tool 和随后的动作，例如说“小鸭过来”：

```text
Sending post-wake transcript to the Responses API.
Agent tool=approach_owner.
Intent approach · starting visual owner search.
```

为保护隐私，转写原文不会打印到终端。

## 鸭子如何从背后找到人

语音识别只决定何时启动、启动哪种行为；它不会给鸭子人物的仿真坐标，也不会从声音里推断声源方向。目标搜索和接近都依赖鸭子自己的相机和 YOLO：

1. 听到唤醒词后，鸭子进入搜索；头部相机先向两侧和前方扫描，身体暂时不动。
2. 头部扫描仍找不到人时，头回到正中，鸭子以低速向前走并转身搜索。这个走路策略在纯原地转向时几乎不转，所以搜索会用 `0.25 m/s` 前进配合 `0.95 rad/s` 转向；半圈的前进轨迹很小，能让身体真正转起来。
3. 相机检测到 `person` 后，鸭子先把相机回正，再转向人物并走近。
4. 鸭子靠近适当距离后如果人物偏在画面一侧，会低速前进并转动身体来重新对准人物；否则该步态在原地转向时不会明显转身。
5. 人物离开视野后，鸭子会继续按“头部扫描、身体转身”的步骤寻找。
6. 说“停止”会让鸭子停下并回到待机；再次说“小鸭”可重新开始。

人物从正后方开始，鸭子一开始看不到人。因此能否找到人，直接检验身体转身搜索是否真实发生，而不是只测试头部摇动。

## 麦克风和模型

电脑的真实麦克风作为主机音频输入；MuJoCo 本身没有房间声学或虚拟麦克风。`whisper.cpp` 在本机按语音活动切分并转写唤醒词和命令片段，临时 WAV 文件会在使用后删除。Sub2API 仅收到转写文字，不接收音频。

macOS 使用 FFmpeg 的 AVFoundation 输入，并优先选择实体麦克风、跳过 BlackHole 或 Loopback 等回环设备。首次使用时，macOS 可能要求给启动仿真的终端麦克风权限。若状态没有显示 `Mic: LISTENING`，看终端中的错误；也可列出输入设备：

```sh
ffmpeg -f avfoundation -list_devices true -i ""
```

再指定设备或调低说话音量门限：

```sh
DUCK_SIM_AUDIO_DEVICE=1 DUCK_SIM_VOICE_THRESHOLD=0.015 scripts/duck-sim find
```

按 `Enter` 后状态应显示 `Mic: MUTED`；再按一次应恢复为 `Mic: LISTENING`。如果点错窗口，先点击 MuJoCo 窗口获得键盘焦点。

## 当前范围与下一步

当前仿真已接入本地 Whisper 语音识别和 Sub2API Responses API Agent。Mac 本机真实唤醒词、背后找人、Sub2API 函数调用和人物靠近动作分别已验证。语音识别运行在这台 Mac 上；实体鸭子仍需接入板端录音和语音识别。

## 代码阅读路径

- `microduck/scripts/duck-sim`：准备人物正后方场景、检查 Whisper 模型并启动控制器。
- `microduck/scripts/duck_vision_follow.py`：本地 Whisper 语音识别、Responses API Agent、YOLO 目标检测、搜索状态机和移动控制。
- `microduck_rl/src/mjlab_microduck/sim/body_server.py`：人物初始坐标、方向键移动、麦克风状态提示，以及 MuJoCo 世界步进。
- `microduck_rl/src/mjlab_microduck/robot/microduck/scene_approach.xml`：人物模型和鸭子相机的仿真场景。
