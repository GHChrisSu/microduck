# 从仿真鸭到会跟随的机器人

这是 MicroDuck 工程学习手册的源文件与成品目录。整本书按章节编写，源文件使用 Quarto Markdown（QMD）。

- 手机/阅读器：打开 _book/microduck-learning.epub
- 电脑网页：打开 _book/index.html
- 封面源文件：images/cover.svg
- Skill 与出版工具比较：production-comparison.qmd

## 重新生成

安装 Quarto CLI 后，在 microduck 仓库根目录运行：

~~~sh
quarto render docs/books/microduck-handbook
~~~

渲染结果写入 _book/。修改章节时编辑 index.qmd、chapters/ 下的章节，封面改动后用 SVG 工具重新导出 images/cover.png。

## 阅读顺序

先读运行时总览和 robotd 控制循环，接着了解 IPC、传感器与更新；再读强化学习环境、MDP 和模型导出；最后通过仿真、YOLO 检测、人物跟随、调试实验和应用场景畅想把整条闭环串起来。
