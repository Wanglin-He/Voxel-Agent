# 给 Claude 的项目说明

对视频里的圆盘阵列做霍夫圆检测，再用外公切线把主体外围框成尖角多边形。

设计文档在 `hough_tangent/README.md`，**动代码之前先读它**——里面记着几个踩过的坑，不读会重新踩一遍。

## 环境

`.venv/` 是 Python 3.9 + OpenCV + NumPy。用 `./.venv/bin/python`，不要用系统 python。
改完代码跑 `./.venv/bin/python -m hough_tangent.selftest`（13 项，纯几何，不需要视频）。

**代码必须保持 Python 3.9 可运行**：不要用 `zip(strict=)`、`match` 语句、`itertools.pairwise`、
`dataclass(slots=True)`。类型标注随便写，因为每个模块都有 `from __future__ import annotations`。

## 几条不要破坏的约定

**栅格是软先验。** 它只用来定归属、剔离群、补漏检，**不去移动已经测到的圆心**。这个阵列在
视频里会实际形变，把观测值吸附到刚性模型上画面会很稳，但那是假的。`--snap` 是留给明确想要
这个效果的人的，默认必须是 0。

**空格位要靠观测得出，不能写死在代码里。** 4×4 的栅格不等于 16 个盘——当前这块板子 (1,1) 和
(2,2) 是空的。预扫描统计占用率来判断，或者用户用 `--cells` 明确指定。

**切线是闭式解，不是多边形逼近。** `hull.py` 用支撑函数上包络求，断点解析求解。自检里验证到
1e-12。任何"用 64 边形近似圆再求凸包"的改法都是退步。

**不要为了画面好看去调阈值。** 检出率这类数字是从 `--debug` 和自检里读出来的，不是调出来的。
改了参数要说清楚是哪个真实失败案例逼出来的。

## 常用命令

```bash
./run.sh 视频.mov                                   # 跑全段
./run.sh 视频.mov out/试.mp4 --stride 30 --debug     # 抽帧 + 画出全部霍夫候选
./.venv/bin/python -m hough_tangent.selftest        # 自检
```
