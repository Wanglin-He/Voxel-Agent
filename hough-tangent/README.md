# hough-tangent

对视频里的圆盘阵列做霍夫圆检测，用**外公切线**把主体外围框成一个尖角多边形，最外圈的每个圆盘都被切到。

算法怎么设计的、每一步在解决什么问题，都写在 [hough_tangent/README.md](hough_tangent/README.md)。这份只讲怎么跑。

---

## 最快的用法

环境已经装好了（`.venv/` 里是 Python 3.9 + OpenCV + NumPy），直接：

```bash
cd ~/Projects/Voxel-Agent/hough-tangent
./run.sh ~/Desktop/你的视频.mov
```

输出会落在 `out/你的视频_tangent.mp4`。想指定输出路径就写第二个参数：

```bash
./run.sh ~/Desktop/你的视频.mov ~/Desktop/结果.mp4
```

`run.sh` 会自己检查虚拟环境，缺了就重建，所以整个文件夹拷到别的电脑上也能跑。

## 在 VS Code 里

1. `File → Open Folder` 打开 `~/Projects/Voxel-Agent`
2. 右下角选 Python 解释器，挑 `hough-tangent/.venv/bin/python`（仓库根目录的 `.vscode/settings.json` 里已经指好了，通常会自动选上）
3. 打开终端（`` Ctrl+` ``），`cd hough-tangent` 之后跑上面的命令

不想用 `run.sh` 也可以直接调模块：

```bash
./.venv/bin/python -m hough_tangent 输入.mov -o 输出.mp4
```

## 先自检一下

不需要视频，纯几何验证，两秒跑完：

```bash
./.venv/bin/python -m hough_tangent.selftest
```

13 项全 ok 就说明装好了。

---

## 换一段新视频要动什么

**只是同一块板子重拍**——什么都不用改，直接跑。程序会先把整段完整过一遍，统计出哪些格位真的有盘，再据此渲染。

**一次跑多段**：

```bash
./batch.sh ~/Desktop/结果文件夹 ~/Desktop/a.mov ~/Desktop/b.mov ~/Desktop/c.mov
```

挨个渲染进那个文件夹。缺文件会先报出来再退出，不会跑到一半才发现。额外参数放 `--` 后面：
`./batch.sh 输出 a.mov b.mov -- --stride 30`

**阵列不是 4×4**：

```bash
./run.sh 视频.mov --grid 5x3
```

**已经知道哪些格位是空的**（覆盖程序自己的判断）。IMG_0947 那块板子是：

```bash
./run.sh 视频.mov --cells 1111,1011,1101,1111
```

一行一排，`1` 有盘 `0` 空。

**被测物体根本没有栅格结构**：

```bash
./run.sh 视频.mov --no-grid
```

这时只剩霍夫圆 + 圆盘校验 + 切线包络，抖动会明显一些，因为没有先验来补漏检。

**物体不是白色的**：改 `hough_tangent/validate.py` 里的 `DiscProfile` —— `bright_on_dark`（亮底还是暗底）、`max_saturation`（颜色有多鲜艳）、`min_contrast`（内外亮度差）。

## 调外观

| 参数 | 默认 | 作用 |
|---|---|---|
| `--outline ring\|hull` | `ring` | `ring` 沿实际占用区域的边界逐个外切，缺口会凹进去；`hull` 是凸包，只切到够得着边的那几个 |
| `--corners sharp\|round` | `sharp` | 拐角走切线交点（尖角）还是沿圆弧 |
| `--margin-frac F` | `0` | 框往外扩，单位是盘半径的倍数。盘倾斜时想把圆柱侧壁也圈进去就调这个，`0.1` 左右 |
| `--tangent-extend PX` | `0` | 把切线延长出切点若干像素，视觉上强调"这是切线" |
| `--no-holes` | 关 | 板子中间被围住的空格位默认会单独画一圈内轮廓；加这个就不画，空格留在外框里 |
| `--no-circles` | 关 | 只画框，不画每个圆 |
| `--no-hud` | 关 | 去掉左上角的帧号和检出计数 |
| `--hull-colour B,G,R` | 青色 | 框的颜色，注意是 BGR 不是 RGB |
| `--out-width N` | `1080` | 输出分辨率宽度 |

## 调参时省时间

整段 4K 跑一次约 5 分钟。试参数的时候先抽帧：

```bash
./run.sh 视频.mov out/试.mp4 --stride 30            # 每 30 帧取一帧
./run.sh 视频.mov out/试.mp4 --start 900 --end 1200 # 只跑中间这段
```

**检测结果不对**的时候加 `--debug`，会用灰色把校验前的**全部**霍夫候选画出来。看一眼就知道是霍夫压根没找到（该放宽 `--r-min-frac` / `--r-max-frac`），还是找到了但被校验筛掉了（该松 `DiscProfile`）。

## 出问题

**`需要 Python 3.9 或更高版本`** —— 装个新的，然后 `PYTHON=/opt/homebrew/bin/python3.12 ./run.sh ...`

**`could not open ... for writing with fourcc 'avc1'`** —— 这台机器的 OpenCV 没带 H.264 编码器，加 `--fourcc mp4v`。

**框跳来跳去** —— `--min-cutoff 0.5` 让它更稳（代价是跟随慢半拍），`--beta 0.05` 让它跟得更紧（代价是更抖）。

**框歪了一整列** —— 说明栅格锁错了位置。用 `--cells` 把占用情况写死，这是最直接的办法。
