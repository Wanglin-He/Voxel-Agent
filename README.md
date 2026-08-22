# Voxel-Agent

For voxel robot.

## 目录

### [`hough-tangent/`](hough-tangent/)

对视频里的圆盘阵列做霍夫圆检测，用**外公切线**把主体外围框成一个尖角多边形，最外圈的每个圆盘都被切到。

```bash
cd hough-tangent
./run.sh ~/Desktop/你的视频.mov
```

首次运行会自动建虚拟环境装依赖。用法、调参、换新拍摄对象要动什么，见
[hough-tangent/README.md](hough-tangent/README.md)；算法怎么设计的、每一步在解决什么问题，
见 [hough-tangent/hough_tangent/README.md](hough-tangent/hough_tangent/README.md)。

在这段 4×4 电机阵列素材上的实测：1830 帧全程锁定，每帧 14/14 个圆盘全部真实检出，
零帧靠外推撑着；轮廓与外圈 12 个圆盘的最大缝隙 0.93px，从未切进任何圆盘。

---

## 本机注意事项：Homebrew 的 git 推不动代码

`/opt/homebrew/bin/git`（2.55.0）在这台机器上跑任何 https 操作都会崩：

```
dyld: Symbol not found: _curl_global_trace
  Referenced from: .../libexec/git-core/git-remote-http
  Expected in:     /usr/lib/libcurl.4.dylib
错误：git-remote-https died of signal 6
```

原因是它硬编码去加载**系统**的 `/usr/lib/libcurl.4.dylib`，而系统那个不导出
`curl_global_trace`（`nm -gU /usr/lib/libcurl.4.dylib | grep -c curl_global_trace` 结果是 0）。
机器上虽然装了 Homebrew curl 8.21.0，但二进制里记的路径是系统那条，装了也用不上。

**能用的办法**（已验证）——用 Apple 自带的 git：

```bash
/usr/bin/git push
```

`.vscode/settings.json` 里已经把 `git.path` 指向它了，所以 VS Code 里的 git 面板正常工作。

**想彻底修好**：让 Homebrew git 重新编译并链到 brew 自己的 curl。

```bash
brew reinstall --build-from-source git
```

这个我没跑（要编译，比较久），成不成得你自己试。不修也完全不影响用，`/usr/bin/git` 够了。
