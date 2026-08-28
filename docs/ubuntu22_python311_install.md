# Ubuntu 22.04 安装 Python 3.11

我已经把一组可直接安装的 `amd64` 包下载到了：

```text
C:\Users\Administrator\Desktop\python311_ubuntu22_debs
```

包含：

- `libpython3.11-minimal_3.11.0~rc1-1~22.04_amd64.deb`
- `python3.11-minimal_3.11.0~rc1-1~22.04_amd64.deb`
- `libpython3.11-stdlib_3.11.0~rc1-1~22.04_amd64.deb`
- `libpython3.11_3.11.0~rc1-1~22.04_amd64.deb`
- `python3.11_3.11.0~rc1-1~22.04_amd64.deb`
- `python3.11-venv_3.11.0~rc1-1~22.04_amd64.deb`

这些包来自 Ubuntu `jammy-updates` 的 `python3.11` 包集。

## 服务器安装

把整个目录上传到 Ubuntu 22.04 服务器后，执行：

```bash
cd /path/to/python311_ubuntu22_debs
sudo apt install ./*.deb
```

装完检查：

```bash
python3.11 --version
```

## 然后怎么用本项目

你现在不要 `.venv` 的话，直接：

```bash
cd /path/to/pi05_jax_sft
USE_VENV=0 INSTALL_MODE=mirror PYTHON_BIN=python3.11 bash scripts/setup_env.sh
```

如果你的镜像里已经配好了内网 pip 源，这样就会直接装到当前环境。

## 一个需要注意的点

不用 `.venv` 不代表 `pip` 会自动切到 `python3.11`。之后安装和运行都建议显式用：

```bash
python3.11 -m pip ...
python3.11 -m pi05_jax_sft.train --config ...
```

如果 `python3.11 -m pip` 不存在，再补一次：

```bash
python3.11 -m ensurepip --upgrade
```

如果系统禁用了 `ensurepip`，那就继续用你的内网源给 `python3.11` 补 pip，或者告诉我你们镜像里现在 `pip3` 和 `python3` 的绑定关系，我再给你一版不踩默认解释器的命令。
