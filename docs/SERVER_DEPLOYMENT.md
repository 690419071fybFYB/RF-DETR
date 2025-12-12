# 🚀 服务器部署教程

> 本教程帮助你在新租用的服务器上快速部署 RF-DETR 项目环境

## 📋 前置信息

在开始之前，请填写以下信息：

```bash
# 服务器信息（替换为你的实际信息）
SERVER_IP="117.50.186.145"
SERVER_PORT="23"  # SSH 端口，默认 22
SERVER_USER="root"
SERVER_PASSWORD="你的密码"

# 本地项目路径
LOCAL_PROJECT="/home/fyb/mydir/rf-detr"
LOCAL_DATASET="/home/fyb/datasets/RSOD_cocoFormat"
LOCAL_WEIGHTS="/home/fyb/mydir/rf-detr/rf-detr-base.pth"

# 远程目标路径
REMOTE_PROJECT="/root/RF-DETR"
REMOTE_DATASET="/root/RSOD_cocoFormat"
```

---

## 🔧 第一部分：本地准备工作

### 1.1 导出环境依赖

```bash
# 使用 uv 导出依赖（推荐）
cd /home/fyb/mydir/rf-detr
uv pip freeze > requirements.txt

# 或使用 pip
pip freeze > requirements.txt
```

### 1.2 提交代码到 GitHub

```bash
cd /home/fyb/mydir/rf-detr
git add .
git commit -m "准备服务器部署"
git push origin $(git branch --show-current)
```

---

## 📤 第二部分：文件传输

### 2.1 传输数据集（多个小文件）

```bash
# 数据集传输（scp 对多个小文件效果好）
scp -r -P 23 /home/fyb/datasets/RSOD_cocoFormat root@117.50.186.145:/root/
```

### 2.2 传输大文件（如预训练权重）

由于大文件容易断连，推荐分块传输：

```bash
# 分块（每块 50MB）
split -b 50M /home/fyb/mydir/rf-detr/rf-detr-base.pth /tmp/rf-detr-base.pth.part_

# 传输分块
scp -P 23 /tmp/rf-detr-base.pth.part_* root@117.50.186.145:/root/RF-DETR/

# 在服务器上合并（SSH 登录后执行）
cat /root/RF-DETR/rf-detr-base.pth.part_* > /root/RF-DETR/rf-detr-base.pth
rm /root/RF-DETR/rf-detr-base.pth.part_*
```

### 2.3 或者直接在服务器下载权重

```bash
# 在服务器上执行
cd /root/RF-DETR
wget https://huggingface.co/rafaelpadilla/RF-DETR/resolve/main/rf-detr-base.pth -O rf-detr-base.pth
```

---

## 🖥️ 第三部分：服务器环境配置

### 3.1 登录服务器

```bash
ssh -p 23 root@117.50.186.145
```

### 3.2 克隆代码

```bash
cd /root
git clone https://github.com/690419071fybFYB/RF-DETR.git
cd RF-DETR
git checkout feature/density-guided-init  # 切换到你的分支
git pull origin feature/density-guided-init
```

### 3.3 安装系统依赖

```bash
# 安装 OpenCV 需要的库
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 可选：安装其他工具
apt-get install -y wget curl git vim htop
```

### 3.4 创建 Python 环境

#### 方法 A：使用 Conda

```bash
# 创建环境
conda create -n rfdetr python=3.11 -y
conda activate rfdetr

# 安装 PyTorch（根据 CUDA 版本选择）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt

# 使用 headless OpenCV（服务器推荐）
pip uninstall opencv-python -y
pip install opencv-python-headless
```

#### 方法 B：使用 uv（更快，推荐）

```bash
# 安装 uv（官方方式）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 或使用国内镜像安装 uv
curl -LsSf https://ghproxy.cc/https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 或使用 pip 安装 uv
pip install uv

# 验证安装
uv --version

# 创建虚拟环境
cd /root/RF-DETR
uv venv --python 3.11

# 激活环境
source .venv/bin/activate

# 安装依赖（比 pip 快 10-100 倍）
uv pip install -r requirements.txt

# 使用国内镜像安装依赖
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 使用 headless OpenCV
uv pip uninstall opencv-python
uv pip install opencv-python-headless
```

### 3.5 配置国内镜像（可选，加速下载）

```bash
# pip 使用清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或在安装时指定
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## ✅ 第四部分：验证环境

```bash
# 激活环境
conda activate rfdetr  # 或 source .venv/bin/activate

# 测试导入
python -c "from rfdetr import RFDETRBase; print('✅ 环境配置成功！')"

# 检查 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 🏃 第五部分：运行训练

### 5.1 运行单个实验

```bash
cd /root/RF-DETR
python experiements/scripts/ablation_baseline.py
```

### 5.2 后台运行（推荐）

```bash
# 使用 nohup 后台运行
nohup python experiements/scripts/ablation_baseline.py > logs/train.log 2>&1 &

# 查看日志
tail -f logs/train.log

# 查看进程
ps aux | grep python
```

### 5.3 运行批量实验

```bash
chmod +x run_ablation_study.sh
nohup ./run_ablation_study.sh > logs/ablation_all.log 2>&1 &
```

---

## 📥 第六部分：下载结果

训练完成后，将结果下载回本地：

```bash
# 在本地执行
scp -r -P 23 root@117.50.186.145:/root/RF-DETR/experiements/results/ /home/fyb/mydir/rf-detr/experiements/results_server/
```

---

## 🔄 快速部署脚本（一键执行）

将以下脚本保存为 `deploy_to_server.sh`，一键部署：

```bash
#!/bin/bash

# ========== 配置信息 ==========
SERVER_IP="117.50.186.145"
SERVER_PORT="23"
SERVER_USER="root"

LOCAL_PROJECT="/home/fyb/mydir/rf-detr"
LOCAL_DATASET="/home/fyb/datasets/RSOD_cocoFormat"

# ========== 传输数据集 ==========
echo "📦 传输数据集..."
scp -r -P $SERVER_PORT $LOCAL_DATASET $SERVER_USER@$SERVER_IP:/root/

# ========== 推送代码 ==========
echo "📤 推送代码到 GitHub..."
cd $LOCAL_PROJECT
git add .
git commit -m "同步到服务器 $(date +%Y%m%d_%H%M%S)"
git push origin $(git branch --show-current)

# ========== 服务器端配置 ==========
echo "🔧 配置服务器环境..."
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP << 'EOF'
# 拉取代码
cd /root/RF-DETR
git pull origin feature/density-guided-init

# 安装系统依赖
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 安装 Python 依赖
pip install -r requirements.txt
pip uninstall opencv-python -y
pip install opencv-python-headless

# 测试
python -c "from rfdetr import RFDETRBase; print('✅ 环境配置成功！')"
EOF

echo "✅ 部署完成！"
```

---

## ❓ 常见问题

### Q1: scp 传输大文件断连
**解决**：使用分块传输（见 2.2）或在服务器直接下载

### Q2: ImportError: libGL.so.1
**解决**：
```bash
apt-get install -y libgl1-mesa-glx
# 或使用 headless OpenCV
pip install opencv-python-headless
```

### Q3: CUDA out of memory
**解决**：减小 batch_size
```python
model.train(batch_size=2, ...)
```

### Q4: 训练太慢
**解决**：检查是否使用了 GPU
```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
```

---

## 🌐 第七部分：服务器代理配置（可选）

> 如果需要在服务器上访问 GitHub、HuggingFace 等国外网站，可以配置代理

### 7.1 安装 mihomo（Clash 替代品）

```bash
# 下载 mihomo（使用国内镜像）
cd /root
wget https://ghproxy.cc/https://github.com/MetaCubeX/mihomo/releases/download/v1.18.10/mihomo-linux-amd64-v1.18.10.gz

# 解压并安装
gunzip mihomo-linux-amd64-v1.18.10.gz
chmod +x mihomo-linux-amd64-v1.18.10
mv mihomo-linux-amd64-v1.18.10 /usr/local/bin/clash

# 验证
clash -v
```

### 7.2 下载订阅配置

```bash
# 创建配置目录
mkdir -p ~/.config/clash

# 下载订阅（替换为你的订阅链接）
wget -O ~/.config/clash/config.yaml --user-agent="ClashforWindows" "你的订阅链接?flag=clash"

# 检查配置
head -20 ~/.config/clash/config.yaml
```

### 7.3 下载 GeoIP 数据库

```bash
# 下载 GeoIP 数据库（必须，否则无法启动）
wget -O ~/.config/clash/geoip.metadb "https://ghproxy.cc/https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/geoip.metadb"

# 下载 GeoSite 数据库
wget -O ~/.config/clash/geosite.dat "https://ghproxy.cc/https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/geosite.dat"

# 检查文件
ls -lh ~/.config/clash/
```

### 7.4 启动代理

```bash
# 后台启动
nohup clash -d ~/.config/clash > /root/clash.log 2>&1 &

# 等待启动
sleep 5

# 检查是否运行
ps aux | grep -v grep | grep clash

# 查看日志
tail -20 /root/clash.log
```

### 7.5 设置环境变量

```bash
# 临时设置（当前会话有效）
# 注意：端口根据你的配置文件中的 mixed-port 或 port 设置
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897

# 永久设置（添加到 ~/.bashrc）
echo 'export http_proxy=http://127.0.0.1:7897' >> ~/.bashrc
echo 'export https_proxy=http://127.0.0.1:7897' >> ~/.bashrc
source ~/.bashrc

# 测试代理
curl -I https://www.google.com
```

### 7.6 关闭代理

```bash
# 取消环境变量
unset http_proxy
unset https_proxy
unset all_proxy

# 停止 clash 进程
pkill clash
```

### 7.7 常用国内镜像（无需代理）

```bash
# pip 使用清华镜像
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# GitHub 文件加速
# 原链接: https://github.com/xxx/xxx/releases/download/v1.0/file.zip
# 加速链接: https://ghproxy.cc/https://github.com/xxx/xxx/releases/download/v1.0/file.zip
```

---

## 📝 备注

- 本教程基于 RF-DETR 项目
- 测试服务器：AutoDL / 阿里云 / 腾讯云
- 最后更新：2024-12-12
