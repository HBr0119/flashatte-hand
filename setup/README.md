# C500 服务器部署指南

从零开始在新 MetaX C500 服务器上部署 FlashAttention Agent Lab。

## 前置条件

- MetaX C500 GPU (xcore1000) + mxdriver
- Ubuntu/Debian Linux
- MACA Toolkit 已安装于 `/opt/maca`
- mxdriver 已安装于 `/opt/mxdriver`

---

## 1. 基础环境

### 1.1 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
```

### 1.2 配置 Shell 环境

```bash
# 复制 MACA 环境变量脚本
cp setup/dotfiles/maca_env.sh /opt/maca_env.sh

# 添加到 ~/.bashrc
echo "source /opt/maca_env.sh" >> ~/.bashrc
source ~/.bashrc
```

### 1.3 配置 pip 镜像 (国内服务器)

```bash
mkdir -p ~/.pip
cp setup/dotfiles/pip.conf ~/.pip/pip.conf
```

### 1.4 配置 Git

```bash
cp setup/dotfiles/gitconfig ~/.gitconfig
# 如果已有 git 配置，手动合并
```

---

## 2. Python 依赖

### 核心依赖

```bash
pip install torch==2.8.0+metax3.3.0.2  # MACA 版本，由 MACA 提供
pip install -r requirements.txt
```

### 完整 pip 列表

参考 [pip_full_list.txt](pip_full_list.txt) 查看当前服务器完整 pip 环境。

---

## 3. VPN 配置 (mihomo)

需要科学上网访问 OpenAI API。

### 3.1 安装 mihomo

```bash
chmod +x setup/vpn/setup.sh
./setup/vpn/setup.sh
```

### 3.2 配置代理节点

编辑 `/etc/mihomo/config.yaml`，参考 [setup/vpn/config.yaml.example](vpn/config.yaml.example)。

**你需要自己准备代理订阅/节点信息**（出于安全原因不包含在仓库中）。

### 3.3 启动 VPN

```bash
# 前台运行（调试用）
mihomo -d /etc/mihomo

# 后台运行
nohup mihomo -d /etc/mihomo > /var/log/mihomo.log 2>&1 &

# 验证
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
curl -I https://api.openai.com
```

---

## 4. 项目配置

### 4.1 克隆仓库

```bash
git clone git@github.com:HBr0119/flashattention-c500-agent-lab.git
cd flashattention-c500-agent-lab
```

### 4.2 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入你的 OpenAI API key
```

`.env` 文件内容：
```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5
OPENAI_REASONING_EFFORT=high
OPENAI_TIMEOUT=600
OPENAI_RETRIES=2
```

### 4.3 验证环境

```bash
# 检查 MACA GPU 可用
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 应输出:
# True
# MetaX xcore1000
```

---

## 5. 目录结构约定

| 路径 | 用途 |
|------|------|
| `/data/` | 工作目录，存放实验数据和模型缓存 |
| `/data/huggingface_home/` | HF 模型缓存 |
| `/opt/maca/` | MACA Toolkit |
| `/opt/mxdriver/` | MACA GPU Driver |
| `/opt/conda/` | Conda 安装路径 |
| `/etc/mihomo/` | VPN 配置 |

---

## 6. 快速验证

```bash
# OJ 赛道 smoke check
python scripts/kernelclaw_flashattn_operator_loop.py \
  --baseline-src submission-oj/best_flashattn_kvcache_decode.cu \
  --out-dir /tmp/smoke_check \
  --task-profile contest \
  --smoke --disable-auxiliary --rounds 1

# 直接验证 OJ 最佳 kernel
python scripts/evaluate_flashattn_submission_oj.py \
  --src submission-oj/best_flashattn_kvcache_decode.cu \
  --out /tmp/validation \
  --warmup 2 --repeat 5
```

---

## 7. SSH 配置 (可选)

将你的 SSH 公钥添加到 `~/.ssh/authorized_keys`：

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "your-ssh-public-key" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**不要将私钥提交到仓库！**
