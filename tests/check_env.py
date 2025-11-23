import importlib
import sys
import torch

print("========== RF-DETR 环境自动检查 ==========\n")

def check_package(name):
    try:
        pkg = importlib.import_module(name)
        print(f"[OK] {name} 已安装 - version: {getattr(pkg, '__version__', 'unknown')}")
        return pkg
    except Exception as e:
        print(f"[ERR] {name} 未安装 or import 失败: {e}")
        return None


# ---------------------------------------------
# 1. 检查 Python 版本
# ---------------------------------------------
print(f"Python 版本: {sys.version}\n")


# ---------------------------------------------
# 2. 检查 PyTorch
# ---------------------------------------------
print("检查 PyTorch:")
torch_pkg = check_package("torch")

if torch_pkg:
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()


# ---------------------------------------------
# 3. 检查 torchvision / torchaudio
# ---------------------------------------------
print("检查 Torch 生态包:")
check_package("torchvision")
check_package("torchaudio")
print()


# ---------------------------------------------
# 4. 检查 Transformers & Tokenizers
# ---------------------------------------------
print("检查 Transformers:")
check_package("transformers")
check_package("tokenizers")
print()


# ---------------------------------------------
# 5. 检查 rfdetr 包（pip install -e .）
# ---------------------------------------------
print("检查 RF-DETR 包:")
rfdetr_pkg = check_package("rfdetr")
print()


# ---------------------------------------------
# 6. 尝试最小前向（如果 rfdetr 存在）
# ---------------------------------------------
if rfdetr_pkg and torch_pkg:
    print("尝试最小模型前向计算...")

    try:
        from rfdetr.models.lwdetr import build_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 极小模型配置，符合 RF-DETR 默认接口
        class Args:
            # 基础
            num_classes = 91
            num_queries = 300
            hidden_dim = 256

            # Backbone 配置（根据你当前模型结构）
            backbone = "dinov2_l"   # 或 "dinov2_large"，两者 RF-DETR 都可自动识别

            # Transformer 配置
            encoder_layers = 12
            decoder_layers = 3

            # 小目标增强
            enable_soqb = False
            enable_small_obj_loss = False

            # 设备
            device = "cuda" if torch.cuda.is_available() else "cpu"


        args = Args()
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        model.eval()

        # 创建 fake 输入
        dummy = torch.randn(1, 3, 320, 320).to(device)

        with torch.no_grad():
            out = model(dummy)

        print("[OK] RF-DETR 最小前向通过！输出键：", out.keys())

    except Exception as e:
        print(f"[ERR] RF-DETR 前向失败: {e}")

print("\n========== 检查结束 ==========")
