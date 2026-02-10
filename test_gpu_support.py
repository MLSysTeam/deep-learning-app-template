#!/usr/bin/env python3
"""
测试GPU支持情况
"""
import torch
import sys
sys.path.append('app')

from model_deployer import ModelDeployer


def test_gpu_support():
    print("🔍 检查GPU支持情况")
    print("="*40)

    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA可用: {cuda_available}")

    if cuda_available:
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - 内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ CUDA不可用，将使用CPU进行推理")

    # 检查是否支持mps (Apple Silicon)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"✓ MPS可用 (Apple Silicon): {mps_available}")

    # 测试基本GPU功能
    print(f"\n🧪 测试GPU功能...")
    try:
        if cuda_available:
            # 测试张量创建和计算
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            print("✓ CUDA张量计算正常")

            # 测试模型移动到GPU
            deployer = ModelDeployer(model_name="resnet18", pretrained=False)
            simple_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            simple_model.eval()
            deployer.original_model = simple_model

            # 尝试将模型移动到GPU
            gpu_model = simple_model.cuda()
            print("✓ 模型可移动到GPU")

            # 尝试推理
            test_input = torch.randn(1, 3, 224, 224).cuda()
            with torch.no_grad():
                output = gpu_model(test_input)
            print(f"✓ GPU推理正常，输出形状: {output.shape}")

        elif mps_available:
            # 测试MPS功能
            x = torch.randn(3, 3).to("mps")
            y = torch.randn(3, 3).to("mps")
            z = torch.mm(x, y)
            print("✓ MPS张量计算正常")

            print("✓ 当前系统支持MPS加速 (Apple Silicon)")
        else:
            print("⚠ 没有检测到GPU支持，将使用CPU")

    except Exception as e:
        print(f"✗ GPU功能测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试ONNX Runtime GPU支持
    print(f"\n🔄 检查ONNX Runtime GPU支持...")
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        print(f"✓ ONNX Runtime可用提供者: {available_providers}")

        if 'CUDAExecutionProvider' in available_providers:
            print("✓ ONNX Runtime支持CUDA加速")
        else:
            print("⚠ ONNX Runtime未检测到CUDA支持")

    except ImportError:
        print("⚠ ONNX Runtime未安装或不可用")
    except Exception as e:
        print(f"⚠ ONNX Runtime GPU检查失败: {e}")

    print(f"\n📋 环境总结:")
    print(f"- PyTorch版本: {torch.__version__}")
    print(f"- CUDA可用: {cuda_available}")
    print(f"- MPS可用: {mps_available}")
    print(f"- CPU核心数: {torch.get_num_threads()}")

    if cuda_available:
        print(f"- 默认设备: CUDA")
        return "cuda"
    elif mps_available:
        print(f"- 默认设备: MPS")
        return "mps"
    else:
        print(f"- 默认设备: CPU")
        return "cpu"


def test_gpu_optimization():
    print(f"\n🚀 测试GPU优化部署...")

    device_type = test_gpu_support()

    if device_type in ["cuda", "mps"]:
        try:
            # 创建部署器并加载模型
            deployer = ModelDeployer(model_name="resnet18", pretrained=False)
            import torchvision.models as models
            model = models.resnet18()
            model.eval()
            deployer.original_model = model

            # 移动到GPU
            if device_type == "cuda":
                model_gpu = model.cuda()
            elif device_type == "mps":
                model_gpu = model.to("mps")

            print("✓ 模型成功移动到GPU")

            # 测试基本推理
            test_input = torch.randn(1, 3, 224, 224)
            if device_type == "cuda":
                test_input = test_input.cuda()
            elif device_type == "mps":
                test_input = test_input.to("mps")

            with torch.no_grad():
                output = model_gpu(test_input)
            print(f"✓ GPU推理成功，输出形状: {output.shape}")

            # 测试优化
            jit_model = deployer.optimize_with_jit()
            print("✓ JIT优化成功")

            # 对于GPU，还可以测试ONNX导出
            onnx_path = deployer.optimize_with_onnx()
            print("✓ ONNX转换成功")

            if onnx_path and torch.cuda.is_available():
                # ONNX Runtime GPU推理测试需要CUDA版本
                try:
                    import onnxruntime as ort
                    if 'CUDAExecutionProvider' in ort.get_available_providers():
                        session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
                        print("✓ ONNX Runtime CUDA推理配置成功")
                    else:
                        print("ℹ ONNX Runtime CUDA提供者不可用")
                except Exception as e:
                    print(f"ℹ ONNX Runtime GPU测试跳过: {e}")

            print("🎉 GPU优化测试完成")
            return True

        except Exception as e:
            print(f"❌ GPU优化测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("ℹ GPU不可用，跳过GPU优化测试")
        return True


if __name__ == "__main__":
    test_gpu_optimization()