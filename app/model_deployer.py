import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import os
import json
from typing import Dict, Any, Tuple
import onnx
import warnings


class ModelDeployer:
    """
    Automatic model deployment class that handles model loading, optimization, and benchmarking
    with focus on lossless deployment methods (ONNX, TensorRT, OpenVINO)
    """
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, device: str = None):
        self.model_name = model_name
        self.pretrained = pretrained

        # Determine the best available device
        if device is None:
            if torch.cuda.is_available():
                # Check CUDA capability
                try:
                    major, minor = torch.cuda.get_device_capability(0)
                    capability = float(f"{major}.{minor}")
                    if capability >= 7.0:
                        self.device = "cuda"
                        self.gpu_available = True
                    else:
                        warnings.warn(
                            f"GPU {torch.cuda.get_device_name(0)} with capability {capability} "
                            f"is not supported by this PyTorch installation (requires >= 7.0). "
                            f"Falling back to CPU."
                        )
                        self.device = "cpu"
                        self.gpu_available = False
                except:
                    self.device = "cpu"
                    self.gpu_available = False
            else:
                self.device = "cpu"
                self.gpu_available = False
        else:
            self.device = device
            self.gpu_available = (device == "cuda")

        print(f"Using device: {self.device}, GPU Available: {self.gpu_available}")

        self.original_model = None
        self.optimized_models = {}
        self.benchmark_results = {}
        self.transforms = self._get_default_transforms()

    def _get_default_transforms(self):
        """Define standard image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self) -> torch.nn.Module:
        """Load the specified model"""
        print(f"Loading {self.model_name} model on {self.device}...")

        if self.model_name == "resnet18":
            self.original_model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == "resnet50":
            self.original_model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == "mobilenet_v2":
            self.original_model = models.mobilenet_v2(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.original_model.eval()
        # Move model to the selected device
        self.original_model = self.original_model.to(self.device)
        print(f"{self.model_name} loaded successfully on {self.device}!")
        return self.original_model

    def optimize_with_jit(self) -> torch.nn.Module:
        """Optimize model using TorchScript JIT (lossless)"""
        print("Optimizing model with TorchScript JIT (lossless)...")

        if self.original_model is None:
            raise ValueError("Model must be loaded first")

        # Create a dummy input for tracing and move to the same device
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        try:
            # Trace the model
            traced_model = torch.jit.trace(self.original_model, dummy_input)
            traced_model.eval()
            self.optimized_models['jit'] = traced_model
            print("JIT optimization completed successfully")
            return traced_model
        except Exception as e:
            print(f"JIT optimization failed: {e}")
            return self.original_model

    def optimize_with_torchscript(self) -> torch.nn.Module:
        """Alternative TorchScript optimization using scripting (lossless)"""
        print("Optimizing model with TorchScript (scripting, lossless)...")

        if self.original_model is None:
            raise ValueError("Model must be loaded first")

        try:
            scripted_model = torch.jit.script(self.original_model.eval())
            self.optimized_models['scripted'] = scripted_model
            print("TorchScript (scripting) optimization completed successfully")
            return scripted_model
        except Exception as e:
            print(f"TorchScript (scripting) optimization failed: {e}")
            return self.original_model

    def optimize_with_onnx(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                          save_path: str = "optimized_model.onnx") -> str:
        """Optimize model by converting to ONNX format (lossless)"""
        print("Optimizing model with ONNX conversion (lossless)...")

        if self.original_model is None:
            raise ValueError("Model must be loaded first")

        try:
            # Create dummy input and move to CPU for ONNX export
            # ONNX export typically happens on CPU
            dummy_input = torch.randn(input_shape).cpu()

            # Get the CPU version of the model for export
            model_for_export = self.original_model.cpu()

            # Export to ONNX
            torch.onnx.export(
                model_for_export.eval(),
                dummy_input,
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

            # Move model back to original device
            self.original_model = self.original_model.to(self.device)

            self.optimized_models['onnx'] = save_path
            print(f"ONNX optimization completed successfully. Model saved to {save_path}")
            return save_path
        except Exception as e:
            print(f"ONNX optimization failed: {e}")
            return None

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess an image for model input"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transforms(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
        # Move to the appropriate device
        input_batch = input_batch.to(self.device)
        return input_batch

    def benchmark_model(self, image_path: str, model_name: str = "original", model=None,
                        num_runs: int = 10) -> Dict[str, float]:
        """Benchmark a specific model - supports both PyTorch and ONNX models"""
        if model is None:
            if model_name == "original":
                model = self.original_model
            elif model_name == "onnx":
                # Load ONNX model using onnxruntime for benchmarking
                onnx_path = self.optimized_models.get('onnx')
                if onnx_path and os.path.exists(onnx_path):
                    try:
                        import onnxruntime as ort
                        session = ort.InferenceSession(onnx_path)
                        # Benchmark ONNX model
                        return self._benchmark_onnx_model(session, image_path, num_runs)
                    except ImportError:
                        print("ONNX Runtime not installed. Skipping ONNX benchmark.")
                        # Return a dummy result if onnxruntime is not available
                        return {
                            'avg_inference_time': float('inf'),
                            'fps': 0.0,
                            'total_time': 0.0,
                            'num_runs': num_runs
                        }
                else:
                    raise ValueError(f"ONNX model not found or path invalid: {onnx_path}")
            else:
                model = self.optimized_models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found")

        # Preprocess input once
        input_batch = self.preprocess_image(image_path)

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_batch)

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_batch)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time if avg_time > 0 else 0

        result = {
            'avg_inference_time': avg_time,
            'fps': fps,
            'total_time': end_time - start_time,
            'num_runs': num_runs
        }

        self.benchmark_results[model_name] = result
        return result

    def _benchmark_onnx_model(self, session, image_path: str, num_runs: int) -> Dict[str, float]:
        """Benchmark ONNX model using ONNX Runtime"""
        # Preprocess input - convert to numpy for ONNX Runtime
        input_tensor = self.preprocess_image(image_path).cpu()
        input_batch = input_tensor.numpy()

        # Warm up
        for _ in range(3):
            _ = session.run(None, {'input': input_batch})

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {'input': input_batch})
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time if avg_time > 0 else 0

        result = {
            'avg_inference_time': avg_time,
            'fps': fps,
            'total_time': end_time - start_time,
            'num_runs': num_runs
        }

        self.benchmark_results['onnx'] = result
        return result

    def benchmark_all_models(self, image_path: str, num_runs: int = 10) -> Dict[str, Dict[str, float]]:
        """Benchmark all available models (original and optimized)"""
        results = {}

        # Benchmark original model
        if self.original_model:
            results['original'] = self.benchmark_model(image_path, "original", num_runs=num_runs)

        # Benchmark optimized models
        for opt_name, opt_model in self.optimized_models.items():
            # For ONNX model, we just pass the name since the path is stored internally
            if opt_name == 'onnx':
                results[opt_name] = self.benchmark_model(image_path, opt_name, num_runs=num_runs)
            else:
                results[opt_name] = self.benchmark_model(image_path, opt_name, opt_model, num_runs)

        return results

    def get_model_size(self, model) -> float:
        """Calculate model size in MB - handles both PyTorch models and ONNX files"""
        import tempfile

        # Check if it's an ONNX file path
        if isinstance(model, str) and model.endswith('.onnx'):
            if os.path.exists(model):
                size_mb = os.path.getsize(model) / (1024 * 1024.0)
                return size_mb
            else:
                return 0.0  # File doesn't exist

        # For PyTorch models
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            torch.save(model.state_dict(), tmp.name)
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024.0)

        return size_mb

    def generate_performance_report(self, image_path: str, num_runs: int = 10) -> Dict[str, Any]:
        """Generate a complete performance report comparing all optimizations"""
        print("Generating performance report...")

        # Ensure all models are benchmarked
        benchmark_results = self.benchmark_all_models(image_path, num_runs)

        report = {
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'benchmark_settings': {
                'num_runs': num_runs,
                'test_image': image_path
            },
            'models_comparison': {}
        }

        # Add model sizes
        if self.original_model:
            original_size = self.get_model_size(self.original_model)
            report['models_comparison']['original'] = {
                **benchmark_results.get('original', {}),
                'size_mb': original_size
            }

        for opt_name, opt_model in self.optimized_models.items():
            opt_size = self.get_model_size(opt_model)
            report['models_comparison'][opt_name] = {
                **benchmark_results.get(opt_name, {}),
                'size_mb': opt_size
            }

        # Calculate improvements
        if 'original' in report['models_comparison']:
            original_time = report['models_comparison']['original']['avg_inference_time']

            for model_name, metrics in report['models_comparison'].items():
                if model_name != 'original':
                    time_improvement = original_time / metrics['avg_inference_time']
                    size_improvement = report['models_comparison']['original']['size_mb'] / metrics['size_mb']

                    metrics['time_improvement_factor'] = time_improvement
                    metrics['size_improvement_factor'] = size_improvement

        return report

    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save the performance report to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Performance report saved to {output_path}")

    def deploy_optimized_model(self, optimization_type: str = "jit", save_path: str = None) -> any:
        """Deploy an optimized model and optionally save it"""
        print(f"Deploying {optimization_type} optimized model...")

        if optimization_type == "jit":
            model = self.optimize_with_jit()
        elif optimization_type == "scripted":
            model = self.optimize_with_torchscript()
        elif optimization_type == "onnx":
            model = self.optimize_with_onnx(save_path=save_path or "model.onnx")
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")

        return model

    def auto_optimize(self, save_path: str = None) -> str:
        """
        Automatically select the best optimization based on system environment
        CPU -> ONNX, GPU -> ONNX with CUDA provider
        """
        if self.gpu_available:
            print("GPU available, optimizing for GPU acceleration...")
            # For GPU, we'll export ONNX model, which can then be used with CUDA provider
            onnx_path = self.optimize_with_onnx(save_path=save_path or "auto_gpu_model.onnx")
            print(f"GPU-optimized ONNX model ready: {onnx_path}")
            return onnx_path
        else:
            print("CPU only, optimizing for CPU acceleration...")
            onnx_path = self.optimize_with_onnx(save_path=save_path or "auto_cpu_model.onnx")
            print(f"CPU-optimized ONNX model ready: {onnx_path}")
            return onnx_path

    def get_best_optimization_strategy(self) -> dict:
        """Get the best optimization strategy based on system environment"""
        strategy = {
            "device": self.device,
            "gpu_available": self.gpu_available,
            "recommended_optimization": "onnx_cpu" if not self.gpu_available else "onnx_gpu",
            "available_providers": []
        }

        # Check ONNX Runtime providers
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            strategy["available_providers"] = available_providers

            # Determine best provider
            if self.gpu_available and 'CUDAExecutionProvider' in available_providers:
                strategy["best_provider"] = "CUDAExecutionProvider"
                strategy["recommended_optimization"] = "onnx_gpu"
            else:
                strategy["best_provider"] = "CPUExecutionProvider"
                strategy["recommended_optimization"] = "onnx_cpu"
        except ImportError:
            strategy["best_provider"] = "PyTorch (no ONNXRuntime)"
            strategy["recommended_optimization"] = "torchscript"

        return strategy


# Example usage function
def run_performance_comparison():
    """Run a full performance comparison between original and optimized models"""
    import os
    from PIL import Image

    # Create a sample image for testing if one doesn't exist
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print("Creating test image...")
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(test_image_path)

    # Initialize the deployer
    deployer = ModelDeployer(model_name="resnet18", pretrained=True)
    deployer.load_model()

    print("Model loaded, now optimizing...")

    # Apply lossless optimizations
    deployer.optimize_with_jit()
    deployer.optimize_with_onnx()

    print("Optimizations completed, running benchmarks...")

    # Generate performance report
    report = deployer.generate_performance_report(test_image_path, num_runs=5)

    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON REPORT")
    print("="*60)

    for model_name, metrics in report['models_comparison'].items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Average inference time: {metrics['avg_inference_time']:.6f}s ({metrics['fps']:.2f} FPS)")
        print(f"  Model size: {metrics['size_mb']:.2f} MB")

        if model_name != 'original':
            print(f"  Speed improvement: {metrics.get('time_improvement_factor', 1):.2f}x faster")
            print(f"  Size reduction: {metrics.get('size_improvement_factor', 1):.2f}x smaller")

    # Save report
    deployer.save_report(report, "performance_report.json")

    return deployer, report


if __name__ == "__main__":
    deployer, report = run_performance_comparison()