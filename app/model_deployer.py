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

        # Keep track of the original device of the model
        original_device = next(self.original_model.parameters()).device
        
        try:
            # Create dummy input and move to CPU for ONNX export
            # ONNX export typically happens on CPU
            dummy_input = torch.randn(input_shape).cpu()

            # Get the CPU version of the model for export
            model_for_export = self.original_model.cpu()

            # Export to ONNX with a higher opset version to avoid automatic upgrades and conversion errors
            torch.onnx.export(
                model_for_export.eval(),
                dummy_input,
                save_path,
                export_params=True,
                opset_version=18,  # Updated to match the automatically selected version
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=None,  # Temporarily disable dynamic_axes to avoid the warning
                # Use newer parameters to avoid dynamo warnings
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False
            )

            # Move model back to original device
            self.original_model = self.original_model.to(original_device)

            self.optimized_models['onnx'] = save_path
            print(f"ONNX optimization completed successfully. Model saved to {save_path}")
            return save_path
        except Exception as e:
            print(f"ONNX optimization failed: {e}")
            
            # Even if ONNX export fails, make sure original model is still on the right device
            self.original_model = self.original_model.to(original_device)
            
            return None

    def optimize_with_tensorrt(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                              save_path: str = "optimized_model.trt") -> str:
        """Optimize model by converting to TensorRT format via ONNX (lossless)"""
        print("Optimizing model with TensorRT conversion via ONNX (lossless)...")

        if self.original_model is None:
            raise ValueError("Model must be loaded first")

        # First convert to ONNX, then to TensorRT
        onnx_path = self.optimize_with_onnx(input_shape, save_path.replace(".trt", ".onnx"))
        
        if onnx_path is None:
            print("Cannot optimize with TensorRT: ONNX conversion failed")
            return None

        try:
            import onnx
            import tensorrt as trt

            # Create TensorRT builder
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            
            # Check for newer TensorRT API (8.5+)
            if hasattr(trt, 'NetworkDefinitionCreationFlag'):
                # Newer API
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            else:
                # Older API
                network = builder.create_network(1)
                
            parser = trt.OnnxParser(network, logger)
            
            # Parse the ONNX file
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(f"TensorRT ONNX parser error {error}: {parser.get_error(error)}")
                    return None

            # Configure the builder
            config = builder.create_builder_config()
            
            # Handle different TensorRT version APIs for setting memory pool
            if hasattr(trt, 'MemoryPoolType'):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB for workspace
            else:
                # Older versions
                config.max_workspace_size = 1 << 30  # 1GB for workspace
            
            # Enable FP16 if available
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # Build the engine - handle different APIs
            if hasattr(builder, 'build_serialized_network'):
                serialized_engine = builder.build_serialized_network(network, config)
            else:
                engine = builder.build_engine(network, config)
                if engine is None:
                    print("ERROR: Failed to build TensorRT engine")
                    return None
                serialized_engine = engine.serialize()

            if serialized_engine is None:
                print("ERROR: Failed to build TensorRT engine")
                return None

            # Save the TensorRT engine
            with open(save_path, 'wb') as f:
                f.write(serialized_engine)

            self.optimized_models['tensorrt'] = save_path
            print(f"TensorRT optimization completed successfully. Engine saved to {save_path}")
            return save_path

        except ImportError as e:
            print(f"TensorRT optimization failed due to missing dependencies: {e}")
            print("To enable TensorRT, install tensorrt: pip install nvidia-tensorrt")
            return None
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            return None

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess an image for model input"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transforms(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
        # Move to the appropriate device
        input_batch = input_batch.to(self.device)
        return input_batch

    def _benchmark_tensorrt_model(self, model_trt_path, image_path: str, num_runs: int) -> Dict[str, float]:
        """Benchmark TensorRT model"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            import numpy as np
            
            # Load the serialized engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(model_trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            # Preprocess input
            input_tensor = self.preprocess_image(image_path)
            input_image = input_tensor.cpu().numpy()  # Convert to numpy array
            
            batch_size = input_image.shape[0]
            image_channel = input_image.shape[1]
            image_height = input_image.shape[2]
            image_width = input_image.shape[3]
            
            with engine.create_execution_context() as context:
                # Check if engine has the newer API attributes
                if hasattr(engine, 'num_io_tensors'):
                    # Newer TensorRT API (8.5+)
                    num_bindings = engine.num_io_tensors
                    for i in range(num_bindings):
                        tensor_name = engine.get_tensor_name(i)
                        
                        # Check if TensorMode attribute exists
                        if hasattr(trt, 'TensorMode'):
                            tensor_mode = engine.get_tensor_mode(tensor_name)
                            is_input = tensor_mode == trt.TensorMode.INPUT
                        else:
                            # Fallback to older method
                            try:
                                # Try to determine if it's an input using get_tensor_loc if available
                                if hasattr(engine, 'get_tensor_loc'):
                                    tensor_loc = engine.get_tensor_loc(tensor_name)
                                    is_input = tensor_loc == trt.TensorLocation.DEVICE
                                else:
                                    # Fallback: assume first tensor is input
                                    is_input = (i == 0)
                            except:
                                # Final fallback: assume first tensor is input
                                is_input = (i == 0)
                                
                        if is_input:
                            input_idx = i
                            if hasattr(context, 'set_tensor_shape'):
                                # New API for setting tensor shape
                                context.set_tensor_shape(tensor_name, (batch_size, image_channel, image_height, image_width))
                            elif hasattr(context, 'set_input_shape'):
                                # Older new API
                                context.set_input_shape(tensor_name, (batch_size, image_channel, image_height, image_width))
                        else:
                            output_idx = i
                            
                    # Allocate host and device buffers for new API
                    bindings = [None] * num_bindings
                    for idx in range(num_bindings):
                        tensor_name = engine.get_tensor_name(idx)
                        
                        if hasattr(context, 'get_tensor_shape'):
                            shape = context.get_tensor_shape(tensor_name)
                        else:
                            shape = context.get_binding_shape(tensor_name) if hasattr(context, 'get_binding_shape') else (batch_size, image_channel, image_height, image_width)
                            
                        size = trt.volume(shape) * batch_size
                        
                        # Determine if tensor is input or output
                        if hasattr(trt, 'TensorMode') and hasattr(engine, 'get_tensor_mode'):
                            tensor_mode = engine.get_tensor_mode(tensor_name)
                            is_input_tensor = tensor_mode == trt.TensorMode.INPUT
                        else:
                            # Fallback to binding_is_input if available
                            try:
                                binding_idx = engine.get_binding_index(tensor_name) if hasattr(engine, 'get_binding_index') else idx
                                is_input_tensor = engine.binding_is_input(binding_idx) if hasattr(engine, 'binding_is_input') else (idx == 0)  # Assume first is input
                            except:
                                is_input_tensor = (idx == 0)  # Default assumption
                        
                        if is_input_tensor:
                            input_buffer = np.ascontiguousarray(input_image.reshape(-1))
                            input_memory = cuda.mem_alloc(input_image.nbytes)
                            bindings[idx] = int(input_memory)
                        else:
                            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name)) if hasattr(engine, 'get_tensor_dtype') else trt.nptype(trt.float32)
                            output_buffer = cuda.pagelocked_empty(size, dtype)
                            output_memory = cuda.mem_alloc(output_buffer.nbytes)
                            bindings[idx] = int(output_memory)

                    # For newer TensorRT versions, we need to set tensor addresses explicitly
                    stream = cuda.Stream()
                    
                    # Set input tensor address explicitly for newer API
                    if hasattr(context, 'set_input_tensor'):
                        # Even newer API that uses set_input_tensor
                        input_tensor_ptr = int(input_memory)
                        context.set_input_tensor(engine.get_tensor_name(input_idx), input_tensor_ptr)
                        output_tensor_ptr = int(output_memory)
                        context.set_output_tensor(engine.get_tensor_name(output_idx), output_tensor_ptr)
                    elif hasattr(context, 'set_tensor_address'):
                        # Newer API that uses set_tensor_address
                        input_tensor_name = engine.get_tensor_name(input_idx)
                        output_tensor_name = engine.get_tensor_name(output_idx)
                        context.set_tensor_address(input_tensor_name, int(input_memory))
                        context.set_tensor_address(output_tensor_name, int(output_memory))
                    else:
                        # Older API still uses bindings array directly
                        pass

                else:
                    # Older TensorRT API (pre-8.5)
                    # Find the input and output indices
                    input_idx = -1
                    output_idx = -1
                    
                    if hasattr(engine, 'num_bindings'):
                        # Using num_bindings property
                        num_bindings = engine.num_bindings
                    else:
                        # Fallback to iterating over engine
                        # This approach is deprecated but works with older versions
                        num_bindings = len([i for i in range(100) if hasattr(engine, 'get_binding_name') and engine.get_binding_name(i)])  # Guessing upper bound
                        
                    for i in range(num_bindings):
                        if engine.binding_is_input(i):
                            input_idx = i
                        else:
                            output_idx = i
                    
                    # Set input shape based on image dimensions for inference
                    if input_idx != -1:
                        if hasattr(context, 'set_binding_shape'):
                            context.set_binding_shape(input_idx, (batch_size, image_channel, image_height, image_width))
                        else:
                            # Older API doesn't support dynamic shapes
                            pass

                    # Allocate host and device buffers for old API
                    bindings = [None] * num_bindings
                    for idx in range(num_bindings):
                        if hasattr(engine, 'get_binding_shape'):
                            size = trt.volume(context.get_binding_shape(idx)) if context.get_binding_shape(idx) else trt.volume(engine.get_binding_shape(idx))
                        else:
                            # For older versions, calculate size differently
                            binding_shape = engine.get_binding_shape(idx) if hasattr(engine, 'get_binding_shape') else (batch_size, image_channel, image_height, image_width)
                            size = trt.volume(binding_shape)
                            
                        dtype = trt.nptype(engine.get_binding_dtype(idx)) if hasattr(engine, 'get_binding_dtype') else trt.float32
                        
                        if engine.binding_is_input(idx):
                            input_buffer = np.ascontiguousarray(input_image)
                            input_memory = cuda.mem_alloc(input_image.nbytes)
                            bindings[idx] = int(input_memory)
                        else:
                            output_buffer = cuda.pagelocked_empty(size, dtype)
                            output_memory = cuda.mem_alloc(output_buffer.nbytes)
                            bindings[idx] = int(output_memory)

                    stream = cuda.Stream()

                # Warm up
                for _ in range(3):
                    # Transfer input data to the GPU.
                    if hasattr(context, 'set_input_shape'):  # Newer API
                        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
                        # Run inference
                        if hasattr(context, 'execute_async_v3'):
                            # New API - no need to pass bindings as they are set via set_tensor_address
                            context.execute_async_v3(stream.handle)
                        else:
                            # Old API
                            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    else:  # Older API
                        cuda.memcpy_htod_async(bindings[input_idx], input_buffer, stream)
                        # Run inference
                        if hasattr(context, 'execute_async_v2'):
                            # Old API
                            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                        elif hasattr(context, 'execute_async_v3'):
                            # New API
                            context.execute_async_v3(stream.handle)
                        else:
                            # Even newer API?
                            context.execute(stream_handle=stream.handle)
                    
                    # Transfer prediction output from the GPU.
                    if hasattr(context, 'set_input_shape'):  # Newer API
                        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                    else:  # Older API
                        cuda.memcpy_dtoh_async(output_buffer, bindings[output_idx], stream)
                    # Synchronize the stream
                    stream.synchronize()

                # Benchmark
                start_time = time.time()
                for _ in range(num_runs):
                    # Transfer input data to the GPU.
                    if hasattr(context, 'set_input_shape'):  # Newer API
                        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
                        # Run inference
                        if hasattr(context, 'execute_async_v3'):
                            # New API - no need to pass bindings as they are set via set_tensor_address
                            context.execute_async_v3(stream.handle)
                        else:
                            # Old API
                            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    else:  # Older API
                        cuda.memcpy_htod_async(bindings[input_idx], input_buffer, stream)
                        # Run inference
                        if hasattr(context, 'execute_async_v2'):
                            # Old API
                            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                        elif hasattr(context, 'execute_async_v3'):
                            # New API
                            context.execute_async_v3(stream.handle)
                        else:
                            # Even newer API?
                            context.execute(stream_handle=stream.handle)
                    
                    # Transfer prediction output from the GPU.
                    if hasattr(context, 'set_input_shape'):  # Newer API
                        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                    else:  # Older API
                        cuda.memcpy_dtoh_async(output_buffer, bindings[output_idx], stream)
                    # Synchronize the stream
                    stream.synchronize()
                end_time = time.time()

                avg_time = (end_time - start_time) / num_runs
                fps = 1.0 / avg_time if avg_time > 0 else 0

                result = {
                    'avg_inference_time': avg_time,
                    'fps': fps,
                    'total_time': end_time - start_time,
                    'num_runs': num_runs
                }

                self.benchmark_results['tensorrt'] = result
                return result
        except ImportError as e:
            print(f"Could not benchmark TensorRT model due to missing dependencies: {e}")
            print("To enable TensorRT benchmarking, install pycuda: pip install pycuda")
            return {
                'avg_inference_time': float('inf'),
                'fps': 0.0,
                'total_time': 0.0,
                'num_runs': num_runs
            }
        except Exception as e:
            print(f"Failed to benchmark TensorRT model: {e}")
            return {
                'avg_inference_time': float('inf'),
                'fps': 0.0,
                'total_time': 0.0,
                'num_runs': num_runs
            }

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
                        
                        # Configure session options for optimal performance
                        sess_options = ort.SessionOptions()
                        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        
                        # Check if CUDA is available and add CUDA provider
                        available_providers = ort.get_available_providers()
                        print(f"Available ONNX providers: {available_providers}")
                        
                        # Prioritize CUDA over TensorRT for ONNX models to avoid conflicts
                        if self.gpu_available and 'CUDAExecutionProvider' in available_providers:
                            print("Using CUDAExecutionProvider for ONNX model")
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        else:
                            print("Using CPUExecutionProvider for ONNX model")
                            providers = ['CPUExecutionProvider']
                        
                        session = ort.InferenceSession(onnx_path, 
                                                      sess_options=sess_options, 
                                                      providers=providers)
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
            elif model_name == "tensorrt":
                # Load and benchmark TensorRT model directly
                trt_path = self.optimized_models.get('tensorrt')
                if trt_path and os.path.exists(trt_path):
                    return self._benchmark_tensorrt_model(trt_path, image_path, num_runs)
                else:
                    raise ValueError(f"TensorRT model not found or path invalid: {trt_path}")
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
        """Benchmark ONNX model using ONNX Runtime with proper device handling"""
        # Preprocess input - convert to numpy for ONNX Runtime
        # Keep input on CPU since ONNX Runtime handles device transfer
        input_tensor = self.preprocess_image(image_path).cpu()
        input_batch = input_tensor.numpy()

        # Get the actual provider being used
        try:
            provider = session.get_providers()[0] if session.get_providers() else 'Unknown'
            print(f"Running ONNX benchmark with provider: {provider}")
        except:
            print("Running ONNX benchmark with default provider")

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
            # For ONNX and TensorRT models, we just pass the name since the path is stored internally
            results[opt_name] = self.benchmark_model(image_path, opt_name, num_runs=num_runs)

        return results

    def get_model_size(self, model) -> float:
        """Calculate model size in MB - handles both PyTorch models, ONNX files and TensorRT engines"""
        import tempfile

        # Check if it's a file path (ONNX or TensorRT engine)
        if isinstance(model, str):
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
            # For ONNX and TensorRT models, we just need to pass the path for size calculation
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
                    original_size = report['models_comparison']['original']['size_mb']
                    current_size = metrics['size_mb']
                    
                    # Avoid division by zero
                    size_improvement = original_size / current_size if current_size != 0 else 1

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
        elif optimization_type == "tensorrt":
            model = self.optimize_with_tensorrt(save_path=save_path or "model.trt")
        elif optimization_type == "auto":
            # Automatically select best optimization based on system
            optimized_model = self.auto_optimize()
            return optimized_model
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
        except ImportError:
            pass

        # Check if TensorRT is available
        try:
            import tensorrt
            import onnx
            strategy["tensorrt_available"] = True
            if self.gpu_available:
                strategy["recommended_optimization"] = "tensorrt"
        except ImportError:
            strategy["tensorrt_available"] = False

        # Determine best provider
        if strategy["tensorrt_available"] and self.gpu_available:
            strategy["best_provider"] = "TensorRT Native"
            strategy["recommended_optimization"] = "tensorrt"
        elif self.gpu_available and strategy.get("available_providers") and 'TensorrtExecutionProvider' in strategy["available_providers"]:
            strategy["best_provider"] = "TensorRTExecutionProvider"
            strategy["recommended_optimization"] = "onnx_tensorrt"
        elif self.gpu_available and strategy.get("available_providers") and 'CUDAExecutionProvider' in strategy["available_providers"]:
            strategy["best_provider"] = "CUDAExecutionProvider"
            strategy["recommended_optimization"] = "onnx_gpu"
        elif self.gpu_available:
            strategy["best_provider"] = "PyTorch+CuDNN"
            strategy["recommended_optimization"] = "torchscript"
        else:
            strategy["best_provider"] = "CPUExecutionProvider"
            strategy["recommended_optimization"] = "onnx_cpu"

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