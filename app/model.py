import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import io
import logging
import base64

# Import rich for better logging
from rich.console import Console
from rich.logging import RichHandler
import rich

console = Console()
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

# Add the 3rd party directory to the Python path to access GroundingDINO and SAM
third_party_path = os.path.join(os.path.dirname(__file__), "3rd_party/Grounded-Segment-Anything")
sys.path.insert(0, third_party_path)

# Also add the individual subdirectories to the path
grounding_dino_path = os.path.join(third_party_path, "GroundingDINO")
sam_path = os.path.join(third_party_path, "segment_anything")
sys.path.insert(0, grounding_dino_path)
sys.path.insert(0, sam_path)

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import GroundingDINO.groundingdino.datasets.transforms as T

# segment anything - fix import path
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


class GroundedSAMPredictor:
    """
    A class that encapsulates the Grounding DINO and SAM models for text-prompted segmentation.
    """
    
    def __init__(self, 
                 grounding_dino_config_path: str,
                 grounding_dino_checkpoint_path: str,
                 sam_checkpoint_path: str,
                 sam_hq_checkpoint_path: str = None,
                 sam_version: str = "vit_h",
                 device: str = "cpu",
                 bert_base_uncased_path: str = None,
                 use_sam_hq: bool = False,
                 save_results_dir: str = "./outputs"):
        """
        Initialize the predictor with model paths and settings.
        """
        self.device = device
        self.use_sam_hq = use_sam_hq
        self.save_results_dir = save_results_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.save_results_dir, exist_ok=True)
        
        # Load models
        self.grounding_model = self._load_grounding_model(
            grounding_dino_config_path, 
            grounding_dino_checkpoint_path, 
            bert_base_uncased_path
        )
        
        self.sam_predictor = self._load_sam_model(
            sam_checkpoint_path, 
            sam_hq_checkpoint_path, 
            sam_version
        )
    
    def _load_grounding_model(self, config_path, checkpoint_path, bert_base_uncased_path):
        """Load the Grounding DINO model."""
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        args.bert_base_uncased_path = bert_base_uncased_path
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        logging.info(f"Grounding DINO model loaded with results: {load_res}")
        _ = model.eval()
        return model.to(self.device)
    
    def _load_sam_model(self, sam_checkpoint_path, sam_hq_checkpoint_path, sam_version):
        """Load the SAM model."""
        if self.use_sam_hq and sam_hq_checkpoint_path:
            sam_model = sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint_path).to(self.device)
        else:
            sam_model = sam_model_registry[sam_version](checkpoint=sam_checkpoint_path).to(self.device)
        
        return SamPredictor(sam_model)
    
    def preprocess_image(self, input_image):
        """
        Preprocess the input image for the Grounding DINO model.
        
        Args:
            input_image: Either a PIL Image, a file path, or bytes
            
        Returns:
            Tuple of (original_pil_image, processed_tensor)
        """
        if isinstance(input_image, str):
            # If input is a file path
            try:
                image_pil = Image.open(input_image).convert("RGB")
            except Exception as e:
                console.print(f"[red]Error opening image file {input_image}: {e}[/red]")
                raise
        elif isinstance(input_image, bytes):
            # If input is bytes
            image_pil = Image.open(io.BytesIO(input_image)).convert("RGB")
        elif isinstance(input_image, Image.Image):
            # If input is already a PIL Image
            image_pil = input_image.convert("RGB")  # Ensure RGB format
        else:
            raise ValueError("Input image must be a file path (str), bytes, or PIL Image")
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    
    def predict_grounded(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25, with_logits=True):
        """
        Run the Grounding DINO model to detect objects based on text prompt.
        
        Args:
            image: Preprocessed image tensor
            text_prompt: Text prompt to guide detection
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text
            with_logits: Whether to include confidence scores in phrases
            
        Returns:
            Tuple of (boxes, phrases)
        """
        caption = text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
            
        image = image.to(self.device)
        self.grounding_model = self.grounding_model.to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # Get phrase
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        # Build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
    
    def predict_sam(self, image_pil, boxes_filt):
        """
        Use the SAM model to segment based on the filtered boxes.
        
        Args:
            image_pil: Original PIL image
            boxes_filt: Filtered boxes from Grounding DINO
            
        Returns:
            Tuple of (masks, boxes)
        """
        # Convert PIL image to opencv format
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Set image for predictor
        self.sam_predictor.set_image(image_np)
        
        # Process boxes
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_np.shape[:2]
        ).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        
        return masks, boxes_filt
    
    def predict(self, input_image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Full prediction pipeline: preprocess image -> detect objects -> segment objects.
        
        Args:
            input_image: Input image (PIL, path, or bytes)
            text_prompt: Text prompt to guide detection
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text
            
        Returns:
            Dictionary with masks, boxes, and phrases
        """
        # Preprocess image
        image_pil, image = self.preprocess_image(input_image)
        
        # Detect objects
        boxes_filt, pred_phrases = self.predict_grounded(
            image, text_prompt, box_threshold, text_threshold
        )
        
        # Segment objects
        masks, boxes_filt = self.predict_sam(image_pil, boxes_filt)
        
        return {
            "masks": masks,
            "boxes": boxes_filt,
            "phrases": pred_phrases,
            "image_pil": image_pil
        }


class ImageSegmenter:
    """
    Wrapper class for image segmentation, adapted for PySide6 desktop application
    """
    def __init__(self):
        # Set up paths to the Grounded-SAM models
        grounding_dino_config_path = os.path.join(os.path.dirname(__file__), "3rd_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        grounding_dino_checkpoint_path = os.path.join(os.path.dirname(__file__), "3rd_party/Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
        sam_checkpoint_path = os.path.join(os.path.dirname(__file__), "3rd_party/Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
        
        try:
            self.segmenter = GroundedSAMPredictor(
                grounding_dino_config_path=grounding_dino_config_path,
                grounding_dino_checkpoint_path=grounding_dino_checkpoint_path,
                sam_checkpoint_path=sam_checkpoint_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                save_results_dir=os.path.join(os.path.dirname(__file__), "infer_results")  # 设置默认保存目录
            )
            console.print("[bold green]Segmenter initialized successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error initializing segmenter: {e}[/bold red]")
            console.print("[yellow]Using simulation mode for demonstration[/yellow]")
            self.segmenter = None

    def segment(self, image_path, text_prompt="object"):
        """
        Segment objects in the image based on the text prompt.
        
        Args:
            image_path: Path to the input image
            text_prompt: Text prompt to guide segmentation
            
        Returns:
            Dictionary with segmentation results
        """
        if self.segmenter is not None:
            try:
                # Run the full prediction pipeline
                results_dict = self.segmenter.predict(
                    input_image=image_path,
                    text_prompt=text_prompt
                )
                
                masks = results_dict["masks"]
                boxes_filt = results_dict["boxes"]
                pred_phrases = results_dict["phrases"]
                
                # Format results to match the expected structure
                formatted_results = {
                    "num_objects_detected": len(pred_phrases),
                    "objects": self._format_objects(boxes_filt, pred_phrases),
                    "masks": self._create_mask_data(pred_phrases, boxes_filt),
                    # 保留原始的masks和boxes用于可视化
                    "raw_masks": masks,
                    "raw_boxes": boxes_filt,
                    "phrases": pred_phrases
                }
                
                return formatted_results
            except Exception as e:
                console.print(f"[bold red]Error during segmentation: {e}[/bold red]")
                # Return a simulated response
                return self._simulate_segmentation()
        else:
            # Simulation mode for demonstration
            return self._simulate_segmentation()

    def _format_objects(self, boxes_filt, pred_phrases):
        """
        Format detected objects into a standardized structure.
        """
        objects = []
        for box, label in zip(boxes_filt, pred_phrases):
            logit_value = None
            obj_label = label
            
            if '(' in label and ')' in label:
                obj_label, logit_str = label.split('(')
                logit_value = float(logit_str[:-1])  # Remove closing parenthesis
                
            objects.append({
                "label": obj_label.strip(),
                "confidence": logit_value,
                "bbox": box.tolist()
            })
        
        return objects

    def _create_mask_data(self, pred_phrases, boxes_filt):
        """
        Creates mask data in the format needed for the API response.
        """
        mask_results = [{"value": 0, "label": "background"}]
        
        for label, box in zip(pred_phrases, boxes_filt):
            name, logit = label.split('(')
            logit = logit[:-1]  # Remove closing parenthesis
            mask_results.append({
                "value": len(mask_results),
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            })
        
        return mask_results

    def _simulate_segmentation(self):
        """
        Simulate segmentation results for demonstration purposes.
        """
        return {
            "num_objects_detected": 1,
            "objects": [
                {
                    "label": "simulated_object",
                    "confidence": 0.8,
                    "bbox": [10, 10, 100, 100]
                }
            ],
            "masks": [
                {"value": 0, "label": "background"},
                {"value": 1, "label": "simulated_object", "logit": 0.8, "box": [10, 10, 100, 100]}
            ],
            "raw_masks": torch.zeros((1, 1, 100, 100)),  # 模拟的mask张量
            "raw_boxes": torch.tensor([[10, 10, 100, 100]]),  # 模拟的boxes张量
            "phrases": ["simulated_object(0.80)"]  # 模拟的短语
        }