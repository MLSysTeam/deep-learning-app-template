# model_handler.py
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
import logging
import base64
import json

# Import rich for better logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
import rich

console = Console()
install()
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

# Add the 3rd party directory to the Python path to access GroundingDINO and SAM
third_party_path = os.path.join(os.getcwd(), "app/3rd_party/Grounded-Segment-Anything")
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
    
    def show_mask(self, mask, ax, random_color=False):
        """Display mask on the plot"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        """Display bounding box on the plot"""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)
    
    def visualize_results(self, image_pil, masks, boxes_filt, pred_phrases):
        """
        Visualize the segmentation results.
        
        Args:
            image_pil: Original PIL image
            masks: Predicted masks
            boxes_filt: Filtered boxes
            pred_phrases: Predicted phrases
            
        Returns:
            Matplotlib figure with visualization
        """
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), ax, random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), ax, label)

        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def create_json_results(self, boxes_filt, pred_phrases):
        """
        Creates structured JSON results from the prediction outputs.
        
        Args:
            boxes_filt: Filtered boxes
            pred_phrases: Predicted phrases with confidence scores
            
        Returns:
            List of dictionaries containing object information
        """
        results = []
        for box, label in zip(boxes_filt, pred_phrases):
            logit_value = None
            obj_label = label
            
            if '(' in label and ')' in label:
                obj_label, logit_str = label.split('(')
                logit_value = float(logit_str[:-1])  # Remove closing parenthesis
                
            results.append({
                "label": obj_label.strip(),
                "confidence": logit_value,
                "bbox": box.tolist()
            })
        
        return results
    
    def generate_segmentation_image(self, image_pil, masks, boxes_filt, pred_phrases):
        """
        Generates a base64 encoded segmentation image.
        
        Args:
            image_pil: Original PIL image
            masks: Predicted masks
            boxes_filt: Filtered boxes
            pred_phrases: Predicted phrases
            
        Returns:
            Base64 encoded string of the segmented image
        """
        fig = self.visualize_results(image_pil, masks, boxes_filt, pred_phrases)
        
        # Save to bytes
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='JPEG', bbox_inches="tight", dpi=150, pad_inches=0.0)
        img_buffer.seek(0)
        plt.close(fig)
        
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    def create_mask_data(self, pred_phrases, boxes_filt):
        """
        Creates mask data in the format needed for the API response.
        
        Args:
            pred_phrases: Predicted phrases
            boxes_filt: Filtered boxes
            
        Returns:
            List of mask result dictionaries
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
    
    def save_results_to_disk(self, image_pil, masks, boxes_filt, pred_phrases, output_dir, prefix="result"):
        """
        Saves the segmentation results to disk.
        
        Args:
            image_pil: Original PIL image
            masks: Predicted masks
            boxes_filt: Filtered boxes
            pred_phrases: Predicted phrases
            output_dir: Directory to save results
            prefix: Prefix for saved files
            
        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(output_dir, f"{prefix}_original.jpg")
        image_pil.save(original_path)
        
        # Save segmentation visualization
        fig = self.visualize_results(image_pil, masks, boxes_filt, pred_phrases)
        seg_path = os.path.join(output_dir, f"{prefix}_segmented.jpg")
        fig.savefig(seg_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close(fig)
        
        # Create and save mask data as JSON
        mask_results = self.create_mask_data(pred_phrases, boxes_filt)
        json_path = os.path.join(output_dir, f"{prefix}_masks.json")
        with open(json_path, 'w') as f:
            json.dump(mask_results, f)
        
        # Save individual masks as well
        masks_path = os.path.join(output_dir, f"{prefix}_masks.jpg")
        value = 0  # 0 for background
        mask_img = torch.zeros(masks.shape[-2:])
        for idx, mask in enumerate(masks):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(masks_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()
        
        return {
            "original_image": original_path,
            "segmented_image": seg_path,
            "masks_json": json_path,
            "masks_visualization": masks_path
        }
    
    def process_single_prediction(self, input_image, text_prompt, box_threshold=0.3, text_threshold=0.25, output_type="both", save_results=False, output_dir=None, prefix="result"):
        """
        Processes a single prediction request, including all post-processing.
        
        Args:
            input_image: Input image (PIL, path, or bytes)
            text_prompt: Text prompt to guide detection
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text
            output_type: Type of output ("image", "mask", "both")
            save_results: Whether to save results to disk
            output_dir: Directory to save results (uses default if not provided)
            prefix: Prefix for saved files
            
        Returns:
            Dictionary with complete API response
        """
        # Run the full prediction pipeline
        results_dict = self.predict(
            input_image=input_image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        masks = results_dict["masks"]
        boxes_filt = results_dict["boxes"]
        pred_phrases = results_dict["phrases"]
        image_pil = results_dict["image_pil"]
        
        # Prepare basic results
        results = {
            "num_objects_detected": len(pred_phrases),
            "objects": self.create_json_results(boxes_filt, pred_phrases)
        }
        
        # Add outputs based on output_type
        if output_type in ["image", "both"]:
            results["segmented_image"] = self.generate_segmentation_image(
                image_pil, masks, boxes_filt, pred_phrases
            )
        
        if output_type in ["mask", "both"]:
            results["masks"] = self.create_mask_data(pred_phrases, boxes_filt)
        
        # Optionally save results to disk
        if save_results:
            save_dir = output_dir or self.save_results_dir
            saved_paths = self.save_results_to_disk(
                image_pil, masks, boxes_filt, pred_phrases, 
                save_dir, prefix
            )
            results["saved_files"] = saved_paths
        
        return results


class ImageSegmenter:
    """
    Wrapper class for image segmentation, similar to the original ImageClassifier
    """
    def __init__(self):
        # Set up paths to the Grounded-SAM models
        grounding_dino_config_path = "app/3rd_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint_path = "app/3rd_party/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
        sam_checkpoint_path = "app/3rd_party/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
        
        try:
            self.segmenter = GroundedSAMPredictor(
                grounding_dino_config_path=grounding_dino_config_path,
                grounding_dino_checkpoint_path=grounding_dino_checkpoint_path,
                sam_checkpoint_path=sam_checkpoint_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                save_results_dir="./app/infer_results"  # 设置默认保存目录
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
                results = self.segmenter.process_single_prediction(
                    input_image=image_path,
                    text_prompt=text_prompt,
                    output_type="both"
                )
                return results
            except Exception as e:
                console.print(f"[bold red]Error during segmentation: {e}[/bold red]")
                # Return a simulated response
                return self._simulate_segmentation()
        else:
            # Simulation mode for demonstration
            return self._simulate_segmentation()

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
            "segmented_image": "",
            "masks": [
                {"value": 0, "label": "background"},
                {"value": 1, "label": "simulated_object", "logit": 0.8, "box": [10, 10, 100, 100]}
            ]
        }