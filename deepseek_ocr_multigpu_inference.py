#!/usr/bin/env python3
"""
DeepSeek-OCR Multi-GPU Inference Script
=======================================

A professional, production-ready script for running DeepSeek-OCR inference
across multiple GPUs in parallel. Automatically detects available GPUs and
distributes image processing tasks efficiently.

Author: DeepSeek-OCR Team
License: MIT
"""

import argparse
import logging
import os
import sys
import time
import torch
import pandas as pd
import glob
import io
from pathlib import Path
from typing import List, Dict, Any
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing import Process, Manager

from transformers import AutoModel, AutoTokenizer
import transformers

# Suppress transformers logging
transformers.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DeepSeekOCRMultiGPUInference:
    """
    Multi-GPU DeepSeek-OCR inference handler.
    
    This class manages the distribution of image processing tasks across
    multiple GPUs and handles the inference pipeline.
    """
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR"):
        """
        Initialize the DeepSeek-OCR inference handler.
        
        Args:
            model_name: The Hugging Face model identifier
        """
        self.model_name = model_name
        self.num_gpus = torch.cuda.device_count()
        
        if self.num_gpus == 0:
            raise RuntimeError("No CUDA GPUs detected. This script requires GPU support.")
        
        logger.info(f"Detected {self.num_gpus} GPU(s) available for inference")
    
    def get_image_files(self, input_folder: str) -> List[str]:
        """
        Get all supported image files from the input folder.
        
        Args:
            input_folder: Path to the input folder containing images
            
        Returns:
            List of image file paths
        """
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        image_files = sorted(image_files)
        
        if not image_files:
            raise ValueError(f"No supported image files found in {input_folder}")
        
        logger.info(f"Found {len(image_files)} image files to process")
        return image_files
    
    def get_gpu_info(self, device_id: int) -> tuple:
        """
        Get GPU information for a specific device.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Tuple of (gpu_name, gpu_max_memory_gb, gpu_used_gb)
        """
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_id)
            gpu_name = props.name
            gpu_max_memory = props.total_memory / (1024 ** 3)
            gpu_used = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            return gpu_name, gpu_max_memory, gpu_used
        return "N/A", 0, 0
    
    def run_inference_on_gpu(self, gpu_id: int, image_files: List[str], 
                           shared_results: List[Dict], output_folder: str,
                           prompt: str, base_size: int, image_size: int) -> None:
        """
        Run inference on a specific GPU for a subset of images.
        
        Args:
            gpu_id: GPU device ID
            image_files: List of image files to process on this GPU
            shared_results: Shared results list for multiprocessing
            output_folder: Output folder for markdown files
            prompt: Prompt for the OCR model
            base_size: Base size parameter for the model
            image_size: Image size parameter for the model
        """
        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Suppress transformers logging in subprocess
        import transformers
        transformers.logging.set_verbosity_error()
        
        device = torch.device("cuda:0")
        
        logger.info(f"[GPU {gpu_id}] Initializing model...")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                self.model_name,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            ).to(device=device, dtype=torch.bfloat16).eval()
            
            # Get GPU info
            gpu_name, gpu_max_memory, _ = self.get_gpu_info(0)
            logger.info(f"[GPU {gpu_id}] Model loaded on {gpu_name} ({gpu_max_memory:.1f} GB)")
            
            # Process images
            for i, image_file in enumerate(image_files, 1):
                filename = os.path.basename(image_file)
                markdown_filename = os.path.splitext(filename)[0] + ".md"
                output_path = os.path.join(output_folder, markdown_filename)
                
                logger.info(f"[GPU {gpu_id}] Processing {filename} ({i}/{len(image_files)})")
                
                try:
                    # Run inference with suppressed output
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        result = model.infer(
                            tokenizer,
                            prompt=prompt,
                            image_file=image_file,
                            output_path=output_path,
                            base_size=base_size,
                            image_size=image_size,
                            crop_mode=False,
                            save_results=True,
                            test_compress=True
                        )
                    
                    logger.info(f"[GPU {gpu_id}] ✓ Successfully processed {filename}")
                    
                    # Record successful processing
                    shared_results.append({
                        "filename": filename,
                        "markdown_filename": markdown_filename,
                        "gpu_id": gpu_id,
                        "gpu_name": gpu_name,
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"[GPU {gpu_id}] ✗ Error processing {filename}: {str(e)}")
                    
                    # Record failed processing
                    shared_results.append({
                        "filename": filename,
                        "markdown_filename": markdown_filename,
                        "gpu_id": gpu_id,
                        "gpu_name": gpu_name,
                        "status": "error",
                        "error": str(e)
                    })
            
            logger.info(f"[GPU {gpu_id}] Completed processing {len(image_files)} images")
            
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Failed to initialize model: {str(e)}")
            # Record GPU failure
            for image_file in image_files:
                filename = os.path.basename(image_file)
                markdown_filename = os.path.splitext(filename)[0] + ".md"
                shared_results.append({
                    "filename": filename,
                    "markdown_filename": markdown_filename,
                    "gpu_id": gpu_id,
                    "gpu_name": "Unknown",
                    "status": "gpu_error",
                    "error": str(e)
                })
    
    def process_images(self, input_folder: str, output_folder: str, 
                      prompt: str = "<image>\n<|grounding|>Convert the document to markdown. ",
                      base_size: int = 1024, image_size: int = 1280) -> pd.DataFrame:
        """
        Process all images in the input folder using multi-GPU inference.
        
        Args:
            input_folder: Path to input folder containing images
            output_folder: Path to output folder for markdown files
            prompt: Prompt for the OCR model
            base_size: Base size parameter for the model
            image_size: Image size parameter for the model
            
        Returns:
            DataFrame with processing results
        """
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get image files
        image_files = self.get_image_files(input_folder)
        
        # Split images among GPUs
        image_splits = [image_files[i::self.num_gpus] for i in range(self.num_gpus)]
        
        # Log distribution
        for gpu_id, images in enumerate(image_splits):
            logger.info(f"GPU {gpu_id} will process {len(images)} images")
        
        # Shared results for multiprocessing
        manager = Manager()
        shared_results = manager.list()
        
        # Launch processes
        processes = []
        start_time = time.time()
        
        logger.info("Starting multi-GPU inference...")
        
        for gpu_id in range(self.num_gpus):
            if image_splits[gpu_id]:  # Only start process if there are images to process
                p = Process(
                    target=self.run_inference_on_gpu,
                    args=(gpu_id, image_splits[gpu_id], shared_results, 
                          output_folder, prompt, base_size, image_size)
                )
                p.start()
                processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Create results DataFrame
        df = pd.DataFrame(list(shared_results))
        
        # Log summary
        if not df.empty:
            success_count = len(df[df['status'] == 'success'])
            error_count = len(df[df['status'] != 'success'])
            
            logger.info(f"Processing completed in {total_time:.2f} seconds")
            logger.info(f"Successfully processed: {success_count} images")
            if error_count > 0:
                logger.warning(f"Failed to process: {error_count} images")
        else:
            logger.warning("No results to process")
        
        return df


def main():
    """Main function to run the DeepSeek-OCR multi-GPU inference script."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR Multi-GPU Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deepseek_ocr_multigpu_inference.py input_folder output_folder
  python deepseek_ocr_multigpu_inference.py ./images ./results --base-size 1024 --image-size 1280
  python deepseek_ocr_multigpu_inference.py ./input ./output --prompt "Convert this document to markdown"
        """
    )
    
    parser.add_argument(
        "input_folder",
        help="Path to the input folder containing images"
    )
    
    parser.add_argument(
        "output_folder", 
        help="Path to the output folder for markdown files"
    )
    
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown. ",
        help="Prompt for the OCR model (default: Convert document to markdown)"
    )
    
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Base size parameter for the model (default: 1024)"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=1280,
        help="Image size parameter for the model (default: 1280)"
    )
    
    parser.add_argument(
        "--results-file",
        default="multigpu_inference_results.xlsx",
        help="Output Excel file for processing results (default: multigpu_inference_results.xlsx)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize inference handler
        inference_handler = DeepSeekOCRMultiGPUInference()
        
        # Process images
        results_df = inference_handler.process_images(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            prompt=args.prompt,
            base_size=args.base_size,
            image_size=args.image_size
        )
        
        # Save results
        if not results_df.empty:
            results_df.to_excel(args.results_file, index=False)
            logger.info(f"Results saved to: {args.results_file}")
        
        logger.info("Multi-GPU inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
