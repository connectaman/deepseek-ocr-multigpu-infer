#!/usr/bin/env python3
"""
DeepSeek-OCR Single GPU Inference Script
========================================

A professional, production-ready script for running DeepSeek-OCR inference
on a single GPU. Optimized for single GPU setups with comprehensive logging
and error handling.

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


class DeepSeekOCRInference:
    """
    Single GPU DeepSeek-OCR inference handler.
    
    This class manages the OCR inference pipeline on a single GPU
    with comprehensive error handling and logging.
    """
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR", gpu_id: int = 0):
        """
        Initialize the DeepSeek-OCR inference handler.
        
        Args:
            model_name: The Hugging Face model identifier
            gpu_id: GPU device ID to use (default: 0)
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires GPU support.")
        
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA GPUs detected. This script requires GPU support.")
        
        # Get GPU info
        self.gpu_name, self.gpu_max_memory, _ = self.get_gpu_info(0)
        logger.info(f"Using GPU {gpu_id}: {self.gpu_name} ({self.gpu_max_memory:.1f} GB)")
        
        # Initialize model and tokenizer
        self._load_model()
    
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
    
    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info("Loading model and tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
            
            # Move model to GPU and set to evaluation mode
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            
            logger.info("✓ Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
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
    
    def process_single_image(self, image_file: str, output_folder: str, 
                           prompt: str, base_size: int, image_size: int, 
                           crop_mode: bool = True) -> Dict[str, Any]:
        """
        Process a single image file.
        
        Args:
            image_file: Path to the image file
            output_folder: Output folder for markdown files
            prompt: Prompt for the OCR model
            base_size: Base size parameter for the model
            image_size: Image size parameter for the model
            crop_mode: Whether to use crop mode
            
        Returns:
            Dictionary with processing results
        """
        filename = os.path.basename(image_file)
        markdown_filename = os.path.splitext(filename)[0] + ".md"
        output_path = os.path.join(output_folder, markdown_filename)
        
        logger.info(f"Processing {filename}")
        
        start_time = time.time()
        
        try:
            # Run inference with suppressed output
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_file,
                    output_path=output_path,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
                    test_compress=True
                )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"✓ Successfully processed {filename} in {processing_time:.2f}s")
            
            return {
                "filename": filename,
                "markdown_filename": markdown_filename,
                "gpu_id": self.gpu_id,
                "gpu_name": self.gpu_name,
                "status": "success",
                "processing_time": processing_time
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.error(f"✗ Error processing {filename}: {str(e)}")
            
            return {
                "filename": filename,
                "markdown_filename": markdown_filename,
                "gpu_id": self.gpu_id,
                "gpu_name": self.gpu_name,
                "status": "error",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def process_images(self, input_folder: str, output_folder: str, 
                      prompt: str = "<image>\n<|grounding|>Convert the document to markdown. ",
                      base_size: int = 1024, image_size: int = 640, 
                      crop_mode: bool = True, num_processes: int = 1) -> pd.DataFrame:
        """
        Process all images in the input folder using single GPU inference.
        
        Args:
            input_folder: Path to input folder containing images
            output_folder: Path to output folder for markdown files
            prompt: Prompt for the OCR model
            base_size: Base size parameter for the model
            image_size: Image size parameter for the model
            crop_mode: Whether to use crop mode
            num_processes: Number of processes to run on the GPU (1-2, default: 1)
            
        Returns:
            DataFrame with processing results
        """
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get image files
        image_files = self.get_image_files(input_folder)
        
        # Validate num_processes
        if num_processes < 1 or num_processes > 2:
            raise ValueError("num_processes must be between 1 and 2")
        
        if num_processes == 1:
            # Single process mode (original behavior)
            results = []
            start_time = time.time()
            
            logger.info(f"Starting single GPU inference on {self.gpu_name}...")
            
            for i, image_file in enumerate(image_files, 1):
                logger.info(f"Processing image {i}/{len(image_files)}")
                
                result = self.process_single_image(
                    image_file=image_file,
                    output_folder=output_folder,
                    prompt=prompt,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode
                )
                
                results.append(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
        else:
            # Multi-process mode
            logger.info(f"Starting single GPU multi-process inference on {self.gpu_name} with {num_processes} processes...")
            
            # Split images among processes
            image_splits = [image_files[i::num_processes] for i in range(num_processes)]
            
            # Shared results for multiprocessing
            manager = Manager()
            shared_results = manager.list()
            
            # Launch processes
            processes = []
            start_time = time.time()
            
            for proc_id in range(num_processes):
                if image_splits[proc_id]:  # Only start process if there are images to process
                    p = Process(
                        target=self.run_inference_process,
                        args=(proc_id, image_splits[proc_id], shared_results, 
                              output_folder, prompt, base_size, image_size, crop_mode)
                    )
                    p.start()
                    processes.append(p)
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
            
            end_time = time.time()
            total_time = end_time - start_time
            results = list(shared_results)
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
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
    
    def run_inference_process(self, proc_id: int, image_files: List[str], 
                            shared_results: List[Dict], output_folder: str,
                            prompt: str, base_size: int, image_size: int, 
                            crop_mode: bool) -> None:
        """
        Run inference in a separate process for multi-process mode.
        
        Args:
            proc_id: Process ID
            image_files: List of image files to process
            shared_results: Shared results list for multiprocessing
            output_folder: Output folder for markdown files
            prompt: Prompt for the OCR model
            base_size: Base size parameter for the model
            image_size: Image size parameter for the model
            crop_mode: Whether to use crop mode
        """
        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        # Suppress transformers logging in subprocess
        import transformers
        transformers.logging.set_verbosity_error()
        
        device = torch.device("cuda:0")
        
        logger.info(f"[Process {proc_id}] Initializing model...")
        
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
            logger.info(f"[Process {proc_id}] Model loaded on {gpu_name} ({gpu_max_memory:.1f} GB)")
            
            # Process images
            for i, image_file in enumerate(image_files, 1):
                filename = os.path.basename(image_file)
                markdown_filename = os.path.splitext(filename)[0] + ".md"
                output_path = os.path.join(output_folder, markdown_filename)
                
                logger.info(f"[Process {proc_id}] Processing {filename} ({i}/{len(image_files)})")
                
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
                            crop_mode=crop_mode,
                            save_results=True,
                            test_compress=True
                        )
                    
                    logger.info(f"[Process {proc_id}] ✓ Successfully processed {filename}")
                    
                    # Record successful processing
                    shared_results.append({
                        "filename": filename,
                        "markdown_filename": markdown_filename,
                        "gpu_id": self.gpu_id,
                        "process_id": proc_id,
                        "gpu_name": gpu_name,
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"[Process {proc_id}] ✗ Error processing {filename}: {str(e)}")
                    
                    # Record failed processing
                    shared_results.append({
                        "filename": filename,
                        "markdown_filename": markdown_filename,
                        "gpu_id": self.gpu_id,
                        "process_id": proc_id,
                        "gpu_name": gpu_name,
                        "status": "error",
                        "error": str(e)
                    })
            
            logger.info(f"[Process {proc_id}] Completed processing {len(image_files)} images")
            
        except Exception as e:
            logger.error(f"[Process {proc_id}] Failed to initialize model: {str(e)}")
            # Record process failure
            for image_file in image_files:
                filename = os.path.basename(image_file)
                markdown_filename = os.path.splitext(filename)[0] + ".md"
                shared_results.append({
                    "filename": filename,
                    "markdown_filename": markdown_filename,
                    "gpu_id": self.gpu_id,
                    "process_id": proc_id,
                    "gpu_name": "Unknown",
                    "status": "process_error",
                    "error": str(e)
                })


def main():
    """Main function to run the DeepSeek-OCR single GPU inference script."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR Single GPU Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deepseek_ocr_inference.py input_folder output_folder
  python deepseek_ocr_inference.py ./images ./results --base-size 1024 --image-size 640
  python deepseek_ocr_inference.py ./input ./output --prompt "Convert this document to markdown"
  python deepseek_ocr_inference.py ./input ./output --gpu-id 1 --crop-mode

Model Size Presets:
  Tiny:   --base-size 512 --image-size 512 --no-crop-mode
  Small:  --base-size 640 --image-size 640 --no-crop-mode  
  Base:   --base-size 1024 --image-size 1024 --no-crop-mode
  Large:  --base-size 1280 --image-size 1280 --no-crop-mode
  Gundam: --base-size 1024 --image-size 640 --crop-mode
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
        default=640,
        help="Image size parameter for the model (default: 640)"
    )
    
    parser.add_argument(
        "--crop-mode",
        action="store_true",
        help="Enable crop mode for processing (default: False)"
    )
    
    parser.add_argument(
        "--no-crop-mode",
        dest="crop_mode",
        action="store_false",
        help="Disable crop mode for processing"
    )
    
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of processes to run on the GPU (1-2, default: 1)"
    )
    
    parser.add_argument(
        "--results-file",
        default="single_gpu_inference_results.xlsx",
        help="Output Excel file for processing results (default: single_gpu_inference_results.xlsx)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize inference handler
        inference_handler = DeepSeekOCRInference(gpu_id=args.gpu_id)
        
        # Process images
        results_df = inference_handler.process_images(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            prompt=args.prompt,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode,
            num_processes=args.num_processes
        )
        
        # Save results
        if not results_df.empty:
            results_df.to_excel(args.results_file, index=False)
            logger.info(f"Results saved to: {args.results_file}")
        
        logger.info("Single GPU inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
