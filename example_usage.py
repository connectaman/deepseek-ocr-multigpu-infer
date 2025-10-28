#!/usr/bin/env python3
"""
Example usage script for DeepSeek-OCR Inference Scripts

This script demonstrates how to use both the single GPU and multi-GPU
inference classes programmatically in your own Python applications.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepseek_ocr_inference import DeepSeekOCRInference
from deepseek_ocr_multigpu_inference import DeepSeekOCRMultiGPUInference


def single_gpu_example():
    """Example usage of the single GPU DeepSeekOCRInference class."""
    
    print("=== Single GPU Inference Example ===")
    
    # Initialize the single GPU inference handler
    print("Initializing DeepSeek-OCR Single GPU Inference...")
    inference_handler = DeepSeekOCRInference(gpu_id=0)
    
    # Define paths
    input_folder = "input_images"
    output_folder = "output_markdowns_single"
    
    # Create example input folder if it doesn't exist
    os.makedirs(input_folder, exist_ok=True)
    
    # Check if input folder has images
    if not any(Path(input_folder).glob("*")):
        print(f"Please add some images to the '{input_folder}' folder and run again.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp")
        return False
    
    # Custom prompt for OCR
    custom_prompt = "<image>\n<|grounding|>Convert this document to markdown format, preserving the structure and layout."
    
    # Process images
    print(f"Processing images from '{input_folder}' to '{output_folder}'...")
    
    try:
        results_df = inference_handler.process_images(
            input_folder=input_folder,
            output_folder=output_folder,
            prompt=custom_prompt,
            base_size=1024,
            image_size=640,
            crop_mode=True
        )
        
        # Save results
        results_file = "single_gpu_example_results.xlsx"
        results_df.to_excel(results_file, index=False)
        
        print(f"\n✅ Single GPU processing completed successfully!")
        print(f"📁 Markdown files saved to: {output_folder}")
        print(f"📊 Results saved to: {results_file}")
        
        # Display summary
        if not results_df.empty:
            success_count = len(results_df[results_df['status'] == 'success'])
            error_count = len(results_df[results_df['status'] != 'success'])
            
            print(f"\n📈 Summary:")
            print(f"   Successfully processed: {success_count} images")
            if error_count > 0:
                print(f"   Failed to process: {error_count} images")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during single GPU processing: {str(e)}")
        return False


def multi_gpu_example():
    """Example usage of the multi-GPU DeepSeekOCRMultiGPUInference class."""
    
    print("\n=== Multi-GPU Inference Example ===")
    
    # Initialize the multi-GPU inference handler
    print("Initializing DeepSeek-OCR Multi-GPU Inference...")
    inference_handler = DeepSeekOCRMultiGPUInference()
    
    # Define paths
    input_folder = "input_images"
    output_folder = "output_markdowns_multi"
    
    # Create example input folder if it doesn't exist
    os.makedirs(input_folder, exist_ok=True)
    
    # Check if input folder has images
    if not any(Path(input_folder).glob("*")):
        print(f"Please add some images to the '{input_folder}' folder and run again.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp")
        return False
    
    # Custom prompt for OCR
    custom_prompt = "<image>\n<|grounding|>Convert this document to markdown format, preserving the structure and layout."
    
    # Process images
    print(f"Processing images from '{input_folder}' to '{output_folder}'...")
    
    try:
        results_df = inference_handler.process_images(
            input_folder=input_folder,
            output_folder=output_folder,
            prompt=custom_prompt,
            base_size=1024,
            image_size=1280
        )
        
        # Save results
        results_file = "multi_gpu_example_results.xlsx"
        results_df.to_excel(results_file, index=False)
        
        print(f"\n✅ Multi-GPU processing completed successfully!")
        print(f"📁 Markdown files saved to: {output_folder}")
        print(f"📊 Results saved to: {results_file}")
        
        # Display summary
        if not results_df.empty:
            success_count = len(results_df[results_df['status'] == 'success'])
            error_count = len(results_df[results_df['status'] != 'success'])
            
            print(f"\n📈 Summary:")
            print(f"   Successfully processed: {success_count} images")
            if error_count > 0:
                print(f"   Failed to process: {error_count} images")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during multi-GPU processing: {str(e)}")
        return False


def main():
    """Main function demonstrating both single GPU and multi-GPU usage."""
    
    print("DeepSeek-OCR Inference Examples")
    print("===============================")
    
    # Check if input folder has images
    input_folder = "input_images"
    if not any(Path(input_folder).glob("*")):
        print(f"Please add some images to the '{input_folder}' folder and run again.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp")
        return 1
    
    # Run single GPU example
    single_success = single_gpu_example()
    
    # Run multi-GPU example
    multi_success = multi_gpu_example()
    
    # Summary
    print(f"\n{'='*50}")
    print("Example Summary:")
    print(f"Single GPU: {'✅ Success' if single_success else '❌ Failed'}")
    print(f"Multi-GPU:  {'✅ Success' if multi_success else '❌ Failed'}")
    
    if not (single_success or multi_success):
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
