# DeepSeek-OCR Inference Scripts

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/connectaman/deepseek-ocr-multigpu-infer)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)

Professional, production-ready Python scripts for running DeepSeek-OCR inference. This repository provides both single GPU and multi-GPU inference options to suit different hardware configurations and use cases.

**Repository**: [https://github.com/connectaman/deepseek-ocr-multigpu-infer](https://github.com/connectaman/deepseek-ocr-multigpu-infer)

## Scripts Available

### 1. Single GPU Inference (`deepseek_ocr_inference.py`)
- 🎯 **Single GPU**: Optimized for single GPU setups
- ⚡ **Fast Setup**: Quick model loading and processing
- 🔧 **Model Presets**: Built-in presets for different model sizes
- 📝 **Crop Mode**: Optional crop mode for better performance

### 2. Multi-GPU Inference (`deepseek_ocr_multigpu_inference.py`)
- 🚀 **Multi-GPU Support**: Automatically detects and utilizes all available CUDA GPUs
- 📁 **Parallel Processing**: Processes entire folders of images in parallel
- ⚖️ **Load Balancing**: Efficiently distributes work across GPUs
- 📊 **Scalable**: Scales with your hardware

## Common Features

- 📁 **Batch Processing**: Processes entire folders of images
- 🔧 **Configurable**: Customizable prompts, image sizes, and processing parameters
- 📊 **Progress Tracking**: Real-time logging and progress monitoring
- 📈 **Results Export**: Excel export of processing results and statistics
- 🛡️ **Error Handling**: Robust error handling with detailed logging
- 📝 **Professional Logging**: Clean, informative logging without experimental metrics

## Requirements

- Python 3.8+
- CUDA-compatible GPU(s)
- NVIDIA drivers and CUDA toolkit

## Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/connectaman/deepseek-ocr-multigpu-infer.git
   cd deepseek-ocr-multigpu-infer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Single GPU Inference

#### Basic Usage
```bash
python deepseek_ocr_inference.py input_folder output_folder
```

#### Advanced Usage
```bash
python deepseek_ocr_inference.py ./images ./results \
    --prompt "Convert this document to markdown" \
    --base-size 1024 \
    --image-size 640 \
    --crop-mode \
    --gpu-id 0 \
    --results-file my_results.xlsx
```

#### Model Size Presets
```bash
# Tiny model (fastest, least accurate)
python deepseek_ocr_inference.py input output --base-size 512 --image-size 512 --no-crop-mode

# Small model
python deepseek_ocr_inference.py input output --base-size 640 --image-size 640 --no-crop-mode

# Base model (default)
python deepseek_ocr_inference.py input output --base-size 1024 --image-size 1024 --no-crop-mode

# Large model (most accurate, slowest)
python deepseek_ocr_inference.py input output --base-size 1280 --image-size 1280 --no-crop-mode

# Gundam model (balanced)
python deepseek_ocr_inference.py input output --base-size 1024 --image-size 640 --crop-mode
```

### Multi-GPU Inference

#### Basic Usage
```bash
python deepseek_ocr_multigpu_inference.py input_folder output_folder
```

#### Advanced Usage
```bash
python deepseek_ocr_multigpu_inference.py ./images ./results \
    --prompt "Convert this document to markdown" \
    --base-size 1024 \
    --image-size 1280 \
    --results-file multigpu_results.xlsx
```

### Command Line Arguments

#### Single GPU Script (`deepseek_ocr_inference.py`)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_folder` | ✅ | - | Path to folder containing input images |
| `output_folder` | ✅ | - | Path to folder for output markdown files |
| `--prompt` | ❌ | `"<image>\n<|grounding|>Convert the document to markdown. "` | Custom prompt for OCR model |
| `--base-size` | ❌ | `1024` | Base size parameter for model |
| `--image-size` | ❌ | `640` | Image size parameter for model |
| `--crop-mode` | ❌ | `False` | Enable crop mode for processing |
| `--gpu-id` | ❌ | `0` | GPU device ID to use |
| `--results-file` | ❌ | `single_gpu_inference_results.xlsx` | Excel file for processing results |

#### Multi-GPU Script (`deepseek_ocr_multigpu_inference.py`)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_folder` | ✅ | - | Path to folder containing input images |
| `output_folder` | ✅ | - | Path to folder for output markdown files |
| `--prompt` | ❌ | `"<image>\n<|grounding|>Convert the document to markdown. "` | Custom prompt for OCR model |
| `--base-size` | ❌ | `1024` | Base size parameter for model |
| `--image-size` | ❌ | `1280` | Image size parameter for model |
| `--results-file` | ❌ | `multigpu_inference_results.xlsx` | Excel file for processing results |

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## GPU Monitoring

### Install GPU Monitoring Tool

```bash
pip install nvitop
```

### Monitor GPU Usage

```bash
nvitop
```

This will show real-time GPU utilization, memory usage, and temperature for all available GPUs.

#### GPU Monitoring Screenshot

![GPU Monitoring with nvitop](screenshot/gpu.png)

*Example of nvitop showing GPU utilization during DeepSeek-OCR inference across multiple GPUs*

## Example Workflow

### Single GPU Workflow

1. **Prepare your images**
   ```bash
   mkdir input_images
   # Copy your images to input_images/
   ```

2. **Run single GPU inference**
   ```bash
   python deepseek_ocr_inference.py input_images output_markdowns
   ```

3. **Monitor progress**
   - Watch the console output for real-time progress
   - Use `nvitop` in another terminal to monitor GPU usage

4. **Check results**
   - Markdown files will be saved in `output_markdowns/`
   - Processing results will be saved in `single_gpu_inference_results.xlsx`

### Multi-GPU Workflow

1. **Prepare your images**
   ```bash
   mkdir input_images
   # Copy your images to input_images/
   ```

2. **Run multi-GPU inference**
   ```bash
   python deepseek_ocr_multigpu_inference.py input_images output_markdowns
   ```

3. **Monitor progress**
   - Watch the console output for real-time progress
   - Use `nvitop` in another terminal to monitor GPU usage across all GPUs

4. **Check results**
   - Markdown files will be saved in `output_markdowns/`
   - Processing results will be saved in `multigpu_inference_results.xlsx`

## Output Structure

### Markdown Files
Each input image generates a corresponding markdown file:
```
input_images/
├── document1.jpg
├── document2.png
└── document3.tiff

output_markdowns/
├── document1.md
├── document2.md
└── document3.md
```

### Results Excel File
The Excel file contains processing metadata:
- `filename`: Original image filename
- `markdown_filename`: Generated markdown filename
- `gpu_id`: GPU that processed the image
- `gpu_name`: Name of the GPU used
- `status`: Processing status (success/error)
- `error`: Error message (if applicable)

## Performance Tips

### Single GPU Optimization
1. **Model Size**: Choose appropriate model size based on your accuracy vs speed requirements
2. **Crop Mode**: Enable crop mode for better performance on smaller images
3. **GPU Selection**: Use `--gpu-id` to select the most powerful GPU if you have multiple
4. **Memory Management**: Monitor GPU memory usage with `nvitop`

### Multi-GPU Optimization
1. **Load Balancing**: The script automatically distributes work evenly across GPUs
2. **GPU Memory**: Ensure sufficient GPU memory on all GPUs for your batch size
3. **Image Size**: Larger images require more memory but may provide better OCR results
4. **Monitoring**: Use `nvitop` to monitor GPU utilization across all GPUs

### General Tips
1. **Batch Processing**: Process images in batches to optimize memory usage
2. **Image Formats**: Use compressed formats (JPEG) for faster loading
3. **Storage**: Use SSD storage for faster image loading

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--image-size` parameter
   - Process fewer images simultaneously
   - Check available GPU memory with `nvidia-smi`

2. **No Images Found**
   - Verify input folder path
   - Check supported image formats
   - Ensure images are not in subdirectories

3. **Model Loading Errors**
   - Verify internet connection for model download
   - Check CUDA installation
   - Ensure sufficient disk space for model cache

### Debug Mode

For detailed debugging, you can modify the logging level in the script:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Open an issue on the repository

---

**Note**: This script requires CUDA-compatible GPUs and the DeepSeek-OCR model. Make sure your system meets the hardware requirements before running.
