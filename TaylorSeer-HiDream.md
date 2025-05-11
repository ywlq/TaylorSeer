## TaylorSeer-HiDream

### 1. Set Up Conda Environment

```bash
# First install Python 3.12
# Then install the requirements
pip install -r requirements.txt
```

### 2. Download Model Checkpoints

If you experience connection issues with Hugging Face, you can use the Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

#### Download HiDream Models

```bash
# Full version
huggingface-cli download --resume-download HiDream-ai/HiDream-I1-Full \
                         --local-dir HiDream-ai/HiDream-I1-Full

# Dev version
huggingface-cli download --resume-download HiDream-ai/HiDream-I1-Dev \
                         --local-dir HiDream-ai/HiDream-I1-Dev

# Fast version
huggingface-cli download --resume-download HiDream-ai/HiDream-I1-Fast \
                         --local-dir HiDream-ai/HiDream-I1-Fast
```

#### Download Llama-3.1-8B

HiDream requires Llama-3.1-8B, which you need to request access for. Once you have the necessary permissions, you can download it using one of these methods:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download --resume-download meta-llama/Llama-3.1-8B \
                         --local-dir meta-llama/Llama-3.1-8B

# Option 2: Using modelscope
modelscope download --model LLM-Research/Meta-Llama-3.1-8B
```

### 3. Quick Start

```bash
# For full model inference
python ./inference.py --model_type full

# For distilled dev model inference
python ./inference.py --model_type dev

# For distilled fast model inference
python ./inference.py --model_type fast

```

### 4. Gradio Demo

```bash
python gradio_demo.py 
```

### 5. Sampling with TaylorSeer-HiDream

We've developed a custom script to evaluate the image generation quality metrics. Use the following command to run sampling tests:

```bash
python sampling.py --prompt_file </path/to/your/test/prompt.txt> \
                   --output_dir </path/to/your/generated/samples/folder> \
                   --add_sampling_metadata 
```

#### Performance Evaluation

We tested TaylorSeer-HiDream on the DrawBench200 benchmark. Here's an example of how to run the test:

```bash
python sampling.py --prompt_file /path/to/your/prompts/DrawBench200.txt \
                   --output_dir /path/to/your/generated/samples/folder \
                   --add_sampling_metadata
```

#### Results

TaylorSeer-HiDream demonstrates significant performance improvements:

The following results are based on evaluations conducted on an **H20 device**:

| Model              | Generation Time | DrawBench200 ImageReward |
| ------------------ | --------------- | ------------------ |
| Original           | 76s /img   | 1.184              |
| TaylorSeer-HiDream | 21s /img   | 1.106              |

TaylorSeer optimization reduces generation time by approximately 72% (from 76s to 21s per image) while maintaining comparable quality metrics on the DrawBench200 benchmark.



### 6. Start with TaylorSeer-HiDream-I1-Fast-nf4

This section explains how to launch TaylorSeer-HiDream using the nf4 quantized model versions. These models are optimized for reduced memory and computation requirements, making them suitable for resource-constrained environments.(A 24GB GPU should be sufficient to run it.)

#### Download nf4 Weights

```bash
# Dev version
huggingface-cli download --resume-download azaneko/HiDream-I1-Dev-nf4 --local-dir /root/autodl-tmp/pretrained_models/azaneko/HiDream-I1-Dev-nf4

# Fast version
huggingface-cli download --resume-download azaneko/HiDream-I1-Fast-nf4 --local-dir /root/autodl-tmp/pretrained_models/azaneko/HiDream-I1-Fast-nf4

# Full version
huggingface-cli download --resume-download azaneko/HiDream-I1-Full-nf4 --local-dir /root/autodl-tmp/pretrained_models/azaneko/HiDream-I1-Full-nf4

#Download Quantized LLaMA Model (INT4)
huggingface-cli download --resume-download hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --local-dir /root/autodl-tmp/pretrained_models/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
```

#### Install hdi1 Inference Package
The nf4 models require the hdi1 inference library, which depends on flash-attn. If you've already installed the base requirements.txt for HiDream, it's recommended to reuse the same Conda environment and install hdi1 as follows:
```bash
pip install hdi1 --no-build-isolation
```
Note: This may recompile parts of flash-attn, so ensure your environment is properly configured.

#### Quick Inference Example
You can quickly generate images using the CLI provided by hdi1. Here’s a sample command:
```bash
python -m hdi1 "A cat holding a sign that says 'hello world'" -m fast
```
The -m fast flag specifies the HiDream-I1-Fast-nf4 model.

Replace the prompt with your own text to generate different images.


