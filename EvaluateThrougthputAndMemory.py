import os
os.environ['HF_DATASETS_CACHE'] = '/scratch-shared/'
os.environ['HF_TOKENIZERS_CACHE'] = '/scratch-shared/tokenizes'
os.environ['HF_HOME'] = '/scratch-shared/HF_HOME'
os.environ['HF_METRICS_CACHE'] = '/scratch-shared/metrics'
os.environ['HF_MODULES_CACHE'] = '/scratch-shared/modules'
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import argparse
import gc
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Measure GPU memory and throughput of LLM inference")
parser.add_argument('--model_name', type=str, default="gpt2", help="Name of the model to use")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
parser.add_argument('--max_length', type=int, default=1024, help="Batch size for inference")
parser.add_argument('--num_repeats', type=int, default=500, help="Number of times to repeat the inference for averaging")
args = parser.parse_args()

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Adjust for speed test for 2k input.
if args.max_length >= 1024:
    tokenizer.model_max_length = 2048
    config = model.config
    config.n_positions = 2048
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True).cuda()
    position_embeddings = model.transformer.wpe
    current_max_len = position_embeddings.weight.size(0)
    if 2048 > current_max_len:
        extra_positions = torch.nn.Parameter(torch.randn(2048 - current_max_len, model.config.n_embd).cuda())
        position_embeddings.data = torch.cat([position_embeddings.data, extra_positions], dim=0)    
    model.transformer.wpe = position_embeddings

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Load a sample of the Wiki dataset
# dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]",trust_remote_code=True)

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

# encoded_dataset = dataset.map(preprocess_function, batched=True)

input_text = "This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word. This is an unlimited word."

input_data = {"text": [input_text]}
encoded_dataset = preprocess_function(input_data)

# Function to get GPU memory usage
def get_gpu_memory_usage():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], 
        encoding="utf-8"
    )
    gpu_memory = int(result.strip().split('\n')[0])
    return gpu_memory

def model_inference(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)

# sample_batch = encoded_dataset.select(range(args.batch_size))
# inputs = {key: torch.tensor(val).cuda() for key, val in sample_batch.to_dict().items() if key in tokenizer.model_input_names}

inputs = {key: torch.tensor(val).cuda() for key, val in encoded_dataset.items() if key in tokenizer.model_input_names}

print(inputs)
for key, val in inputs.items():
    print(f"{key}: {val.shape}")

memory_usages = []
inference_times = []
memory_usages_before=[]
#
test_cuda=[]
test_cuda_before=[]
# Repeat the measurement
for _ in range(args.num_repeats):
    torch.cuda.synchronize()  
    #before_memory_allocated = torch.cuda.memory_allocated()

    start_time = time.time()
    with torch.no_grad():
        output = model(**inputs)
    end_time = time.time()

    torch.cuda.synchronize()  
    after_memory_allocated = torch.cuda.max_memory_allocated()
    memory_usages.append(after_memory_allocated)
    #memory_usages_before.append(before_memory_allocated)
    inference_time = end_time - start_time
    inference_times.append(inference_time)

    del output
    torch.cuda.empty_cache()  # Clear the cache to get more accurate measurements
    gc.collect()

# Calculate averages
average_memory_usage = np.mean(memory_usages)
#average_memory_usage_before=np.mean(memory_usages_before)
average_inference_time = np.mean(inference_times)
throughput = args.batch_size / average_inference_time

print(f"Average memory used during inference: {average_memory_usage/1024**2} MB")
print(f"Average inference time: {average_inference_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} inferences/second")