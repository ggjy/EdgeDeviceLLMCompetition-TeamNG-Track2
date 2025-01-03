## Edge-Device Large Language Model Competition

Track2 TeamNG's final submission 

### Submission Checklist

#### Model Checkpoint

The config file can be found in [code](https://github.com/ggjy/EdgeDeviceLLMCompetition-TeamNG-Track2/tree/main/code). 

The Huggingface checkpoint can be found [here](https://github.com/ggjy/EdgeDeviceLLMCompetition-TeamNG-Track2/releases/download/Model-CKPT/model.safetensors).

#### Converted MLC Model File

Converted MLC model files can be found [here](https://github.com/ggjy/EdgeDeviceLLMCompetition-TeamNG-Track2/releases/download/mlc-file/gpt2_pruned-q0f16-MLC.rar).

#### Result CSV File

```
pip install opencompass==0.3.1

opencompass --datasets commonsenseqa_7shot_cot_gen_734a22 FewCLUE_chid_gen humaneval_gen gsm8k_gen truthfulqa_gen bbh_gen  --hf-type base --hf-path ./EdgeDeviceLLMCompetition-TeamNG-Track2 --model-kwargs device_map='auto' trust_remote_code=True --max-out-len 100 --max-seq-len 512 -r latest --max-num-workers 8
```

Check results.csv [here](https://github.com/ggjy/EdgeDeviceLLMCompetition-TeamNG-Track2/blob/main/results.csv).

### Training data

This model is trained from scratch on [C4](https://huggingface.co/datasets/legacy-datasets/c4) dataset.

### Speed Test

Our GPT2-based architectuire is trained on wpe=1024. You can refer to this [code](https://github.com/ggjy/EdgeDeviceLLMCompetition-TeamNG-Track2/blob/main/EvaluateThrougthputAndMemory.py#L28) to adapt for 2k input length.

### Environment

- pytorch 2.3.0+cu121
- transformers >= 4.40.0
- datasets
- tokenizers
- accelerate
