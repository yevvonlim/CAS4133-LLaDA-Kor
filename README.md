# Large Language Diffusion Models
[![arXiv](https://img.shields.io/badge/arXiv-2502.09992-red.svg)](https://arxiv.org/abs/2502.09992)
[![deploy](https://img.shields.io/badge/Hugging%20Face%20-LLaDA_Base%20-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
[![deploy](https://img.shields.io/badge/Hugging%20Face%20-LLaDA_Instruct%20-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
[![deploy](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Spaces%20demo%20-blue)](https://huggingface.co/spaces/multimodalart/LLaDA)
[![deploy](https://img.shields.io/badge/Zhihu-知乎-blue)](https://zhuanlan.zhihu.com/p/24214732238)

We introduce LLaDA (<b>L</b>arge <b>La</b>nguage <b>D</b>iffusion with m<b>A</b>sking), a diffusion model with an unprecedented 8B scale, trained entirely from scratch, 
rivaling LLaMA3 8B in performance.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## News
- [2024.05] We have provided evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base.
- [2024.02] We have uploaded our paper to [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).


## LLaDA Training (@sionic-ai we provide 😎)

### 1. Environment Setup

#### 1.1 Docker Container

1. Pull and run the Docker image with GPU support:

   ```bash
   docker pull pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
   docker run --gpus all -it \
     -v $(pwd):/workspace/LLaDA \
     --workdir /workspace/LLaDA \
     pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel bash
   ```

2. Inside the container, create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install project dependencies and synchronize:

   ```bash
   pip install uv
   uv sync
   ```

---

### 2. Training Script (`train.sh`)

The training is orchestrated by `train.sh`. It utilizes `torchrun` for distributed training across 8 GPUs on a single node.

```bash
#!/bin/bash

torchrun \
  --nproc_per_node 8 \                       # Number of GPUs per node
  --nnodes 1 \                                # Number of nodes
  -m train.finetune \                         # Entry point module
  --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \  # Pretrained model to fine-tune
  --data_path "/workspace/LLaDA/ko-gpt3_14k/data_train.jsonl" \  # Path to training data
  --bf16 True \                               # Use bfloat16 precision
  --output_dir "llada_kor" \                 # Directory for checkpoints and logs
  --num_train_epochs 5 \                      # Total number of training epochs
  --per_device_train_batch_size 8 \           # Batch size per GPU for training
  --per_device_eval_batch_size 1 \            # Batch size per GPU for evaluation
  --gradient_accumulation_steps 8 \           # Steps to accumulate gradients before updating
  --save_strategy "steps" \                  # Save checkpoints every N steps
  --save_steps 1000 \                         # Number of steps between checkpoint saves
  --save_total_limit 10 \                     # Maximum number of checkpoints to keep
  --learning_rate 1e-5 \                      # Initial learning rate
  --weight_decay 0.1 \                        # Weight decay for optimizer
  --adam_beta2 0.95 \                         # Adam optimizer beta2 parameter
  --warmup_ratio 0.01 \                       # Warmup as a fraction of total steps
  --lr_scheduler_type "cosine" \             # Learning rate scheduler type
  --logging_steps 10 \                        # Log metrics every N steps
  --report_to "wandb" \                      # Report metrics to Weights & Biases
  --model_max_length 512 \                    # Maximum sequence length
  --gradient_checkpointing False \            # Toggle gradient checkpointing
  --use_lora \                                # Enable LoRA parameter-efficient fine-tuning
  --deepspeed deepspeed/zero2.json \          # DeepSpeed config for Zero2 optimization
  --label_names prompt_lengths                 # Input field names used as labels
```

#### 2.1 Argument Explanations

* `--model_name_or_path`
  Path or identifier of the pretrained model to fine-tune (e.g., Hugging Face repo).

* `--data_path`
  Location of the training dataset in JSONL format.

* `--bf16 True`
  Enable bfloat16 mixed-precision training for faster compute and reduced memory usage.

* `--output_dir`
  Directory where model checkpoints, logs, and outputs will be stored.

* `--num_train_epochs`
  Number of passes over the entire training dataset.

* `--per_device_train_batch_size`
  Batch size for each GPU during training.

* `--per_device_eval_batch_size`
  Batch size for each GPU during evaluation.

* `--gradient_accumulation_steps`
  Number of forward/backward passes before performing an optimizer step, effectively increasing the batch size.

* `--save_strategy` (`"steps"`)
  Strategy for checkpointing. With `steps`, checkpoints are saved every `--save_steps` steps.

* `--save_steps`
  Interval (in steps) at which to save model checkpoints.

* `--save_total_limit`
  Maximum number of checkpoints to keep; older ones will be deleted.

* `--learning_rate`
  Learning rate for the optimizer.

* `--weight_decay`
  L2 regularization factor.

* `--adam_beta2`
  Second beta coefficient for the Adam optimizer.

* `--warmup_ratio`
  Fraction of total training steps to linearly increase the learning rate before decaying.

* `--lr_scheduler_type` (`"cosine"`)
  Type of learning rate scheduler; cosine decay in this case.

* `--logging_steps`
  Frequency (in steps) at which training metrics are logged.

* `--report_to`
  Destination for logging and experiment tracking (e.g., `wandb`, `tensorboard`).

* `--model_max_length`
  Maximum sequence length (number of tokens) the model will process.

* `--gradient_checkpointing`
  If `True`, enable checkpointing to trade compute for memory.

* `--use_lora`
  Enable Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

* `--deepspeed`
  Path to DeepSpeed JSON configuration file (e.g., for ZeRO optimizations).

* `--label_names`
  Names of input fields in the dataset to be treated as labels (e.g., sequence lengths).

---

### 3. Usage

1. Ensure you are in the project root and the container is running with the virtual environment activated.

2. Make `train.sh` executable:

   ```bash
   chmod +x train.sh
   ```

3. Launch training:

   ```bash
   ./train.sh
   ```

---

Happy training! 🚀



## Inference
The [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) are uploaded
in Huggingface. Please first install `transformers==4.38.2` and employ the [transformers](https://huggingface.co/docs/transformers/index) to load.

```angular2html
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

We provide `get_log_likelihood()` and `generate()` functions in `get_log_likelihood.py` 
and `generate.py` respectively, for conditional likelihood evaluation and conditional generation.

You can directly run `python chat.py` to have multi-round conversations with LLaDA-8B-Instruct.

In addition, please refer to our paper and [GUIDELINES.md](GUIDELINES.md) for more details about the inference methods.

## Gradio demo 
Thank you very much to [apolinário](https://github.com/apolinario) for helping us create this amazing demo!

First, install [Gradio](https://www.gradio.app) `pip install gradio`, and then you can directly run `python app.py`

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Pre-training and Supervised Fine-Tuning

We will not provide the training framework and data as most open-source LLMs do.

However, the pre-training and Supervised Fine-Tuning of LLaDA are straightforward. If 
you have a codebase for training an autoregressive model, you can modify it to 
adapt to LLaDA with just a few lines of code.

We provide guidelines for the pre-training and SFT of LLaDA in [GUIDELINES.md](GUIDELINES.md). 
You can also refer to [SMDM](https://github.com/ML-GSAI/SMDM), which has a similar training process to LLaDA 
and has open-sourced the training framework.

## Evaluation

We use two evaluation methods: conditional likelihood estimation and conditional generation. For the base model, conditional likelihood estimation is applied to specific metrics and conditional generation to the rest. For the Instruct model, conditional generation is used for all metrics.

In our [paper](https://arxiv.org/abs/2502.09992), we implement conditional likelihood estimation using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library, while conditional generation is performed with an internal library, as lm-evaluation-harness lacks support for certain metrics (i.e., HumanEval-FIM). Please refer to Appendix B.5. of our [paper](https://arxiv.org/abs/2502.09992) for all evaluation details.

**2024.05.04.** We now provide evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base, including conditional likelihood estimation and conditional generation. However, for the Instruct model (both LLama3-Instruct and LLaDA-Instruct), we encountered some bugs. We are still debugging the issue. 

Please refer to [EVAL.md](EVAL.md) for the usage of the evaluation code and the BUG details. Any help in resolving this bug would be sincerely appreciated.

## FAQ
Here, we address some common questions about LLaDA.

### 0. How do I train my own LLaDA?
Please refer to [GUIDELINES.md](GUIDELINES.md) for the guidelines. 
You can also refer to [SMDM](https://github.com/ML-GSAI/SMDM), which follows the same training 
process as LLaDA and has open-sourced its code.


### 1. What is the difference between LLaDA and BERT?

Our motivation is not to improve BERT, nor to apply image generation methods like [MaskGIT](https://arxiv.org/abs/2202.04200) 
to text. **Our goal is to explore a theoretically complete language modeling approach — masked diffusion models.** 
During this process, we simplified the approach and discovered that the loss function of masked diffusion models 
is related to the loss functions of BERT and MaskGIT. You can find our theoretical research process in Question 7.

Specifically, LLaDA employs a masking ratio that varies randomly between 0 and 1, while BERT uses 
a fixed ratio. This subtle difference has significant implications. **The training
objective of LLaDA is an upper bound on the negative log-likelihood of the model 
distribution, making LLaDA a generative model.** This enables LLaDA to naturally 
perform in-context learning, instruction-following, and ensures Fisher consistency 
for scalability with large datasets and models. You can also find a direct answer 
to this question in Section 2.1 of our paper.


### 2. What is the relationship between LLaDA and Transformer?
Network structure and probabilistic modeling are two distinct approaches that collectively form the 
foundation of language models. LLaDA, like GPT, adopts the 
Transformer architecture. The key difference lies in the probabilistic modeling approach: GPT 
utilizes an autoregressive next-token prediction method, 
while LLaDA employs a diffusion model for probabilistic modeling.


### 3. What is the sampling efficiency of LLaDA?
Currently, LLaDA's sampling speed is slower than the autoregressive baseline for three reasons: 
1. LLaDA samples with a fixed context length;
2. LLaDA cannot yet leverage techniques like KV-Cache;
3. LLaDA achieves optimal performance when the number of sampling steps equals the response length.
Reducing the number of sampling steps leads to a decrease in performance, as detailed in Appendix B.4 
and Appendix B.6 of our paper.

In this work, we aim to explore the upper limits of LLaDA's capabilities, **challenging the assumption 
that the key LLM abilities are inherently tied to autoregressive models**. We will continue 
to optimize its efficiency in the future. We believe this research approach is reasonable, 
as verifying the upper limits of diffusion language models' capabilities will provide us with
more resources and sufficient motivation to optimize efficiency.

Recall the development of diffusion models for images, from [DDPM](https://arxiv.org/abs/2006.11239) 
to the [Consistency model](https://arxiv.org/pdf/2410.11081), where sampling speed accelerated nearly 
1000 times over the course of 4 years. **We believe there is significant room for optimization in LLaDA's 
sampling efficiency as well**. Current solutions, including semi-autoregressive sampling (as 
detailed in [GUIDELINES.md](GUIDELINES.md)), can mitigate the fixed context length issue, and 
[consistency distillation](https://arxiv.org/pdf/2502.05415) can reduce the number of sampling steps.


### 4. What is the training stability of LLaDA?
For details on the pre-training process of LLaDA, please refer to Section 2.2 of our paper. 
During the total pre-training on 2.3T tokens, we encountered a training crash (loss becoming NaN) 
only once at 1.2T tokens. Our solution was to resume checkpoint and reduce 
the learning rate from 4e-4 to 1e-4.


### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

**The mask predictor has successfully predicted the reasoning process. However, during the 
remasking process, the reasoning steps are masked out again.** As shown in the figure 
below, the non-white background represents the model's generation process, while the 
white-background boxes indicate the predictions made by the mask predictor at each step. 
We adopt a randomly remasking strategy.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/diff_remask.gif" style="width: 80%" />
</div>

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?
This is because our pre-training and SFT data were designed for training an autoregressive model, 
whereas LLaDA directly utilizes data that contains identity markers.


### 7. Our journey in developing LLaDA?
LLaDA is built upon our two prior works, [RADD](https://arxiv.org/abs/2406.03736) and 
[SMDM](https://arxiv.org/abs/2410.18514). 

RADD demonstrated that the **training objective of LLaDA serves as an upper bound on the negative 
log-likelihood** of the model’s distribution, a conclusion also supported by [MD4](https://arxiv.org/abs/2406.04329) 
and [MDLM](https://arxiv.org/abs/2406.07524). 
Furthermore, RADD was the first to theoretically prove that **masked diffusion models do not require time t 
as an input to Transformer**. This insight provides the theoretical 
justification for LLaDA’s unmodified use of the Transformer architecture. Lastly, 
RADD showed that **the training objective of masked diffusion models is equivalent to that of 
any-order autoregressive models**, offering valuable insights into how masked diffusion models can 
overcome the reversal curse.

SMDM introduces the first **scaling law** for masked diffusion models and demonstrates that, with the 
same model size and training data, masked diffusion models can achieve downstream benchmark results 
on par with those of autoregressive models. Additionally, SMDM presents a simple, **unsupervised 
classifier-free guidance** method that greatly improves downstream benchmark performance, which has 
been adopted by LLaDA.


## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

