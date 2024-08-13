import os
import torch

from utils import LOGGER
from typing import Optional, Union, List
from itertools import chain
from dotenv import load_dotenv
from dataset import load_dataset
from dataclasses import dataclass, field
from utils import (
    DefaultCollator,
    SimCSETrainer,
    load_loss
)
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed
    )
from peft import PeftModel, LoraConfig, get_peft_model
from models import LLM2Vec

from tqdm import tqdm

load_dotenv()



@dataclass
class ModelArguments:
    model_name: str
    peft_model_name_or_path: str
    device_map: str = field(default='auto')
    cache_dir: str = field(default='/data')
    torch_dtype: Optional[torch.dtype] = field(default=torch.bfloat16)
    trust_remote_code: bool = field(default=False)
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"choices": ["eager", "sdpa", "flash_attention_2"]})
    lora_dropout: float = field(default=0.05)
    bidirectional: bool = field(default=True)
    simcse_dropout: float = field(default=0.3)
    merge_peft: bool = field(default=True)
    pooling_mode: str = field(default="mean", metadata={"choices": ["mean", "eos_token", "bos_token"]})
    loss_scale: float = field(default=20.0)


@dataclass
class DataArguments:
    data_dir: str = field(default="data/")
    mlm_probability: float = field(default=0.15)
    max_seq_length: int = field(default=4096)
    validation_split_percentage: float = field(default=0.05)
    mask_token_type: str = field(default="blank")


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default='results/')
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=16)
    optim: str = field(default="adamw_torch")
    warmup_steps: int = field(default=800)
    lr_scheduler_type: Optional[str] = field(default='cosine')
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    logging_steps: int = field(default=10)
    logging_dir: str = field(default='logs/')
    gradient_accumulation_steps: int = field(default=4)
    model_max_length: int = field(default=512)
    max_grad_norm: float = field(default=1.0)
    save_strategy: str = field(default="epoch")
    save_steps: float = field(default=500)
    gradient_checkpointing: bool = field(default=True)
    report_to: Optional[str] = field(default='tensorboard')
    do_train: Optional[bool] = field(default=True)
    do_eval: Optional[bool] = field(default=False)
    evaluation_strategy: Optional[str] = None
    eval_steps: Optional[int] = None
    eval_accumulation_steps: None


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    LOGGER.info(f"Model name : {model_args.model_name} with LoRA")

    save_dir = model_args.model_name[model_args.model_name.rfind('/')+1:].lower()

    training_args.output_dir = os.path.join(training_args.output_dir, save_dir + '-simcse')
    training_args.logging_dir = os.path.join(training_args.logging_dir, save_dir + "-simcse")

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    ### Model
    model = LLM2Vec.from_pretrained(
            base_model_name_or_path=model_args.model_name,
            enable_bidirectional=model_args.bidirectional,
            peft_model_name_or_path=model_args.peft_model_name_or_path,
            merge_peft=True,
            pooling_mode=model_args.pooling_mode,
            torch_dtype=model_args.torch_dtype,
            attn_implementation=model_args.attn_implementation,
            attention_dropout=model_args.simcse_dropout,
            cache_dir=model_args.cache_dir
        )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=None,
        target_modules=['o_proj', 'q_proj', 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj'],
        
    )

    model.model = get_peft_model(model.model, lora_config)

    model.model.print_trainable_parameters()
    model.model.config.use_cache = False

    tokenizer = model.tokenizer

    train_loss = load_loss(model_args.loss_scale)

    ### Data
    data_path = ["/data/wiki1m_for_simcse.txt", "data/sim_cse/data.txt"]
    train_dataset = load_dataset(data_path)
    
    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
        )
    ]

    data_collator = DefaultCollator(model)

    trainer = SimCSETrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        loss_function=train_loss
        )
    
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train()