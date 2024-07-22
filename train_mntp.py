import os
import torch

from utils import LOGGER
from typing import Optional
from itertools import chain
from dotenv import load_dotenv
from datasets import load_dataset
from dataclasses import dataclass, field
from utils import (
    DataCollatorForLanguageModelingWithFullMasking,
    DataCollatorForLanguageModeling,
    MNTPTrainer
)
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    HfArgumentParser,
    )
from peft import LoraConfig, get_peft_model
from models import MistralBiForMNTP, LlamaBiForMNTP


load_dotenv()


def get_model_class(config):
    config_class_name = config.__class__.__name__
    if config_class_name == "MistralConfig":
        return MistralBiForMNTP
    elif config_class_name == "LlamaConfig":
        return LlamaBiForMNTP
    else:
        raise ValueError()


@dataclass
class ModelArguments:
    model_name: str
    device_map: str = field(default='auto')
    cache_dir: str = field(default='/data')
    torch_dtype: Optional[torch.dtype] = field(default=torch.bfloat16)
    trust_remote_code: bool = field(default=False)
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"choices": ["eager", "sdpa", "flash_attention_2"]})


@dataclass
class DataArguments:
    data_dir: str = field(default="data/")
    data_path: str = field(default="/data/wiki1m_for_simcse.txt")
    mlm_probability: float = field(default=0.15)
    max_seq_length: int = field(default=4096)
    validation_split_percentage: float = field(default=0.05)
    mask_token_type: str = field(default="blank")


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default='results/')
    num_train_epochs: int = field(default=10)
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
    model_max_length: int = field(default=4096)
    max_grad_norm: float = field(default=1.0)
    save_strategy: str = field(default="steps")
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

    training_args.output_dir = os.path.join(training_args.output_dir, save_dir + '-mntp')
    training_args.logging_dir = os.path.join(training_args.logging_dir, save_dir + "-mntp")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    ### Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name,
            padding_side="right",
            use_fast=False,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir
        )
    if tokenizer.mask_token is None:
        if data_args.mask_token_type == "blank":
            tokenizer.mask_token = "_"
        elif data_args.mask_token_type == "eos":
            tokenizer.mask_token = tokenizer.eos_token
        elif data_args.mask_token_type == "mask":
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ### Model
    config = AutoConfig.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code
        )
    model_class = get_model_class(config)

    model = model_class.from_pretrained(
        model_args.model_name,
        config=config,
        device_map = model_args.device_map,
        torch_dtype = model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        attn_implementation=model_args.attn_implementation
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.3,
        bias="none",
        task_type=None,
        target_modules=['o_proj', 'q_proj', 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj'],
        
    )

    model.model = get_peft_model(model.model, lora_config)

    model.model.print_trainable_parameters()
    model.config.use_cache = False

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    ### Data
    raw_datasets = load_dataset("text", data_files=data_args.data_path, split='train')
    LOGGER.info(f"Line of Dataset : {len(raw_datasets)}")

    padding = "max_length"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    def group_texts(examples):
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # `max_seq_length`보다 작은 경우 마지막 청크를 삭제
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        
        return result
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    if training_args.do_eval:
        tokenized_datasets = tokenized_datasets['train'].train_test_split(test_size=0.2)
        train_dataset = tokenized_datasets['train']
        eval_dataset = tokenized_datasets['test']
    
    # Llama : BERT(20%) / Mistral : RoBERTa (80%)
    if 'llama' in model_args.model_name.lower():
        data_collator_cls = DataCollatorForLanguageModeling
        mlm_probability = 0.2
    elif 'mistral' in model_args.model_name.lower():
        data_collator_cls = DataCollatorForLanguageModelingWithFullMasking
        mlm_probability = 0.8
    else:
        raise ValueError
    
    data_collator = data_collator_cls(tokenizer=tokenizer, mlm_probability=mlm_probability)

    trainer = MNTPTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset if training_args.do_eval else tokenized_datasets,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator
        )
    
    trainer.train()
    trainer.save_model()
    trainer.save_state()

if __name__ == "__main__":
    train()