import logging

IGNORE_INDEX = -100

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()],
    level=logging.INFO
)

logger_name = "MNTP-Train"

LOGGER = logging.getLogger(logger_name)

from .train_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from .collator import DataCollatorForLanguageModelingWithFullMasking, DataCollatorForLanguageModeling, DefaultCollator
from .trainers import MNTPTrainer