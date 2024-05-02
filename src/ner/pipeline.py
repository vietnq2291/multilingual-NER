from transformers import (
    AutoTokenizer,
    MT5Tokenizer,
    AutoModelForCausalLM,
    MT5ForConditionalGeneration,
)
import torch
import sys

from ner.data_formatter import DataFormatter
from ner.utils import get_pipe_config


class PipelineNER:
    def __init__(self, pipe_config_id, continue_from_model_id=None, usage="inference"):
        self.pipe_config_id = pipe_config_id
        self.usage = usage
        self._load_pipe_from_config(continue_from_model_id)
        self._setup_usage()

    def predict(self, inp, format_output=True):
        input_tokenized = self.tokenizer(inp, add_special_tokens=True)
        input_ids = (
            torch.tensor(input_tokenized["input_ids"]).unsqueeze(0).to(self.device)
        )
        attention_mask = (
            torch.tensor(input_tokenized["attention_mask"]).unsqueeze(0).to(self.device)
        )

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
        )[0]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        if format_output:
            output = self.data_formatter.format_output(output, self.model.config)
        return output

    def _load_pipe_from_config(self, continue_from_model_id):
        pipe_config = get_pipe_config(self.pipe_config_id, sys.modules[__name__])

        # Get the model_id based on the usage
        if self.usage == "train":
            # Finetune the `base_model_id`, then output the `model_id`
            model_id = pipe_config["base_model_id"]
        elif self.usage == "continue_train":
            # Continue finetuning from the `model_id`
            model_id = continue_from_model_id
        else:
            # Inference or evaluate
            model_id = pipe_config["model_id"]
        self.model = pipe_config["model_class"].from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = pipe_config["tokenizer_class"].from_pretrained(
            model_id, **pipe_config["tokenizer_configs"], trust_remote_code=True
        )
        self.data_formatter = DataFormatter(self.tokenizer, pipe_config["data_style"])  # data_style: the expected format of the input data for the model
        self.max_length = pipe_config["context_length"]

    def _setup_usage(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.usage == "inference" or self.usage == "evaluate":
            self.model.eval()
        elif self.usage == "train" or self.usage == "continue_train":
            self.model.train()
