from transformers import (
    MT5ForConditionalGeneration,
    MT5TokenizerFast,
)
from ner.utils import get_pipe_config
from ner.data_formatter import DataFormatter

import torch
import sys


class PipelineNER:

    def __init__(self, pipe_config_id, usage="inference"):
        self.pipe_config_id = pipe_config_id
        self.usage = usage
        self._load_pipe_from_config()
        self._setup_usage()

    def predict(self, raw_input, format_output=True):
        input_tokenized = self.tokenizer(raw_input, add_special_tokens=True)
        input_ids = torch.tensor(input_tokenized["input_ids"]).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(input_tokenized["attention_mask"]).unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
        )[0]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        if format_output:
            output = self.data_formatter.format_output(output)
        return output
    
    def _load_pipe_from_config(self):
        pipe_config = get_pipe_config(self.pipe_config_id, sys.modules[__name__])

        model_id = pipe_config["model_id"]
        base_model_id = pipe_config["base_model_id"]

        self.model = pipe_config["model_class"].from_pretrained(
            model_id, trust_remote_code=True
        )
        self.tokenizer = pipe_config["tokenizer_class"].from_pretrained(
            base_model_id, **pipe_config["tokenizer_configs"], trust_remote_code=True
        )

        self.data_formatter = DataFormatter()
        self.max_length = pipe_config["context_length"]

    def _setup_usage(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
