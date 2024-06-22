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

    def predict(self, format_output=True, format_prompt=False, **kwargs):
        if format_prompt:
            raw_input = self.data_formatter.format_instruction_input(
                kwargs["text"], kwargs["entity_type"]
            )
        else:
            raw_input = kwargs["prompt"]

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
  
    def evaluate(self, eval_dataset):
        # Eval dataset must have 3 fields: "text", "entity_type", "label"
        print(f'Evaluating dataset: {eval_dataset} ...')

        # Make predictions
        preds = eval_dataset.map(lambda x: {'pred': self.predict(**x, format_prompt=True)})['pred']
        labels = eval_dataset['label']

        # Calculate metrics
        f1_score = self.calculate_f1(preds, labels)
        return {
            'f1_score': f1_score,
        }


    def calculate_f1(self, preds, labels):
        set_preds = [set(pred) for pred in preds]
        set_labels = [set(label) for label in labels]

        tp = sum([len(pred & label) for pred, label in zip(set_preds, set_labels)])
        fp = sum([len(pred - label) for pred, label in zip(set_preds, set_labels)])
        fn = sum([len(label - pred) for pred, label in zip(set_preds, set_labels)])

        print(f'TP: {tp}, FP: {fp}, FN: {fn}')

        precision = tp / (tp + fp)
        print(f'Precision: {precision}')
        recall = tp / (tp + fn)
        print(f'Recall: {recall}')
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f'F1 Score: {f1_score}')

        return f1_score

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
