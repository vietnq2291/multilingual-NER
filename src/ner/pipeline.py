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
            if self.usage == "evaluate":
                # Format prompt for each sample in the evaluation batch
                raw_input = [self.data_formatter.format_instruction_input(text, entity_type) for text, entity_type in zip(kwargs["samples"]["text"], kwargs["samples"]["entity_type"])]
            else:
                raw_input = self.data_formatter.format_instruction_input(
                    kwargs["text"], kwargs["entity_type"]
                )
        else:
            raw_input = kwargs["prompt"]

        if self.usage == "evaluate":
            # Tokenize the inputs for evaluation
            input_tokenized = self.tokenizer(raw_input, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
            input_ids = input_tokenized["input_ids"].to(self.device)
            attention_mask = input_tokenized["attention_mask"].to(self.device)
        else:
            input_tokenized = self.tokenizer(raw_input, add_special_tokens=True)
            input_ids = torch.tensor(input_tokenized["input_ids"]).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(input_tokenized["attention_mask"]).unsqueeze(0).to(self.device)

        # Generate outputs
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
        )

        if self.usage == "evaluate":
            # Decode the outputs for evaluation
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        else:
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if format_output:
            if self.usage == "evaluate":
                outputs = [self.data_formatter.format_output(o) for o in outputs]
            else:
                output = self.data_formatter.format_output(output)

        return outputs if self.usage == "evaluate" else output

  
    def evaluate(self, eval_dataset, batch_size=1, report=False):
        # Eval dataset must have 3 fields: "text", "entity_type", "label"
        print(f'Evaluating dataset: {eval_dataset} ...')

        # Make predictions
        preds = eval_dataset.map(
            lambda x: {'pred': self.predict(samples=x, format_prompt=True)},
            batched=True,
            # set number of samples per batch
            batch_size=batch_size,
        )['pred']
        # convert to list for all values in labels
        if eval_dataset.features['label'].dtype != 'list':
            labels = [eval(x['label']) for x in eval_dataset]
        else:
            labels = eval_dataset['label']

        # Calculate metrics
        f1_score, list_correct, list_preds = self.calculate_f1(preds, labels, report=report)
        if report:
            df_report = eval_dataset.to_pandas()
            df_report['pred'] = list_preds
            df_report['is_correct'] = list_correct
            # df_report.to_csv('eval/eval_report.csv', index=False)
            df_report.to_excel('eval/eval_report.xlsx')

        return {
            'f1_score': f1_score,
        }


    def calculate_f1(self, preds, labels, report=False):
        set_preds = [set(pred) for pred in preds]
        set_labels = [set(label) for label in labels]

        if report:
            list_correct = [1 if pred == label else 0 for pred, label in zip(set_preds, set_labels)]

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

        if report:
            return f1_score, list_correct, [list(pred) for pred in set_preds]
        return f1_score, None, None

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
