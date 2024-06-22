from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    MT5TokenizerFast,
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)
import wandb
from huggingface_hub import login as hf_login


# Initialization
hf_login(token="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")
wandb.login(relogin=True, key="4b032b8bbff0e916bbc330ca43341be268d70498")
wandb.init(
    project='thesis',
    name='mt0_base-sft-open_ner_en-mix-QA_vi',
)

output_model_dir = 'mt0_base-sft-open_ner_en-mix-QA_vi'
model_checkpoint = "bigscience/mt0-base"


def main():
    # Load model and tokenizer
    tokenizer = MT5TokenizerFast.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Prepare dataset
    vi_qa_ds = load_dataset('nqv2291/vi-mT5-QA-ViQuAD_v1.1')
    en_open_ner_ds = load_dataset('nqv2291/en-mT5-ner-open_domain')

    en_open_ner_ds['train'] = en_open_ner_ds['train'].select(range(293_000))

    dataset = DatasetDict({
        'train': concatenate_datasets([vi_qa_ds['train'], en_open_ner_ds['train']]),
        'test':  en_open_ner_ds['test'],
    })
    dataset = dataset.shuffle(9)

    # Prepare training
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        padding='longest',
        return_tensors="pt",
        pad_to_multiple_of=8
    )

    args = Seq2SeqTrainingArguments(
        output_dir=output_model_dir,
        overwrite_output_dir=True,
        report_to='wandb',
        save_strategy = "steps",
        eval_strategy = "steps",
        num_train_epochs=1,
        save_steps=400,
        eval_steps=200,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=64,
        gradient_checkpointing=True,
        logging_steps=10,
        learning_rate=3e-5,
        lr_scheduler_type='constant',
        # predict_with_generate=True,
        bf16=True,
        tf32=False,
        optim='adafactor',
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=lambda x: compute_bleu(x, tokenizer),
    )

    model.train()
    trainer.train()
    print("Training finished!")

if __name__ == '__main__':
    main()
