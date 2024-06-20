from datasets import load_dataset, Dataset
from transformers import MT5TokenizerFast
import argparse
from data.data_utils.utils import *
from groq import RateLimitError
import os, pytz, datetime


# Initialization
# model
MAX_LENGTH = 1024
model_id = 'google/mt5-base'
# Corpus
duplicate_filter_threshold = 20     # Filter text that has duplicated string with length > threshold


def load_corpus(data_path: str, from_sample_no: int, num_samples: int): 
    corpus = load_dataset(data_path, split='train', streaming=True)
    corpus = corpus.skip(from_sample_no).take(num_samples)
    corpus = Dataset.from_generator(
        lambda: (yield from corpus),
        features=corpus.features,
    )
    return corpus

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/datasets/llama3_gen_data')
    parser.add_argument('--data_corpus_path', type=str, default='bkai-foundation-models/BKAINewsCorpus')
    parser.add_argument('--from_sample_no', type=int)
    parser.add_argument('--num_samples', type=int)

    args = parser.parse_args()

    # Load tokenizer
    print(f'Loading tokenizer from {model_id}...')
    tokenizer = MT5TokenizerFast.from_pretrained(model_id)

    # Load corpus
    print(f'Loading corpus from {args.data_corpus_path}...')
    corpus = load_corpus(args.data_corpus_path, args.from_sample_no, args.num_samples)
    print(f"Loaded corpus: {corpus}")
    print("Processing corpus...")
    # Filter corpus
    corpus = corpus.filter(
        lambda examples: [longest_duplicated_substring(text) > duplicate_filter_threshold for text in examples['text']],
        batched=True,
    )
    # Split corpus
    corpus = corpus.map(
        lambda samples: split_samples(tokenizer, samples),
        batched=True,
        remove_columns=corpus.column_names,
    )
    print(f"Processed corpus: {corpus}")

    # Load api client generator
    api_generator = GroqApiGenerator()
    data_generator = Llama3DataGenerator(api_generator, tokenizer, corpus.num_rows)
    print('Generating data with llama3....')
    corpus_generated = corpus.map(
        lambda samples: data_generator.gen_data(samples),
        batched=True,
        remove_columns=corpus.column_names,
    )

    # Save to disk
    output_path = args.output_dir + '/corpus_generated_' + str(args.from_sample_no) + '_to_' + str(args.from_sample_no + args.num_samples)
    print(f'Saving to disk file named {output_path}...')
    os.makedirs(args.output_dir, exist_ok=True)
    corpus_generated.save_to_disk(output_path)

    # Print end time
    print(f"End time: {datetime.datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))}")


if __name__ == '__main__':
    main()
