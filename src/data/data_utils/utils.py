from groq import Groq, RateLimitError
import yaml

from omegaconf import open_dict
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
from transformers import BatchEncoding, AutoTokenizer

# Data generation functions ---------------------------------------------------
class GroqApiGenerator:

    def __init__(self, config_path='data/data_utils/config.yaml'):
        self.api_keys = self._load_api_keys()
        self.cur_idx = -1

    def get_new_client(self):
        self.cur_idx = (self.cur_idx + 1) % len(self.api_keys)
        return Groq(api_key=self.api_keys[self.cur_idx])

    def _load_api_keys(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.api_keys = config['llama3_api_keys']
        return self.api_keys

class Llama3DataGenerator:

    def __init__(self, api_generator, num_total_samples):
        self.api_generator = api_generator
        self.client = api_generator.get_new_client()
        self.num_total_samples = num_total_samples

        self.max_err_count = len(api_generator.api_keys)
        self.cur_err_count = 0
        self.cur_sample_no = 0


    def gen_ner_data(self, samples):
        ids, texts, entities = [], [], []

        for idx, text in zip(samples['id'], samples['text']):
            while True:
                if self.cur_err_count >= self.max_err_count:
                    print('Too many errors, stopping...')
                    break
                print(f'generate sample no {self.cur_sample_no}/{self.num_total_samples} index: {idx}...')
                try:
                    out = self.ner_api(text)
                    cur_entities = eval(out)
                    
                    ids.append(idx)
                    texts.append(text)
                    entities.append(cur_entities)
                    self.cur_sample_no += 1
                    self.cur_err_count = 0
                except RateLimitError as e:
                    print('Rate limit error, changing api...')
                    self.client = self.api_generator.get_new_client()
                    self.cur_err_count += 1
                    continue
                except Exception as e:
                    print(e)
                break

            if self.cur_err_count >= self.max_err_count:
                break
        return {
            'id': ids,
            'text': texts,
            'entities': entities,
        }

    def gen_prompt_data(self, message, verbose=False):
        while True:
            if self.cur_err_count >= self.max_err_count:
                print('Too many errors, stopping...')
                break
            if verbose:
                print(f'generate sample no {self.cur_sample_no}/{self.num_total_samples}...')
            try:
                out = self.prompt_api(message)
                self.cur_sample_no += 1
                self.cur_err_count = 0
                return out
            except RateLimitError as e:
                print('Rate limit error, changing api...')
                self.client = self.api_generator.get_new_client()
                self.cur_err_count += 1
                continue
            except Exception as e:
                print(e)
            break

    def prompt_api(self, message):
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=message,
            temperature=0,
            max_tokens=1024,
            stream=False,
            seed=42
        )

        return completion.choices[0].message.content

    def ner_api(self, passage):
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful information extraction system. You understand Vietnamese really well and you have a wide knowledge of different types of entities based on the context."
                },
                {
                    "role": "user",
                    "content": f"""Given a passage in Vietnamese, your task is to extract all entities and identify their entity types precisely. The output must be in a list of tuples of the following format: [("entity 1", "type of entity 1"), ... ]. Only outputs the list of entities without any other information. Here is the passage:
                    
                    {passage}""",
                }
            ],
            temperature=0,
            max_tokens=1024,
            stream=False,
            seed=42
        )

        return completion.choices[0].message.content

# Corpus filtering functions ---------------------------------------------------
# Filter text that has long duplicated substrings (for example: a paragraph is a news article that has title and content are duplicated)

def build_suffix_array(s):
    suffixes = [(s[i:], i) for i in range(len(s))]
    suffixes.sort()
    suffix_array = [suffix[1] for suffix in suffixes]
    return suffix_array

def build_lcp_array(s, suffix_array):
    n = len(s)
    rank = [0] * n
    lcp = [0] * n

    for i, suffix in enumerate(suffix_array):
        rank[suffix] = i

    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while (i + h < n) and (j + h < n) and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1

    return lcp

def longest_duplicated_substring(s, return_length=True):
    if not s:
        return ""

    suffix_array = build_suffix_array(s)
    lcp = build_lcp_array(s, suffix_array)

    max_len = 0
    start_index = 0
    for i in range(1, len(lcp)):
        if lcp[i] > max_len:
            # Check if substrings are non-overlapping
            suffix1 = suffix_array[i]
            suffix2 = suffix_array[i - 1]
            if abs(suffix1 - suffix2) >= lcp[i]:
                max_len = lcp[i]
                start_index = suffix1

    if return_length:
        return max_len
    return s[start_index:start_index + max_len]

# Split the corpus as long paragraphs into smaller paragraphs
def split_into_sentences(paragraph, min_word_count=10):
    # Split the paragraph into sentences that has at least min_word_count words
    sentences = []

    prev_sentence = ""
    for sentence in paragraph.split('.'):
        cur_sentence = (prev_sentence + ' ' + sentence).strip()
        if len(cur_sentence.split()) >= min_word_count:
            sentences.append(cur_sentence)
            prev_sentence = ""
        else:
            prev_sentence = cur_sentence.strip()

    return sentences

def split_paragraph(tokenizer, paragraph, min_tok_count=150, max_tok_count=256):
    sentences = [sent for para in paragraph.split('\n') if para.strip() for sent in split_into_sentences(para) ]
    paragraphs = []
    current_paragraph = []
    current_tok_count = 0

    for sentence in sentences:
        no_sent_tokens = len(tokenizer(sentence + '. ', padding=False, truncation=False, add_special_tokens=False)['input_ids'])
        
        if no_sent_tokens > max_tok_count:
            continue

        if (current_tok_count + no_sent_tokens) > max_tok_count:
            paragraphs.append(('. '.join(current_paragraph)).strip())
            current_paragraph = []
            current_tok_count = 0

        current_paragraph.append(sentence)
        current_tok_count += no_sent_tokens

    if current_paragraph and current_tok_count > min_tok_count:
        paragraphs.append((' '.join(current_paragraph)).strip())

    return paragraphs

def split_samples(tokenizer, samples):
    ids, texts = [], []
    for sample_text, sample_idx in zip(samples['text'], samples['id']):
        paragraphs = split_paragraph(tokenizer, sample_text)
        ids.extend([f"{sample_idx}_{idx}" for idx in range(len(paragraphs))])
        texts.extend(paragraphs)
    return {
        'id': ids,
        'text': texts,
    }

# Pretraining corpus processing functions --------------------------------------
def process_pretraining_dataset(dataset, args, tokenizer):
    # We increase the input_length, because instead of masking tokens T5 replaces
    # masked spans with a single token, therefore to avoid padding we need to have
    # longer sequences at the start, before masking
    before_mask_input_length, target_length = compute_input_and_target_lengths(
        inputs_length=args.data.input_length,
        noise_density=args.data.mlm_probability,
        mean_noise_span_length=args.data.mean_noise_span_length,
    )

    with open_dict(args):
        args.data.before_mask_input_length = before_mask_input_length
        args.data.target_length = target_length


    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={
            'tokenizer': tokenizer,
            'in_length': before_mask_input_length,
        },
        remove_columns=['text'],
    )

    processed_dataset = processed_dataset.shuffle(seed=args.seed)

    return processed_dataset

def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        # _output_length = num_noise_tokens + num_noise_spans + 1
        _output_length = tokens_length
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length

def tokenize_function(examples, tokenizer, in_length):
    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
    )

    input_ids = tokenizer_out["input_ids"]

    concatenated_ids = np.concatenate(input_ids)

    total_length = concatenated_ids.shape[0]
    total_length = (total_length // in_length) * in_length

    concatenated_ids = concatenated_ids[:total_length].reshape(-1, in_length)
    result = {"input_ids": concatenated_ids}

    return result

@dataclass
class DataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        # labels_mask = ~mask_indices
        labels_mask = np.full_like(input_ids, fill_value=False)

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        # if batch["labels"].shape[-1] != self.target_length:
            # raise ValueError(
            #     f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
            #     f" {self.target_length}."
            # )
        if batch["labels"].shape[-1] != self.target_length+1:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length+1}."
            )


        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]



# Format data for training -----------------------------------------------------
def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction: {instruction}\n\n### Response:\n").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction: {instruction}\n\n### Input: {input}\n\n### Response:\n").format_map(row)

def ner_instruction(entity_type):
    return f"What describes {entity_type.lower()} in the text?"

def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)