from groq import Groq, RateLimitError
import yaml


class GroqApiGenerator:

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.cur_idx = -1

    def get_new_client(self):
        self.cur_idx = (self.cur_idx + 1) % len(self.api_keys)
        return Groq(api_key=self.api_keys[self.cur_idx])

    def _load_api_keys(self):
        with open('data/data_utils/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.api_keys = config['llama3_api_keys']
        return self.api_keys

class Llama3DataGenerator:

    def __init__(self, api_generator, tokenizer, num_total_samples):
        self.api_generator = api_generator
        self.client = api_generator.get_new_client()
        self.tokenizer = tokenizer
        self.num_total_samples = num_total_samples

        self.max_err_count = len(api_generator.api_keys)
        self.cur_err_count = 0
        self.cur_sample_no = 0


    def gen_data(self, samples):
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