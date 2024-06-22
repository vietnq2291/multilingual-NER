from ner.pipeline import PipelineNER
import argparse
from datasets import load_dataset
import yaml


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe_config_id", default=None)
    parser.add_argument("--data_path")

    args = parser.parse_args()
    pipe_config_id = args.pipe_config_id
    data_path = args.data_path

    # deine pipeline
    ner_pipe = PipelineNER(pipe_config_id=pipe_config_id, usage="evaluate")

    # load dataset
    if ".csv" in data_path:
        eval_dataset = load_dataset("csv", data_files=data_path, split="test")
    elif ".json" in data_path:
        eval_dataset = load_dataset("json", data_files=data_path, split="test")
    else:
        eval_dataset = load_dataset(data_path, split="test")
    
    # evaluate
    result = ner_pipe.evaluate(eval_dataset)
    print(result)


if __name__ == "__main__":
    main()
