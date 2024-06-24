from ner.pipeline import PipelineNER
import argparse
from datasets import load_dataset
import yaml


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe_config_id", default=None)
    parser.add_argument("--data_path")
    parser.add_argument("--data_split", default="test")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--report", action="store_true")

    args = parser.parse_args()
    pipe_config_id = args.pipe_config_id
    data_path = args.data_path
    data_split = args.data_split
    if args.batch_size:
        batch_size = args.batch_size
    do_report = True if args.report else False

    # deine pipeline
    ner_pipe = PipelineNER(pipe_config_id=pipe_config_id, usage="evaluate")

    # load dataset
    if ".csv" in data_path:
        eval_dataset = load_dataset("csv", data_files=data_path, split=data_split)
    elif ".json" in data_path:
        eval_dataset = load_dataset("json", data_files=data_path, split=data_split)
    else:
        eval_dataset = load_dataset(data_path, split=data_split)
    
    # evaluate
    result = ner_pipe.evaluate(eval_dataset, batch_size=batch_size, report=do_report)
    print(result)


if __name__ == "__main__":
    main()
