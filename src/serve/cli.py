from ner.pipeline import PipelineNER
import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe_config_id", default=None)

    args = parser.parse_args()
    pipe_config_id = args.pipe_config_id

    # deine pipeline
    ner_pipe = PipelineNER(pipe_config_id=pipe_config_id, usage="inference")

    # Inference loop
    while True:
        text = input("Text: ")
        if text == "":
            break
        entity_type = input("Entity type: ")
        if entity_type == "":
            break

        pred = ner_pipe.predict(text=text, entity_type=entity_type, format_output=True, format_prompt=True)
        print("Output:", pred)
        print("----------------------------------")


if __name__ == "__main__":
    main()
