import json


def get_pipe_config(pipe_config_id, caller_module_name):
    with open("ner/pipe_config.json", "r") as f:
        configs = json.load(f)
        pipe_config = configs[pipe_config_id]

    pipe_config["model_class"] = str_to_class(
        pipe_config["model_class"], caller_module_name
    )
    pipe_config["tokenizer_class"] = str_to_class(
        pipe_config["tokenizer_class"], caller_module_name
    )

    return pipe_config


def str_to_class(classname, caller_module_name):
    return getattr(caller_module_name, classname)

