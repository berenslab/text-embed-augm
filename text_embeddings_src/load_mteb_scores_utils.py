import os
import json
from pathlib import Path

import numpy as np
import pandas as pd



def load_mteb_results(directory):
    results = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Create a key based on the relative path
                relative_path = os.path.relpath(file_path, directory)
                key = os.path.splitext(relative_path)[0]  # Remove the .json extension
                results[key] = data

    return results


# CREATE DIC FOR RESULTS
def create_results_dict(mteb_results):
    mteb_scores = dict()
    for key in np.sort(list(mteb_results.keys())):
        task = key.split("/")[-1]
        if "scores" in mteb_results[key].keys():
            if len(mteb_results[key]["scores"]["test"]) != 1:
                # more than 1 language (1 test set)
                for i in range(len(mteb_results[key]["scores"]["test"])):
                    if mteb_results[key]["scores"]["test"][i]["languages"] == [
                        "eng-Latn"
                    ]:
                        score = mteb_results[key]["scores"]["test"][i]["main_score"]
                        mteb_scores[task] = score

            else:
                score = mteb_results[key]["scores"]["test"][0]["main_score"]
                mteb_scores[task] = score

        else:
            continue

    return mteb_scores

def create_versions_dict(mteb_results):
    mteb_scores = dict()
    for key in np.sort(list(mteb_results.keys())):
        if "scores" in mteb_results[key].keys():
            dict_versions = {}
            dict_versions["dataset_revision"] = mteb_results[key][
                "dataset_revision"
            ]
            dict_versions["mteb_version"] = mteb_results[key][
                "mteb_version"
            ]
            task = key.split("/")[-1]
            mteb_scores[task] = dict_versions

        else:
            continue

    return mteb_scores


def load_mteb_layers_results(model_name, finetune_type, variables_path):
    saving_dir_str = (
        f"results_{model_name.lower()}"
        if finetune_type == ""
        else f"results_{model_name.lower()}_{finetune_type}_finetuning"
    )

    mteb_scores_layers_model = pd.DataFrame()

    for layer_number in np.arange(13):
        # layer_number = 12
        saving_path = (
            Path("embeddings_" + model_name.lower())
            / Path("updated_dataset")
            / Path("mteb_benchmark")
            / Path("eval_layers")
            / Path(saving_dir_str)
            / Path(f"layer_{layer_number}")
        )
        root_directory = variables_path / saving_path
        mteb_results_model = load_mteb_results(root_directory)

        # unpack
        mteb_scores_model = create_results_dict(mteb_results_model)

        mteb_scores_layers_model[layer_number] = mteb_scores_model

    return mteb_scores_layers_model.T
