
"""
Entry point for the EEG seizure classification experimentation framework.

Usage:
    python main.py
    python main.py --config my_config.yaml
    python main.py --config config.yaml --only train
    python main.py --config config.yaml --only eval
    python main.py --config config.yaml --only plots
    python main.py --config config.yaml --only xai
"""

import argparse
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experimentation import Experiment
from experimentation.models import (
    get_enabled_ml_models, get_enabled_dl_models,
    load_model, is_raw_dl_model, PYTORCH_AVAILABLE,
)


def main():
    parser = argparse.ArgumentParser(
        description="EEG Seizure Classification Experimentation Framework",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["train", "eval", "plots", "xai"],
        help="Run only a specific pipeline phase",
    )
    args = parser.parse_args()

    exp = Experiment(config_path=args.config)

    if args.only is None:
        exp.run()
    elif args.only == "train":
        exp.load_or_extract_features()
        exp.select_features()
        if PYTORCH_AVAILABLE:
            raw_models = [m for m in get_enabled_dl_models(exp.config) if is_raw_dl_model(m)]
            if raw_models:
                exp.load_raw_data()
        exp.train()
    elif args.only == "eval":
        exp.load_or_extract_features()
        exp.select_features()
        for mk in get_enabled_ml_models(exp.config):
            load_model(exp.config, mk, exp.pipelines, exp.selectors, exp.output_suffix)
        if PYTORCH_AVAILABLE:
            for mk in get_enabled_dl_models(exp.config):
                load_model(exp.config, mk, exp.pipelines, exp.selectors, exp.output_suffix)
                if is_raw_dl_model(mk) and exp.X_test_raw is None:
                    exp.load_raw_data()
        exp.evaluate_validation()
        exp.evaluate_test()
        exp.save_predictions()
    elif args.only == "plots":
        exp.load_or_extract_features()
        exp.select_features()
        for mk in get_enabled_ml_models(exp.config):
            load_model(exp.config, mk, exp.pipelines, exp.selectors, exp.output_suffix)
        if PYTORCH_AVAILABLE:
            for mk in get_enabled_dl_models(exp.config):
                load_model(exp.config, mk, exp.pipelines, exp.selectors, exp.output_suffix)
                if is_raw_dl_model(mk) and exp.X_test_raw is None:
                    exp.load_raw_data()
        exp.evaluate_test()
        exp.generate_plots()
    elif args.only == "xai":
        exp.load_or_extract_features()
        exp.select_features()
        for mk in get_enabled_ml_models(exp.config):
            load_model(exp.config, mk, exp.pipelines, exp.selectors, exp.output_suffix)
        exp.generate_xai()


if __name__ == "__main__":
    main()
