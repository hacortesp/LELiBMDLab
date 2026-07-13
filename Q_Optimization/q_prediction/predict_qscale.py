#!/usr/bin/env python3
"""Predict q_scale for one new electrolyte system."""

import argparse
import os
import sys

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
sys.modules.setdefault("pyarrow", None)

import joblib

from qscale_common import DEFAULT_OUTPUT_DIR, build_input_dataframe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict recommended q_scale for a new electrolyte."
    )
    parser.add_argument(
        "-anion_name",
        type=str,
        default="PF6",
        help="Anion name (default: PF6).",
    )
    parser.add_argument(
        "-salt-conc",
        type=float,
        default=1.0,
        help="Salt concentration in mol/L (default: 1.0).",
    )
    parser.add_argument(
        "-solvents",
        nargs="+",
        type=str,
        default=["EC", "DMC"],
        help="List of solvents (default: EC DMC).",
    )
    parser.add_argument(
        "-solvent-fracs",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="Relative solvent volume fractions (default: 0.5 0.5).",
    )
    parser.add_argument(
        "-temperature",
        type=float,
        default=298.0,
        help="Temperature in Kelvin (default: 298.0).",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "qscale_extratrees_model.joblib"),
        help="Path to trained Extra Trees model.",
    )
    parser.add_argument(
        "--preprocessor",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "qscale_preprocessor.joblib"),
        help="Path to fitted preprocessing object.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)
    preprocessor = joblib.load(args.preprocessor)

    features = build_input_dataframe(
        anion_name=args.anion_name,
        salt_conc=args.salt_conc,
        solvents=args.solvents,
        solvent_fracs=args.solvent_fracs,
        temperature=args.temperature,
    )
    transformed = preprocessor.transform(features)
    prediction = model.predict(transformed)[0]

    print("Recommended q_scale: {0}".format(prediction))
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[0]
        print("")
        print("Class probabilities:")
        for q_scale, probability in zip(model.classes_, probabilities):
            print("q_scale {0}: {1:.4f}".format(q_scale, float(probability)))


if __name__ == "__main__":
    main()
