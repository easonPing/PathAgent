"""Compatibility filename. The historical fuzzy evaluator is intentionally removed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.datasets import answer_label, normalize_choices
from scripts.evaluate import main


def acc_of_seq(choices, gt, res):
    choices = normalize_choices(choices)
    predicted, gold = answer_label(res, choices), answer_label(gt, choices)
    return predicted is not None and gold is not None and predicted == gold


if __name__ == '__main__':
    main()
