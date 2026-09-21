"""Manifest-based evaluation: missing/invalid predictions are incorrect."""
from collections import defaultdict
from data_processing.datasets import answer_label, normalize_text


def score(samples, results, open_metrics=False):
    expected = {r["sample_id"] for r in samples}
    if set(results) - expected:
        raise ValueError("Results contain samples outside the evaluation manifest")
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    open_rows, missing, invalid = [], 0, 0
    for row in samples:
        pred = results.get(row["sample_id"])
        missing += pred is None
        valid = pred is not None and pred.get("status") == "ok"
        invalid += pred is not None and not valid
        answer = pred.get("pred_answer", "") if valid else ""
        if row["choices"]:
            correct = bool(valid and answer_label(answer, row["choices"]) == row["ground_truth_label"])
            keys = ["closed_overall"]
            if row.get("source"):
                keys.append("source:" + row["source"])
            if row.get("task"):
                task = "Receptor" if row["task"] in {"ER", "PR", "HER2"} else row["task"]
                keys.extend(["task:" + row["task"], "paper_group:" + task])
            for key in keys:
                groups[key]["correct"] += int(correct)
                groups[key]["total"] += 1
        else:
            open_rows.append((row["ground_truth"], answer))
    for value in groups.values():
        value["accuracy_percent"] = 100.0 * value["correct"] / value["total"]
    output = {"expected": len(samples), "received": len(results), "missing": missing,
              "invalid": invalid, "complete": missing == 0, "closed": dict(groups)}
    if open_rows:
        output["open"] = {"total": len(open_rows), "exact_match_percent": 100 * sum(
            bool(p) and normalize_text(g) == normalize_text(p) for g, p in open_rows) / len(open_rows)}
        if open_metrics:
            from eval.tokenizer import PTBTokenizer
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
            tokenizer = PTBTokenizer()
            refs = tokenizer.tokenize({i: [{"caption": g}] for i, (g, p) in enumerate(open_rows)})
            preds = tokenizer.tokenize({i: [{"caption": p}] for i, (g, p) in enumerate(open_rows)})
            for metric, names in [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
                                  (Meteor(), ["METEOR"]), (Rouge(), ["ROUGE-L"]), (Cider(), ["CIDEr"])]:
                values, _ = metric.compute_score(refs, preds)
                if len(names) == 1:
                    values = [values]
                output["open"].update(dict(zip(names, map(float, values))))
    return output
