import torch
import torch.nn.functional as F
from pathlib import Path
import re
import json
import warnings

def kg_alignment_loss(joint_emb, batch_ids, kg_embs, node2id, trainer,
                      labels=None, label_cols=None, loss_type="cosine"):
    if joint_emb.device != kg_embs.device:
        kg_embs = kg_embs.to(joint_emb.device)

    kg_vecs = []
    for i, id_ in enumerate(batch_ids):
        node_key = f"report:{id_}"
        if node_key in node2id:
            kg_vecs.append(kg_embs[node2id[node_key]])
        else:
            # fallback: label-based
            if labels is not None and label_cols is not None and i < len(labels):
                label_vec = labels[i].cpu().numpy()
                pos_labels = [label_cols[j] for j, v in enumerate(label_vec) if v > 0.5]

                label_embs = [
                    kg_embs[node2id[f"label:{lab}"]]
                    for lab in pos_labels if f"label:{lab}" in node2id
                ]
                if len(label_embs) > 0:
                    kg_vecs.append(torch.stack(label_embs).mean(dim=0))
                    continue
            # otherwise fallback to zero
            kg_vecs.append(torch.zeros_like(kg_embs[0]))

    kg_vecs = torch.stack(kg_vecs).to(joint_emb.device)
    joint_proj = trainer.proj_to_kg(joint_emb)

    if loss_type == "mse":
        return F.mse_loss(joint_proj, kg_vecs)
    elif loss_type == "cosine":
        return 1 - F.cosine_similarity(joint_proj, kg_vecs).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
def clean_ttl_file(input_path, output_path):
    """
    Clean TTL file:
    - Keeps only valid TTL-like lines (prefix, URIs, CURIEs, comments).
    - Removes decorative separators / plain text.
    - Auto-fixes missing semicolons after literal values.
    """
    keep_pattern = re.compile(r'(@prefix|<.*>|:|#)')

    with open(input_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()

            # keep blank lines
            if not stripped:
                fout.write(line)
                continue

            # skip non-TTL junk
            if not keep_pattern.search(stripped):
                continue

            # fix: if a line looks like `:predicate "value"` but has no ending ; or .
            if re.match(r"^:\w+\s+\".*\"$", stripped):
                fout.write(line.rstrip() + " ;\n")
            else:
                fout.write(line)

    return output_path

def log_and_print(*msgs, log_file=None):
    """
    Prints and logs the given messages to the specified log file.

    Args:
        *msgs: The messages to print and log.
        log_file: The path to the log file. Skip log if log file path is none

    Raises:
        ValueError: If log_file is None.
    """
    text = " ".join(str(m) for m in msgs)
    print(text)
    
    if log_file is None:
        warnings.warn("Log file is None skip saving log to file")
        return
    
    log_file = Path(log_file)
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_round_summary(summary, log_dir="logs"):
    """
    Logs the given round summary as a JSON object to a file in the given log directory.

    Args:
        summary (dict): The round summary to log.
        log_dir (str, optional): The directory to log to. Defaults to "logs".
    """
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_path = Path(log_dir) / "round_summary.jsonl"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")