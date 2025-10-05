from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import torch
import numpy as np

from KnowledgeGraphModel.KG_Trainer import KGTrainer
from Helpers.configLoader import Config

# -------------------------
# Load config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
cfg_path = BASE_DIR / "config" / "config.yaml"
cfg = Config.load(cfg_path)
cfg.validate()
cfg.set_run_name("diagnostic")

print("Loaded config:", cfg.to_dict())

# -------------------------
# Init KGTrainer
# -------------------------
kg_dir = BASE_DIR / "knowledge_graph"
kg = KGTrainer(
    kg_dir=kg_dir,
    emb_dim=cfg.kg_emb_dim,
    joint_dim=cfg.kg_emb_dim,
    margin=cfg.kg_margin,
    lr=cfg.kg_lr,
    model_name=cfg.kg_model,
    model_kwargs=cfg.kg_model_kwargs,
    adv_temp=cfg.adv_temp
)

kg.load_triples()
kg.load_embeddings(suffix="best", strict_shapes=False)

device = kg.device
print("Loaded model:", kg.model_name, "higher_better:", getattr(kg.model, "higher_better", False))

# -------------------------
# Pick some validation triples
# -------------------------
sample_triples = kg.val_triples[:5]
print("Checking", len(sample_triples), "validation triples\n")

# build true_map once (same as evaluate)
all_triples = kg.triples if getattr(kg, "triples", None) else (kg.train_triples + kg.val_triples)
true_map = {}
for s, r, o, *_ in all_triples:
    true_map.setdefault((s, r), set()).add(o)

n_nodes = int(kg.model.ent.num_embeddings)
higher_better = bool(getattr(kg.model, "higher_better", False))

for (s, r, o, conf, src) in sample_triples:
    print(f"\n=== triple (s={s}, r={r}, o={o}, src={src}, conf={conf:.2f}) ===")
    s_t = torch.tensor([s], device=device)
    r_t = torch.tensor([r], device=device)
    all_objs = torch.arange(n_nodes, device=device)

    # score all entities
    scores = kg.model.score(s_t.repeat(n_nodes), r_t.repeat(n_nodes), all_objs)
    scores = scores.detach().cpu().numpy()

    true_score = scores[o]
    rand_idx = np.random.choice(n_nodes, size=10, replace=False)
    print("true score:", float(true_score))
    print("random tail scores:", ", ".join([f"{idx}:{scores[idx]:.4f}" for idx in rand_idx]))

    # mask other true tails
    scores_masked = scores.copy()
    for other in true_map.get((s, r), set()):
        if other != o:
            scores_masked[other] = -1e9 if higher_better else 1e9

    # rank
    sorted_idx = np.argsort(-scores_masked) if higher_better else np.argsort(scores_masked)
    rank = np.where(sorted_idx == o)[0][0] + 1
    print("rank of true tail:", rank)
    print("MRR contrib:", 1.0 / rank, "Hits@1:", int(rank == 1), "Hits@10:", int(rank <= 10))

    # show top-k predictions
    topk = 5
    print("Top-5 predictions:")
    for idx in sorted_idx[:topk]:
        tag = "<-- TRUE" if idx == o else ""
        print(f"  {idx}: {scores[idx]:.4f} {tag}")
