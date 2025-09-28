import torch
import torch.nn.functional as F

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