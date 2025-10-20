from pathlib import Path
import json
import csv
import os
import numpy as np
import random
from typing import Optional, Tuple
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .compgcn_conv import CompGCNConv
from collections import defaultdict
import torch.nn.functional as F
import wandb

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
KG_DIR = BASE_DIR / "knowledge_graph"
KG_DIR.mkdir(parents=True, exist_ok=True)

class BaseKGEModel(nn.Module):
    def __init__(self, n_nodes: int, n_rels: int, emb_dim: int, higher_better: bool = False):
        super().__init__()
        self.ent = nn.Embedding(n_nodes, emb_dim)
        self.rel = nn.Embedding(n_rels, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight.data)
        nn.init.xavier_uniform_(self.rel.weight.data)
        self.higher_better = higher_better

    def score(self, s_idx, r_idx, o_idx):
        raise NotImplementedError

class TransEModel(BaseKGEModel):
    def __init__(self, n_nodes, n_rels, emb_dim=200, p_norm=1):
        super().__init__(n_nodes, n_rels, emb_dim)
        self.p = p_norm

    def score(self, s_idx, r_idx, o_idx):
        e_s = self.ent(s_idx)
        e_o = self.ent(o_idx)
        r = self.rel(r_idx)
        diff = e_s + r - e_o
        return torch.norm(diff, p=self.p, dim=1)

class TransHModel(BaseKGEModel):
    def __init__(self, n_nodes, n_rels, emb_dim=200, p_norm=1):
        super().__init__(n_nodes, n_rels, emb_dim)
        self.norm = nn.Embedding(n_rels, emb_dim)  # relation-specific normal vector
        nn.init.xavier_uniform_(self.norm.weight.data)
        self.p = p_norm

    def project(self, e, r_idx):
        n = self.norm(r_idx)
        n = n / torch.norm(n, p=2, dim=1, keepdim=True)
        return e - torch.sum(e * n, dim=1, keepdim=True) * n

    def score(self, s_idx, r_idx, o_idx):
        e_s = self.project(self.ent(s_idx), r_idx)
        e_o = self.project(self.ent(o_idx), r_idx)
        r = self.rel(r_idx)
        diff = e_s + r - e_o
        return torch.norm(diff, p=self.p, dim=1)
    
class RotatEModel(BaseKGEModel):
    def __init__(self, n_nodes, n_rels, emb_dim=200):
        super().__init__(n_nodes, n_rels, emb_dim * 2)  # use 2x dim for complex space
        self.emb_dim = emb_dim
        self.higher_better = False  # RotatE uses distance (lower is better)

    def normalize_entities(self):
        with torch.no_grad():
            w = self.ent.weight.data  # shape [n, emb_dim*2]
            # use reshape (safe for non-contiguous tensors) to shape into complex pairs
            w_c = w.reshape(-1, self.emb_dim, 2)   # [n, emb_dim, 2]
            mag = torch.sqrt((w_c**2).sum(dim=-1, keepdim=True)).clamp(min=1e-9)  # [n,emb_dim,1]
            w_c = w_c / mag
            self.ent.weight.data = w_c.reshape_as(w)

    def forward(self, s_idx, r_idx, o_idx):
        # ensure normalization every time we fetch embeddings
        self.normalize_entities()

        # reshape to complex numbers
        e_s = torch.view_as_complex(self.ent(s_idx).view(-1, self.emb_dim, 2))
        e_o = torch.view_as_complex(self.ent(o_idx).view(-1, self.emb_dim, 2))
        r = torch.view_as_complex(self.rel(r_idx).view(-1, self.emb_dim, 2))

        # enforce unit modulus on relations
        r = r / torch.abs(r).clamp(min=1e-9)

        rotated = e_s * r
        diff = rotated - e_o
        return torch.norm(torch.view_as_real(diff), dim=(1, 2))

    def score(self, s_idx, r_idx, o_idx):
        return self.forward(s_idx, r_idx, o_idx)

class CompGCNModel(nn.Module):
    def __init__(self, n_nodes, n_rels, emb_dim=200, num_layers=2,
                 dropout=0.3, bias=True, opn="corr"):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_rels = n_rels
        self.emb_dim = emb_dim

        # entity & relation embeddings
        self.ent = nn.Embedding(n_nodes, emb_dim)
        self.rel = nn.Embedding(n_rels, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight.data)
        nn.init.xavier_uniform_(self.rel.weight.data)

        # lightweight params object (no recursion)
        params = type("Params", (), {"dropout": dropout, "bias": bias, "opn": opn})()

        # CompGCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                CompGCNConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim,
                    num_rels=n_rels,
                    act=torch.relu,
                    params=params
                )
            )

        # scoring function (TransE-style for simplicity)
        self.higher_better = False

    def forward(self, edge_index, edge_type):
        x = self.ent.weight
        r = self.rel.weight
        for conv in self.layers:
            x, r = conv(x, edge_index, edge_type, r)  # CompGCNConv returns (x, r)
        return x, r

    def score(self, s_idx, r_idx, o_idx, edge_index=None, edge_type=None):
        # forward pass to get updated embeddings
        x, r = self.forward(edge_index, edge_type)
        e_s = x[s_idx]
        e_o = x[o_idx]
        r_vec = r[r_idx]
        diff = e_s + r_vec - e_o
        return torch.norm(diff, p=1, dim=1)

class KGTrainer:
    """
    Class wrapper for training TransE on the triples.csv created by KGBuilder.

    Example:
        tr = KGTrainer(kg_dir="kg", emb_dim=128)
        tr.load_triples()
        tr.train(epochs=10)
        tr.save_embeddings("kg/node_emb.npy", "kg/rel_emb.npy")
    """

    def __init__(self, kg_dir: str = "kg", emb_dim: int = 200, joint_dim: Optional[int] = None, 
                    margin: float = 1.0, lr: float = 1e-3, curated_factor: float = 3.0, 
                    device: Optional[str] = None, model_name: str = "TransE",
                    adv_temp: float = 1.0, clip_grad_norm: float = 5.0, weight_decay: float = 1e-5, model_kwargs=None):
        if kg_dir is None:
            self.kg_dir = KG_DIR
        else:
            self.kg_dir = Path(kg_dir) if Path(kg_dir).is_absolute() else (BASE_DIR / kg_dir)
        self.emb_dim = emb_dim
        self.joint_dim = joint_dim or emb_dim  # default to emb_dim
        self.margin = margin
        self.lr = lr
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.curated_factor = curated_factor
        self.model_name = model_name

        # these will be filled by load_triples()
        self.node2id = {}
        self.rel2id = {}
        self.triples = []  # tuples (s,r,o,conf,src)
        self.train_triples = []
        self.val_triples = []
        self.model = None
        self.optimizer = None
        self.model_kwargs = model_kwargs or {}
        
        self.adv_temp = adv_temp
        self.clip_grad_norm = clip_grad_norm
        self.weight_decay = weight_decay

        if self.joint_dim != self.emb_dim:
            self.proj_to_kg = nn.Linear(self.joint_dim, self.emb_dim, bias=False).to(self.device)
        else:
            self.proj_to_kg = nn.Identity()

    def load_maps(self):
        with (self.kg_dir / "node2id.json").open(encoding='utf8') as f:
            raw = json.load(f)
            # ensure values are ints
            self.node2id = {k: int(v) for k, v in raw.items()}
        with (self.kg_dir / "relation2id.json").open(encoding='utf8') as f:
            raw = json.load(f)
            self.rel2id = {k: int(v) for k, v in raw.items()}

    def load_triples(self, triples_csv: str = None):
        # default to kg/triples.csv
        tpath = Path(triples_csv) if triples_csv else self.kg_dir / "triples.csv"
        if not tpath.exists():
            raise FileNotFoundError(tpath)
        # ensure maps loaded
        self.load_maps()
        self.triples = []
        with tpath.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = int(row['s_id']); r = int(row['r_id']); o = int(row['o_id'])
                conf = float(row.get('confidence', 1.0))
                src = row.get('source', 'extracted')

                # boost curated edges
                if src == "curated":
                    conf *= getattr(self, "curated_factor", 3.0)
                
                self.triples.append((s, r, o, conf, src))
        print(f"[KGTrainer] loaded {len(self.triples)} triples")

        # train/val split
        random.shuffle(self.triples)
        split = int(0.9 * len(self.triples))
        self.train_triples = self.triples[:split]
        self.val_triples = self.triples[split:]

        # build train-only arrays
        if self.model_name == "CompGCN":
            # build directed lists
            srcs = [t[0] for t in self.train_triples]
            dsts = [t[2] for t in self.train_triples]

            # create bidirectional edges (s->o and o->s)
            edge_index = torch.tensor(
                [srcs + dsts, dsts + srcs],
                dtype=torch.long,
                device=self.device
            )
            edge_type = torch.tensor(
                [t[1] for t in self.train_triples] + [t[1] for t in self.train_triples],
                dtype=torch.long,
                device=self.device
            )
            self.edge_index = edge_index
            self.edge_type = edge_type
                
        self.pos_s = np.array([t[0] for t in self.train_triples], dtype=np.int64)
        self.pos_r = np.array([t[1] for t in self.train_triples], dtype=np.int64)
        self.pos_o = np.array([t[2] for t in self.train_triples], dtype=np.int64)
        self.pos_conf = np.array([t[3] for t in self.train_triples], dtype=np.float32)

        heads_per_rel = defaultdict(set)   # relation -> set(heads)
        tails_per_rel = defaultdict(set)   # relation -> set(tails)
        pair_count = defaultdict(int)      # (r) -> number of pairs

        for s, r, o, *_ in self.train_triples:
            heads_per_rel[r].add(s)
            tails_per_rel[r].add(o)
            pair_count[r] += 1

        # compute bernoulli probabilities
        self._bern_prob = {}
        for r in range(len(self.rel2id)):
            heads = len(heads_per_rel.get(r, [])) or 1
            tails = len(tails_per_rel.get(r, [])) or 1
            self._bern_prob[r] = float(np.clip(tails / (heads + tails), 0.05, 0.95))

        print(f"[KGTrainer] computed bernoulli probs for {len(self._bern_prob)} relations")

        # init model
        n_nodes = len(self.node2id)
        n_rels = len(self.rel2id)

        if self.model_name == "TransE":
            self.model = TransEModel(n_nodes, n_rels, self.emb_dim, **self.model_kwargs).to(self.device)
        elif self.model_name == "TransH":
            self.model = TransHModel(n_nodes, n_rels, self.emb_dim, **self.model_kwargs).to(self.device)
        elif self.model_name == "RotatE":
            self.model = RotatEModel(n_nodes, n_rels, self.emb_dim).to(self.device)
        elif self.model_name == "CompGCN":
            self.model = CompGCNModel(n_nodes, n_rels, self.emb_dim, **self.model_kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown KG model: {self.model_name}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self,
            epochs: int = 5,
            batch_size: int = 1024, 
            normalize: bool = True,
            save_every: int = 1,
            wandb_config: Optional[dict] = None,
            log_to_wandb: bool = True,
            patience: Optional[int] = None,
            metric: str = "mrr",
            num_negatives: Optional[int] = 4
            ):
        """
        Train KG TransE embeddings.

        - Supports optional early stopping with patience.
        - Saves best checkpoint (by chosen metric).
        - Robust wandb handling.
        """
        if self.model is None:
            raise RuntimeError("Call load_triples() first.")

        n = len(self.pos_s)
        steps = max(1, (n + batch_size - 1) // batch_size)

        # wandb init bookkeeping
        started_wandb = False
        if log_to_wandb:
            try:
                if getattr(wandb, "run", None) is None:
                    run_name = wandb_config.get("name") if wandb_config and "name" in wandb_config \
                            else f"kg_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    init_kwargs = {
                        "project": wandb_config.get("project", "fl-toy-kg") if wandb_config else "fl-toy-kg",
                        "name": run_name,
                        "config": wandb_config if wandb_config else None,
                        "reinit": True,
                    }
                    if os.environ.get("WANDB_MODE") == "offline" or (wandb_config and wandb_config.get("mode") == "offline"):
                        init_kwargs["mode"] = "offline"
                    wandb.init(**init_kwargs)
                    started_wandb = True
            except Exception as e:
                print(f"[WARN] wandb.init() failed — disabling wandb logging: {e}")
                log_to_wandb = False

        # early stopping bookkeeping
        best_val = -float("inf")
        bad_epochs = 0

        # build (s,r) -> {o} map for negatives filtering
        all_triples = self.triples if getattr(self, "triples", None) else (self.train_triples + self.val_triples)
        true_map = {}
        for s, r, o, *_ in all_triples:
            true_map.setdefault((s, r), set()).add(o)

        for epoch in tqdm(range(1, epochs + 1), desc="KG Train", unit="epoch"):
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            epoch_loss = 0.0

            for step in tqdm(range(steps), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False):
                start = step * batch_size
                end = min((step + 1) * batch_size, n)
                if start >= end:
                    continue
                batch_idx = idxs[start:end]

                s_batch = torch.LongTensor(self.pos_s[batch_idx]).to(self.device)
                r_batch = torch.LongTensor(self.pos_r[batch_idx]).to(self.device)
                o_batch = torch.LongTensor(self.pos_o[batch_idx]).to(self.device)
                conf_batch = torch.FloatTensor(self.pos_conf[batch_idx]).to(self.device)

                # positive scores
                if self.model_name == "CompGCN":
                    pos_scores = self.model.score(s_batch, r_batch, o_batch, self.edge_index, self.edge_type)
                else:
                    pos_scores = self.model.score(s_batch, r_batch, o_batch)

                # filtered negative sampling
                neg_s_list, neg_r_list, neg_o_list = [], [], []
                # precompute replacement probs for this batch from per-relation bernoulli
                if getattr(self, "_bern_prob", None) is not None:
                    probs = np.array([self._bern_prob[int(self.pos_r[i])] for i in batch_idx], dtype=np.float32)
                else:
                    probs = np.full(len(batch_idx), 0.5, dtype=np.float32)

                for _ in range(num_negatives):
                    # relation-aware decide which side to corrupt
                    corrupt_head = np.random.rand(len(batch_idx)) < probs

                    neg_s = self.pos_s[batch_idx].copy()
                    neg_o = self.pos_o[batch_idx].copy()
                    rand_nodes = np.random.randint(0, self.model.ent.num_embeddings, size=len(batch_idx))

                    # reject if rand_nodes[j] is a known true object for (s,r)
                    for j in range(len(batch_idx)):
                        s_j = int(self.pos_s[batch_idx[j]])
                        r_j = int(self.pos_r[batch_idx[j]])
                        o_j = int(self.pos_o[batch_idx[j]])
                        cand = int(rand_nodes[j])

                        max_tries = 50
                        tries = 0

                        if corrupt_head[j]:
                            # we will replace head -> ensure (cand, r_j) does NOT produce o_j
                            forbidden = true_map.get((cand, r_j), set())
                            while (o_j in forbidden) and tries < max_tries:
                                cand = np.random.randint(0, self.model.ent.num_embeddings)
                                forbidden = true_map.get((cand, r_j), set())
                                tries += 1
                        else:
                            # we will replace tail/object -> ensure candidate is not in true objects for (s_j, r_j)
                            forbidden = true_map.get((s_j, r_j), set())
                            while (cand in forbidden) and tries < max_tries:
                                cand = np.random.randint(0, self.model.ent.num_embeddings)
                                tries += 1

                        # final fallback if still problematic (rare)
                        def _candidate_ok(candidate, replace_head, s_j_local, r_j_local, o_j_local):
                            # When replacing the head, ensure the candidate head does NOT map to the same o_j via relation r_j_local.
                            if replace_head:
                                return o_j_local not in true_map.get((int(candidate), r_j_local), set())
                            # When replacing the tail, make sure candidate is not in true objects for (s_j_local, r_j_local)
                            return int(candidate) not in true_map.get((s_j_local, r_j_local), set())

                        if not _candidate_ok(cand, corrupt_head[j], s_j, r_j, o_j):
                            tries2 = 0
                            # bounded search to avoid pathological infinite loops; 200 tries is large but safe
                            while tries2 < 200:
                                candidate = np.random.randint(0, self.model.ent.num_embeddings)
                                if _candidate_ok(candidate, corrupt_head[j], s_j, r_j, o_j):
                                    cand = int(candidate)
                                    break
                                tries2 += 1
                        rand_nodes[j] = int(cand)

                    # apply corruption according to bernoulli decision
                    neg_s[corrupt_head] = rand_nodes[corrupt_head]
                    neg_o[~corrupt_head] = rand_nodes[~corrupt_head]

                    neg_s_list.append(torch.LongTensor(neg_s).to(self.device))
                    neg_r_list.append(r_batch)
                    neg_o_list.append(torch.LongTensor(neg_o).to(self.device))

                # concatenate all negatives, shape: (num_negatives * B,)
                neg_s_t = torch.cat(neg_s_list)
                neg_r_t = torch.cat(neg_r_list)
                neg_o_t = torch.cat(neg_o_list)

                if self.model_name == "CompGCN":
                    neg_scores = self.model.score(neg_s_t, neg_r_t, neg_o_t, self.edge_index, self.edge_type)
                else:
                    neg_scores = self.model.score(neg_s_t, neg_r_t, neg_o_t)

                # reshape to (num_negatives, B)
                neg_all = neg_scores.view(num_negatives, -1)  # [neg, B]
                pos = pos_scores  # [B]

                # For distance-based scores (lower is better), invert scores so higher=harder
                if getattr(self.model, "higher_better", False):
                    neg_for_weights = neg_all  # higher => harder already
                else:
                    neg_for_weights = -neg_all

                # avoid numerical issues: subtract column-wise max (logsumexp trick)
                m = neg_for_weights.max(dim=0, keepdim=True)[0]             # [1, B]
                stable = neg_for_weights - m
                neg_weights = F.softmax(self.adv_temp * stable, dim=0)      # [neg, B]

                with torch.no_grad():
                    pos_mean = pos.mean().item()
                    neg_mean = neg_all.mean().item()
                    gap = (neg_mean - pos_mean) if not getattr(self.model, "higher_better", False) else (pos_mean - neg_mean)
                    # entropy of weights (avg over batch)
                    ent = -(neg_weights * (neg_weights + 1e-12).log()).sum(dim=0).mean().item()
                
                #print(f"[DBG] epoch {epoch} step {step}: pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}, gap={gap:.4f}, neg_entropy={ent:.4f}")

                # weighted softplus margin loss per negative (smoother than relu)
                # For distance (lower better): target pos + margin should be < neg -> softplus(pos + margin - neg)
                if getattr(self.model, "higher_better", False):
                    loss_terms = F.softplus(neg_all + self.margin - pos.unsqueeze(0))  # [neg, B]
                else:
                    loss_terms = F.softplus(pos.unsqueeze(0) + self.margin - neg_all)  # [neg, B]

                # apply weights across negatives, then average across batch, weighted by conf
                loss_sample = (neg_weights * loss_terms).sum(dim=0)  # [B]
                loss = (loss_sample * conf_batch).mean()

                # -----------------------------
                # Backprop + stability: grad clip, optimizer step, immediate normalization
                # -----------------------------
                self.optimizer.zero_grad()
                loss.backward()
                # clip grads
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=getattr(self, "clip_grad_norm", 5.0))
                self.optimizer.step()

                # immediate normalization for embeddings for stability
                if normalize:
                    with torch.no_grad():
                        if self.model_name == "RotatE":
                            try:
                                # call the model's normalization method (preferred)
                                self.model.normalize_entities()
                            except Exception:
                                # fallback to safe row-wise normalization if the method isn't available / fails
                                w = self.model.ent.weight.data
                                denom = w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
                                self.model.ent.weight.data = w / denom

                            # relations: ensure unit modulus per complex dim (safe attempt)
                            try:
                                rel_w = self.model.rel.weight.data
                                rel_real = rel_w.reshape(-1, self.model.emb_dim, 2)
                                mag = torch.sqrt((rel_real**2).sum(dim=-1, keepdim=True)).clamp(min=1e-9)
                                rel_real = rel_real / mag
                                self.model.rel.weight.data = rel_real.reshape_as(rel_w)
                            except Exception:
                                # can't view/reshape relations as expected; skip and rely on epoch normalization
                                pass
                        else:
                            # generic models: normalize entity rows to unit norm
                            w = self.model.ent.weight.data
                            denom = w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
                            self.model.ent.weight.data = w / denom

                epoch_loss += float(loss.item())

            # normalize embeddings
            if normalize:
                with torch.no_grad():
                    w = self.model.ent.weight.data
                    denom = w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
                    self.model.ent.weight.data = w / denom

            avg_loss = epoch_loss / max(1, steps)
            print(f"[KGTrainer] Epoch {epoch}/{epochs} avg_loss={avg_loss:.6f}")

            # prepare logging payload
            log_data = {"kg/epoch": epoch, "kg/loss": avg_loss}

            # evaluate
            if hasattr(self, "val_triples") and len(self.val_triples) > 0:
                try:
                    mrr, hits1, hits10 = self.evaluate(self.val_triples, k=10)
                    print(f"[Eval] MRR={mrr:.4f}, Hits@1={hits1:.4f}, Hits@10={hits10:.4f}")
                    log_data.update({
                        "kg/val_mrr": mrr,
                        "kg/val_hits1": hits1,
                        "kg/val_hits10": hits10,
                    })

                    # early stopping
                    val_score = {"mrr": mrr, "hits1": hits1, "hits10": hits10}[metric]
                    if val_score > best_val:
                        best_val = val_score
                        bad_epochs = 0
                        self.save_embeddings(suffix="best")
                    else:
                        bad_epochs += 1
                        if patience and bad_epochs >= patience:
                            print(f"[EarlyStop] No improvement in {patience} epochs. Stopping at epoch {epoch}.")
                            break

                except Exception as e:
                    print(f"[WARN] evaluation failed: {e}")

            if log_to_wandb:
                try:
                    wandb.log(log_data)
                except Exception as e:
                    print(f"[WARN] wandb.log failed: {e}")

            if epoch % save_every == 0:
                try:
                    self.save_embeddings(suffix=f"epoch{epoch}")
                except Exception as e:
                    print(f"[WARN] save_embeddings failed: {e}")

        if started_wandb and getattr(wandb, "run", None) is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARN] wandb.finish failed: {e}")

    def evaluate(self, triples: list, k: int = 10):
        """
        Filtered evaluation: masks other true objects for the same (s,r) pair.
        Returns (mrr, hits1, hits10).
        """
        all_triples = self.triples if getattr(self, "triples", None) else (self.train_triples + self.val_triples)

        # build (s,r) -> set(o) map
        true_map = {}
        for s, r, o, *_ in all_triples:
            true_map.setdefault((s, r), set()).add(o)

        self.model.eval()
        with torch.no_grad():
            ranks = []
            n_nodes = int(self.model.ent.num_embeddings)
            higher_better = bool(getattr(self.model, "higher_better", False))

            for s, r, o, *_ in triples:
                s_t = torch.tensor([s], device=self.device)
                r_t = torch.tensor([r], device=self.device)
                all_objs = torch.arange(n_nodes, device=self.device)

                # compute scores
                if self.model_name == "CompGCN":
                    scores = self.model.score(
                        s_t.repeat(n_nodes),
                        r_t.repeat(n_nodes),
                        all_objs,
                        self.edge_index,
                        self.edge_type
                    )
                else:
                    scores = self.model.score(
                        s_t.repeat(n_nodes),
                        r_t.repeat(n_nodes),
                        all_objs
                    )

                # mask other true objects (except the held-out o)
                scores_masked = scores.clone()
                for other in true_map.get((s, r), set()):
                    if other != o:
                        if higher_better:
                            scores_masked[other] = -1e9
                        else:
                            scores_masked[other] = 1e9

                # rank entities
                sorted_idx = torch.argsort(scores_masked, descending=higher_better)
                pos = (sorted_idx == o).nonzero(as_tuple=False)

                if pos.numel() == 0:
                    rank = n_nodes + 1
                else:
                    rank = int(pos[0, 0].item()) + 1

                ranks.append(rank)

            mrr = float(np.mean([1.0 / r for r in ranks]))
            hits1 = float(np.mean([1 if r <= 1 else 0 for r in ranks]))
            hits10 = float(np.mean([1 if r <= 10 else 0 for r in ranks]))

        self.model.train()
        return mrr, hits1, hits10

    def save_embeddings(self, node_out: str = None, rel_out: str = None, suffix: str = ""):
        node_out = Path(node_out) if node_out else (self.kg_dir / f"node_embeddings{('_'+suffix) if suffix else ''}.npy")
        rel_out = Path(rel_out) if rel_out else (self.kg_dir / f"rel_embeddings{('_'+suffix) if suffix else ''}.npy")
        meta_out = self.kg_dir / f"embeddings_meta{('_'+suffix) if suffix else ''}.json"

        ent_w = self.model.ent.weight.detach().cpu().numpy()
        rel_w = self.model.rel.weight.detach().cpu().numpy()

        if self.model_name == "CompGCN":
            # run graph forward to get propagated embeddings
            x, r = self.model(self.edge_index, self.edge_type)
            ent_w = x.detach().cpu().numpy()
            rel_w = r.detach().cpu().numpy()
            np.save(node_out, ent_w)
            np.save(rel_out, rel_w)
            print(f"[KGTrainer] saved CompGCN propagated embeddings -> {node_out}, {rel_out}")

            # save metadata as well (previously skipped)
            meta = {
                "model_name": self.model_name,
                "emb_dim": self.emb_dim,
                "n_nodes": ent_w.shape[0],
                "n_rels": rel_w.shape[0],
                "ent_shape": list(ent_w.shape),
                "rel_shape": list(rel_w.shape),
                "higher_better": getattr(self.model, "higher_better", False),
            }
            with meta_out.open("w") as f:
                json.dump(meta, f, indent=2)
            print(f"[KGTrainer] saved metadata -> {meta_out}")
            return  # no need to save again

        if self.model_name == "RotatE":
            # reshape to (n, emb_dim, 2) -> complex array
            n_ent = ent_w.reshape(ent_w.shape[0], self.model.emb_dim, 2)
            n_rel = rel_w.reshape(rel_w.shape[0], self.model.emb_dim, 2)

            n_ent = n_ent[..., 0] + 1j * n_ent[..., 1]
            n_rel = n_rel[..., 0] + 1j * n_rel[..., 1]

            np.save(node_out, n_ent)
            np.save(rel_out, n_rel)
            print(f"[KGTrainer] saved RotatE complex embeddings -> {node_out}, {rel_out}")
        else:
            np.save(node_out, ent_w)
            np.save(rel_out, rel_w)
            print(f"[KGTrainer] saved embeddings -> {node_out}, {rel_out}")

        # save metadata with full shapes
        meta = {
            "model_name": self.model_name,
            "emb_dim": self.emb_dim,
            "n_nodes": ent_w.shape[0],
            "n_rels": rel_w.shape[0],
            "ent_shape": list(ent_w.shape),
            "rel_shape": list(rel_w.shape),
            "higher_better": getattr(self.model, "higher_better", False),
        }
        with meta_out.open("w") as f:
            json.dump(meta, f, indent=2)
        print(f"[KGTrainer] saved metadata -> {meta_out}")

    def _resize_embeddings(self, arr: np.ndarray, target_shape: Tuple[int, int], name: str) -> np.ndarray:
        """Resize embeddings array (pad or truncate) to match target shape."""
        out = np.zeros(target_shape, dtype=arr.dtype)
        min_rows = min(arr.shape[0], target_shape[0])
        min_cols = min(arr.shape[1], target_shape[1])
        out[:min_rows, :min_cols] = arr[:min_rows, :min_cols]

        if arr.shape[0] < target_shape[0] or arr.shape[1] < target_shape[1]:
            print(f"[WARN] {name} embeddings padded from {arr.shape} -> {target_shape}")
            # Xavier init for padded region
            fan_in, fan_out = target_shape[1], target_shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            pad = np.random.uniform(-limit, limit, size=target_shape).astype(arr.dtype)
            out[min_rows:, :] = pad[min_rows:, :]
            out[:, min_cols:] = pad[:, min_cols:]
        elif arr.shape != target_shape:
            print(f"[WARN] {name} embeddings truncated from {arr.shape} -> {target_shape}")
        return out

    def load_embeddings(self, node_in: str = None, rel_in: str = None, suffix: str = "", strict_shapes: bool = False):
        node_in = Path(node_in) if node_in else (self.kg_dir / f"node_embeddings{('_'+suffix) if suffix else ''}.npy")
        rel_in = Path(rel_in) if rel_in else (self.kg_dir / f"rel_embeddings{('_'+suffix) if suffix else ''}.npy")
        meta_in = self.kg_dir / f"embeddings_meta{('_'+suffix) if suffix else ''}.json"

        # load metadata if available
        meta = {}
        if meta_in.exists():
            with meta_in.open() as f:
                meta = json.load(f)
            print(f"[KGTrainer] loaded metadata <- {meta_in}")
            self.model.higher_better = meta.get("higher_better", False)
        else:
            print(f"[KGTrainer] WARNING: no metadata found, using defaults")
            self.model.higher_better = False

        if self.model_name == "RotatE":
            ent_w = np.load(node_in)
            rel_w = np.load(rel_in)

            assert np.iscomplexobj(ent_w), "RotatE node embeddings must be complex"
            assert np.iscomplexobj(rel_w), "RotatE rel embeddings must be complex"

            # convert back to stacked real
            ent_real = np.stack([ent_w.real, ent_w.imag], axis=-1).reshape(ent_w.shape[0], -1)
            rel_real = np.stack([rel_w.real, rel_w.imag], axis=-1).reshape(rel_w.shape[0], -1)

            if strict_shapes:
                if list(ent_real.shape) != list(self.model.ent.weight.shape):
                    raise ValueError(f"Entity shape mismatch: {ent_real.shape} vs {tuple(self.model.ent.weight.shape)}")
                if list(rel_real.shape) != list(self.model.rel.weight.shape):
                    raise ValueError(f"Relation shape mismatch: {rel_real.shape} vs {tuple(self.model.rel.weight.shape)}")
            else:
                ent_real = self._resize_embeddings(ent_real, self.model.ent.weight.shape, "RotatE nodes")
                rel_real = self._resize_embeddings(rel_real, self.model.rel.weight.shape, "RotatE rels")

            self.model.ent.weight.data = torch.tensor(ent_real, dtype=torch.float32, device=self.device)
            self.model.rel.weight.data = torch.tensor(rel_real, dtype=torch.float32, device=self.device)
            print(f"[KGTrainer] loaded RotatE complex embeddings <- {node_in}, {rel_in}")

        else:
            ent_w = np.load(node_in)
            rel_w = np.load(rel_in)

            if strict_shapes:
                if list(ent_w.shape) != list(self.model.ent.weight.shape):
                    raise ValueError(f"Entity shape mismatch: {ent_w.shape} vs {tuple(self.model.ent.weight.shape)}")
                if list(rel_w.shape) != list(self.model.rel.weight.shape):
                    raise ValueError(f"Relation shape mismatch: {rel_w.shape} vs {tuple(self.model.rel.weight.shape)}")
            else:
                ent_w = self._resize_embeddings(ent_w, self.model.ent.weight.shape, "nodes")
                rel_w = self._resize_embeddings(rel_w, self.model.rel.weight.shape, "rels")

            self.model.ent.weight.data = torch.tensor(ent_w, dtype=torch.float32, device=self.device)
            self.model.rel.weight.data = torch.tensor(rel_w, dtype=torch.float32, device=self.device)
            print(f"[KGTrainer] loaded embeddings <- {node_in}, {rel_in}")
