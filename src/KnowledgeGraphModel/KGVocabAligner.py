import rdflib
import torch

class KGVocabAligner:
    def __init__(self, kg_builder, vocab, device="cpu"):
        """
        Args:
            kg_builder: a KGBuilder instance (already .build() called)
            vocab (dict): {token: token_id}
            device: torch device
        """
        self.graph = kg_builder.graph
        self.entity2id = kg_builder.entity2id
        self.vocab = vocab
        self.device = device

        # Build mapping from KG labels → entity ids
        self.label2id = self._extract_entity_labels()

        # Final alignment {token_id: entity_id}
        self.mapping = self._map_vocab_to_kg()

    def _extract_entity_labels(self, prefix="http://example.org/human_body#"):
        """
        Collect canonical names + rdfs:labels for entities.
        """
        label2id = {}
        for ent, eid in self.entity2id.items():
            # URI local name
            if ent.startswith(prefix):
                local = ent.replace(prefix, "")
                label2id[local.lower()] = eid
            # rdfs:label (multiword like "Chest Pain")
            for _, _, lbl in self.graph.triples((rdflib.URIRef(ent), rdflib.RDFS.label, None)):
                label2id[str(lbl).lower()] = eid
        return label2id

    def _map_vocab_to_kg(self):
        """
        Align dataset vocab tokens → KG entity IDs.
        Note: only single-token matches here.
        """
        mapping = {}
        for tok, vid in self.vocab.items():
            if tok.lower() in self.label2id:
                mapping[vid] = self.label2id[tok.lower()]
        return mapping

    def inject_embeddings(self, model, node_embs):
        """
        Copy KG embeddings into model.token_embeddings using alignment.
        """
        with torch.no_grad():
            for tok_id, ent_id in self.mapping.items():
                if ent_id < node_embs.size(0) and tok_id < model.token_embeddings.weight.size(0):
                    model.token_embeddings.weight[tok_id].copy_(node_embs[ent_id])
        print(f"[KGVocabAligner] Injected {len(self.mapping)} KG embeddings into vocab.")
