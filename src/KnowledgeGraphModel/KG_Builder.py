import rdflib

class KGBuilder:
    """
    Build a Knowledge Graph from a .ttl ontology file.
    Converts entities/relations into IDs and outputs triples.
    """

    def __init__(self, ttl_path: str, namespace_filter: str = None):
        """
        Args:
            ttl_path (str): path to the .ttl file
            namespace_filter (str, optional): only keep entities/relations
                                               within this namespace
                                               (e.g., "http://example.org/human_body#")
        """
        self.ttl_path = ttl_path
        self.graph = rdflib.Graph()
        self.namespace_filter = namespace_filter

        # storage
        self.entities = set()
        self.relations = set()
        self.triples = []

        self.entity2id = {}
        self.relation2id = {}

    def build(self):
        """Parse the .ttl file and build ID-based triples."""
        # Parse TTL
        self.graph.parse(self.ttl_path, format="turtle")

        for s, p, o in self.graph:
            s, p, o = str(s), str(p), str(o)

            # filter by namespace if provided
            if self.namespace_filter:
                if not (s.startswith(self.namespace_filter) and o.startswith(self.namespace_filter)):
                    continue

            # only keep triples where object is an entity (not literal)
            if o.startswith("http://") or o.startswith("https://"):
                self.entities.add(s)
                self.entities.add(o)
                self.relations.add(p)
                self.triples.append((s, p, o))

        # map entities and relations to IDs
        self.entity2id = {e: i for i, e in enumerate(sorted(self.entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(self.relations))}

        # convert triples to ID form
        triples_id = [
            (self.entity2id[h], self.relation2id[r], self.entity2id[t])
            for (h, r, t) in self.triples
        ]

        return triples_id, self.entity2id, self.relation2id

    def summary(self):
        """Print a summary of the KG."""
        print("KG Summary")
        print("==========")
        print(f"TTL file: {self.ttl_path}")
        print(f"Entities: {len(self.entity2id)}")
        print(f"Relations: {len(self.relation2id)}")
        print(f"Triples: {len(self.triples)}")
        if self.triples:
            print("Example triple:", self.triples[0])
