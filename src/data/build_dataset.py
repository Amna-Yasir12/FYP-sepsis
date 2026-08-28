from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def load_labels():
    meta = pd.read_csv(RAW / "Metadata.csv")
    meta = meta[["Isolate", "Year", "CIP"]].rename(columns={"Isolate": "isolate", "CIP": "label"})
    before = len(meta)
    meta = meta[meta["label"].isin(["S", "R"])].copy()
    dropped = before - len(meta)
    meta["label"] = (meta["label"] == "R").astype(int)
    print(f"Loaded {len(meta)} isolates with a valid CIP label ({dropped} dropped for missing label).")
    return meta

def load_gene_presence(filename, source_tag):
    df = pd.read_csv(RAW / filename, sep="\t")
    df = df[["Sample", "gene"]].dropna()
    df["gene"] = source_tag + "__" + df["gene"].astype(str)
    df["present"] = 1
    wide = df.pivot_table(index="Sample", columns="gene", values="present", fill_value=0, aggfunc="max")
    wide.index.name = "isolate"
    return wide

GRAPH_CUTOFF = "cutoff_210"

def build_similarity_edges():
    pop = pd.read_csv(RAW / "PopulationStructure.csv", index_col=0)
    pop.index.name = "isolate"
    clusters = pop[GRAPH_CUTOFF]

    edges = []
    for _, group in clusters.groupby(clusters):
        members = group.index.tolist()
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j]))
    return pd.DataFrame(edges, columns=["isolate_a", "isolate_b"])

def main():
    labels = load_labels()

    card = load_gene_presence("CARD_resistance.csv", "CARD")
    resfinder = load_gene_presence("ResFinder_resistance.csv", "ResFinder")

    features = card.join(resfinder, how="outer").fillna(0).astype(int)
    features = features.reset_index()

    dataset = labels.merge(features, on="isolate", how="left")
    gene_cols = [c for c in dataset.columns if c not in ("isolate", "Year", "label")]
    dataset[gene_cols] = dataset[gene_cols].fillna(0).astype(int)

    dataset.to_csv(PROCESSED / "features.csv", index=False)

    edges = build_similarity_edges()
    valid = set(dataset["isolate"])
    edges = edges[edges["isolate_a"].isin(valid) & edges["isolate_b"].isin(valid)]
    edges.to_csv(PROCESSED / "edges.csv", index=False)

    n_nodes = len(dataset)
    n_edges = len(edges)
    avg_degree = 2 * n_edges / n_nodes
    label_counts = dataset["label"].value_counts()

    summary = (
        f"Isolates: {n_nodes}\n"
        f"Feature columns (gene presence/absence): {len(gene_cols)}\n"
        f"Label distribution: resistant={label_counts.get(1, 0)}, susceptible={label_counts.get(0, 0)}\n"
        f"Graph: {n_edges} edges (cutoff={GRAPH_CUTOFF}), avg degree={avg_degree:.1f}\n"
    )
    print(summary)
    (PROCESSED / "dataset_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
