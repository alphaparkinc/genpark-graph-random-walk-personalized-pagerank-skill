"""
Example usage of Graph Random Walk Personalized PageRank Skill.
"""

from client import PersonalizedPageRank


def main():
    print("=== Graph Random Walk Personalized PageRank Demonstration ===")
    ppr = PersonalizedPageRank(damping=0.85)

    # Build agent memory graph
    # Agent knows about Database, Indexing, Vector Search, SQL, Cache, and User Auth
    ppr.add_edge("Database", "SQL", weight=2.0, bidirectional=True)
    ppr.add_edge("Database", "Indexing", weight=3.0, bidirectional=True)
    ppr.add_edge("Indexing", "B-Tree", weight=1.5, bidirectional=True)
    ppr.add_edge("Indexing", "Vector Search", weight=2.5, bidirectional=True)
    ppr.add_edge("Vector Search", "HNSW", weight=2.0, bidirectional=True)
    ppr.add_edge("Vector Search", "Embeddings", weight=2.0, bidirectional=True)
    ppr.add_edge("Database", "Cache", weight=1.0, bidirectional=True)
    ppr.add_edge("Cache", "Redis", weight=2.0, bidirectional=True)
    ppr.add_edge("User Auth", "JWT", weight=2.0, bidirectional=True)
    ppr.add_edge("User Auth", "Database", weight=0.5, bidirectional=True)

    # Context: User asks about "Vector Search" & "HNSW"
    # Personalized query focuses random walks around Vector Search
    seeds = {"Vector Search": 0.7, "HNSW": 0.3}
    print("\nSeed Context:", seeds)

    top_associates = ppr.top_k_associates(seeds, k=5)
    print("\nTop Associated Entities via Personalized PageRank:")
    for node, score in top_associates:
        print(f"  {node:<15}: PPR Score = {score:.4f}")


if __name__ == "__main__":
    main()
