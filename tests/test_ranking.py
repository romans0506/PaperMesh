from papermesh.ranking import rank_papers


def test_rank_papers_orders_by_relevance() -> None:
    papers = [
        {
            "id": "low",
            "title": "Sourdough baking",
            "abstract": "A guide to baking sourdough bread with a wild yeast starter and long fermentation.",
        },
        {
            "id": "high",
            "title": "Graph neural networks for molecules",
            "abstract": "We propose a graph neural network that predicts molecular properties "
            "by passing messages between atoms in a molecular graph.",
        },
        {
            "id": "mid",
            "title": "Convolutional networks for images",
            "abstract": "We train a deep convolutional neural network for image classification.",
        },
    ]

    ranked = rank_papers("graph neural networks for molecular property prediction", papers)

    assert [p["id"] for p in ranked] == ["high", "mid", "low"]
    assert all(0.0 <= p["relevance_score"] <= 1.0 for p in ranked)
    assert ranked[0]["relevance_score"] > ranked[-1]["relevance_score"]
