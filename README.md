# Embedding-Drift Impact Reduction Through Continuous Monitoring & Automated Retraining

## Resume Bullet
> **"Embedding-drift impact reduction of 32% through continuous embedding-space monitoring and automated retraining triggers"**

A runnable, end-to-end project that teaches you exactly what every word means — with code you can step through, modify, and explain in an interview.

---

## What Does This Actually Mean? (Plain English)

Your ML system uses **embeddings** (dense vectors) to represent items, users, or documents. These embeddings are learned when you train the model.

**The problem:** The real world keeps changing, but your embeddings are frozen.
- New products appear that don't match old embeddings
- User behavior shifts (pandemic, trends, seasons)
- Language evolves ("AI agent" means something different in 2024 vs 2023)

**Result:** Your search/recommendation quality silently degrades. Users get worse results. Nobody notices until it's bad.

**The solution:** Continuously monitor the embedding space for drift. When drift exceeds thresholds, automatically trigger model retraining. This catches degradation early instead of letting it accumulate.

### The Three Key Concepts

| Concept | What It Means | Why It Matters |
|---|---|---|
| **Embedding Drift** | Gap between frozen model embeddings and current reality | Search/recs silently degrade over time |
| **Continuous Monitoring** | Scheduled computation of drift metrics (centroid shift, KL divergence, recall) | Catches drift early before users notice |
| **Automated Retraining Triggers** | Rules that fire model retraining when metrics cross thresholds | No manual intervention needed, minimal quality loss |

---

## How to Run

### VS Code (Interactive)
1. Install Jupyter extension
2. Open `embedding_drift_monitor.py` — each `# %%` is a runnable cell
3. `pip install numpy pandas scikit-learn matplotlib seaborn scipy`
4. Step through cell by cell

### Google Colab
1. Upload `embedding_drift_monitor.py` to Colab
2. Run All (Ctrl+F9) — ~2 minutes, no GPU needed

### Command Line
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
python embedding_drift_monitor.py
```

---

## What the Code Does (Section by Section)

| Section | What Happens | Key Concept |
|---|---|---|
| **2** | Create initial embedding space (items + queries) | Embeddings start calibrated at t=0 |
| **3** | Simulate 3 drift types over 20 time steps | Gradual + sudden + seasonal drift |
| **4** | Compute 6 drift metrics at each step | Centroid shift, KL div, neighborhood stability, recall |
| **5** | Define automated retraining trigger rules | Threshold, rate-of-change, and composite triggers |
| **6** | Run A/B: no-monitoring vs monitored | Fixed-schedule vs trigger-based retraining |
| **7** | Compare results → ~32% impact reduction | The headline number |
| **8** | Visualizations: recall curves, drift metrics, bar charts | Evidence you can show |
| **9-10** | Production architecture + interview Q&A | What to SAY about this |

---

## Drift Detection Metrics

| Metric | What It Detects | How It Works |
|---|---|---|
| **Centroid Shift** | Global distribution movement | Cosine distance between mean embedding vectors |
| **KL Divergence** | Statistical distribution change | Compares similarity score histograms over time |
| **Neighborhood Stability** | Retrieval result changes | Jaccard overlap of top-K neighbors now vs. reference |
| **Per-Item Cosine Drift** | Individual item degradation | Cosine similarity of each item's embedding vs. original |
| **Recall@K** | Business impact | Are we still finding the right items? |
| **Wasserstein Distance** | Distribution shape change | Earth mover's distance between embedding norm distributions |

---

## Retraining Trigger Types

| Trigger Type | Rule | Use Case |
|---|---|---|
| **Threshold** | `centroid_shift > 0.04` | Catches any significant drift |
| **Rate-of-Change** | `recall_drop > 5% per period` | Catches sudden drift events |
| **Composite** | `centroid > 0.03 AND recall < 0.85` | Reduces false alarms |
| **Cooldown** | Min 4 steps between retrains | Prevents trigger storms |

---

## Production Architecture

```
Live Traffic → Embedding Service → Metrics Store (Prometheus)
                                         │
                                   Drift Monitor (Airflow DAG)
                                   • Centroid shift
                                   • KL divergence
                                   • Recall@K
                                   • Neighborhood stability
                                         │
                                   Trigger Engine
                                   IF centroid > 0.04 OR recall < 0.80
                                         │
                                   Retraining Pipeline (SageMaker)
                                   1. Pull fresh data
                                   2. Fine-tune model
                                   3. Generate new embeddings
                                   4. A/B test → promote if better
```

---

## Dependencies

- Python 3.8+
- NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn
- **No GPU, no PyTorch required**

---

## From This Demo → Production

| This Demo | Production Version |
|---|---|
| Simulated drift | Real traffic data capture |
| In-memory metrics | Prometheus / CloudWatch time-series DB |
| Print-based alerts | PagerDuty / Slack / OpsGenie |
| simulate_retrain() | SageMaker / Vertex AI training job |
| Same concepts | Same interview answers |
