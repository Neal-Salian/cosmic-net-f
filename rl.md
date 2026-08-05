here is the new plan improvements to turn it into capstone project
Here is a comprehensive summary block that you can copy and paste directly into your other Claude chat to seamlessly continue the work:
### Handoff Context & Project State: Cosmic-Net to RL-Cosmic-Net
We are advancing **Cosmic-Net**—our graph neural network (GNN) framework for inferring dark matter halo masses from cosmological subhalo catalogs—into a top-tier machine learning research project targeting **NeurIPS AI4Science Workshop** and **ICLR Main Track**.
#### 1. Current Foundation & Baseline
 * **Dataset:** IllustrisTNG-100-1 (Snapshot 99, z=0) subhalo catalog subset comprising **541 graph-structured halo samples** (432 train / 109 test).
 * **Current Performance:** Achieves an RMSE of **0.137 dex** and successfully distills Faber-Jackson-like physical equations via Symbolic Regression (R^2 = 0.965).
 * **Frozen Backbone:** best_model_augmented.pt will serve as the fixed feature extractor during RL policy training to avoid gradient divergence.
#### 2. The Core Methodological Pivot: Physics-Informed RL Graph Sparsification
Instead of relying on standard dense k-NN graphs and slow, post-hoc explainers (like GNNExplainer/PGExplainer), we are introducing a novel **Reinforcement-Learned (RL) Edge-Selection Policy**:
 * **Mechanism:** A lightweight policy network (\pi_\theta) that takes edge features and frozen GNN node embeddings, outputting discrete keep/drop binary decisions for graph edges *before* final prediction.
 * **Intrinsic Interpretability:** The sparsified graph *is* the explanation. By stripping away redundant edges prior to inference, the model achieves efficiency and transparency by construction.
 * **Physics-Informed Constraints:** The RL reward function is jointly optimized for:
   1. Downstream predictive accuracy (\text{RMSE}).
   2. Extreme edge sparsity (\frac{\vert{}E'\vert{}}{\vert{}E\vert{}}).
   3. Physical consistency (rewarding/penalizing edge selections based on violations of gravitational dynamics like the Virial Theorem).
#### 3. Execution Roadmap (Phases 0–3)
 * **Phase 0 (Baselines):** Build non-RL sparsification baselines (Random drop, Attention-weighted top-k, Gumbel-softmax edge masking) to defend against reviewer pushback (*"Why use RL instead of differentiable masks?"*).
 * **Phase 1 (RL Policy Loop):** Implement the REINFORCE/PPO training loop with a moving baseline to learn the edge-pruning policy over the frozen GNN backbone.
 * **Phase 2 (Astrophysical & OOD Validation):** Validate that the RL-retained edges map to high gravitational binding energy (U_{ij}), and test cross-simulation generalization on the **CAMELS** dataset.
 * **Phase 3 (Standard ML Benchmarks):** Evaluate the sparsification framework on standard graph ML datasets (e.g., MUTAG/OGB) if needed for main-track ICLR review requirements.
#### 4. Target Venues & Deadlines
 * **NeurIPS 2026 AI4Science Workshop**
   * *Suggested Contribution Deadline:* August 29, 2026
   * *Focus:* Domain application, physics integration, early RL-sparsification results.
 * **ICLR 2027 Main Track**
   * *Submission Deadline:* September 24, 2026 (AoE)
   * *Focus:* Rigorous methodology, graph structure learning, intrinsic interpretability, and thorough baseline comparisons.


conference target is the ML4PS workshop or wtv so see past eligibilty of it and all too like paper length and all with other ieee or indigo conferences as backups and iclr main conference if it goes well


write the following for it

papers that should be included in lit review name top 10 priority then 10 more in lower priority

For literature review, 
Sr no, names of authors in short, name of journal or conference, year, dataset used if any, methodology used in short, quanative results, limitations.
 
A
 
