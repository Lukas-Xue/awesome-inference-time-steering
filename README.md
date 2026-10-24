# Awesome Inference-Time Steering

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![Preprint](https://img.shields.io/badge/Preprint-Research%20Square-blue)](https://www.researchsquare.com/article/rs-9849186/v1) [![DOI](https://img.shields.io/badge/DOI-10.21203%2Frs.3.rs--9849186%2Fv1-green)](https://doi.org/10.21203/rs.3.rs-9849186/v1)

A curated paper list for **inference-time model steering** — methods that control large language models by intervening on their internal computations during generation, with strength tunable and the intervention reversible per request.

Based on our survey: [Inference-Time Model Steering via Predictive-State Intervention: A Survey](paper.pdf)

Feel free to suggest relevant papers in the following format.

```markdown
**Paper Title**
Authors. Venue Year. [paper](url) [code](url)
Note: brief description of contribution.
```

### Table of Contents

- [Scope and Eligibility](#scope-and-eligibility)
- [Latent-Level Methods](#latent-level-methods)
  - [Training-Free Contrastive Extraction](#training-free-contrastive-extraction)
  - [Supervised Probe Directions](#supervised-probe-directions)
  - [Sparse Features and Concept Dictionaries](#sparse-features-and-concept-dictionaries)
  - [Transport-Based Methods](#transport-based-methods)
  - [Stability-Oriented Methods](#stability-oriented-methods)
  - [Multi-Property and Adaptive Steering](#multi-property-and-adaptive-steering)
  - [Human Preference Alignment](#human-preference-alignment)
- [Logit-Level Methods](#logit-level-methods)
- [Distribution-Level Methods](#distribution-level-methods)
- [Decoding-Level Methods (Boundary)](#decoding-level-methods-boundary)
- [Safety and Robustness](#safety-and-robustness)
- [Evaluation and Benchmarks](#evaluation-and-benchmarks)
- [Representation Engineering and Interpretability](#representation-engineering-and-interpretability)
- [Multimodal Steering](#multimodal-steering)
- [Related Surveys](#related-surveys)
- [Toolkits and Frameworks](#toolkits-and-frameworks)
- [Background and Foundations](#background-and-foundations)
- [Boundary Cases (Not Steering, but Related)](#boundary-cases-not-steering-but-related)

---

## Scope and Eligibility

Following Sections 1 and 3 and Appendix F of the [survey](paper.pdf), core steering methods must satisfy both defining criteria:

1. **Tunability at serving time:** an external per-request signal controls the intervention, such as a direction, feature coefficient, logit bias, or interpolation weight, without retraining the base model.
2. **Per-request reversibility:** removing the intervention restores the base computation without persistent changes to the base model's weights or state.

The intervention must act on the **predictive state**: hidden activations, output logits, or the next-token distribution. Classify methods by what they modify; a method named “decoding” can belong to the logit family if it directly changes token scores. Search and reranking that change the selection rule remain decoding boundary cases.

Offline learning of probes, SAEs, steering vectors, or auxiliary experts is compatible with these criteria when the deployed intervention remains tunable and removable and the base model stays fixed. Prompting, RAG that only changes input context, learned input tokens, adapters, and persistent weight editing are related approaches rather than core methods. Hybrid preparation stages are identified in their notes.

The paper covers **text generation**, with a **May 2026** literature cutoff. This living list also tracks later work; the [multimodal section](#multimodal-steering) is a related extension beyond the paper's scope. Evaluation, interpretability, survey, toolkit, and background entries provide context and need not themselves introduce an eligible steering method.

**Literature update: October 16, 2026.** Additions were checked against primary paper or proceedings pages, including earlier omissions and work published after the survey cutoff. Unconfirmed venues are left as arXiv preprints.

**Citation and venue audit: October 18, 2026.** Titles, author lists, contribution notes, and venue labels were checked against primary sources. Published versions link to proceedings or publisher records where available; accepted papers without published proceedings link to official acceptance records or author-declared acceptance on arXiv. Submission and review records are not treated as acceptance. Preprint years refer to first arXiv submission; workshop publications are labeled explicitly.

---

## Latent-Level Methods

Methods that modify hidden states $h_{\ell,t}$ inside the residual stream. Perturbations propagate through all subsequent layers, making these the most expressive but also carrying the largest side-effect surface.

### Training-Free Contrastive Extraction

1. **Steering Language Models With Activation Engineering (ActAdd)**
   Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid. arXiv 2023 (revised 2024). [paper](https://arxiv.org/abs/2308.10248)
   Note: introduces activation addition from single contrastive pair; foundational method.

2. **Steering Llama 2 via Contrastive Activation Addition (CAA)**
   Rimsky, Gabrieli, Schulz, Tong, Hubinger, Turner. ACL 2024. [paper](https://aclanthology.org/2024.acl-long.828/)
   Note: scales contrastive extraction to hundreds of prompt pairs; evaluates on behavioral dimensions (sycophancy, refusal).

3. **Representation Engineering: A Top-Down Approach to AI Transparency (RepE)**
   Zou, Phan, Chen, Campbell, Guo, Ren, Pan, Yin, Mazeika, Dombrowski, Goel, Li, Byun, Wang, Mallen, Basart, Koyejo, Song, Fredrikson, Kolter, Hendrycks. arXiv 2023. [paper](https://arxiv.org/abs/2310.01405)
   Note: PCA over activation differences; broad concept control framework.

4. **Style Vectors for Steering Generative Large Language Models**
   Konen, Jentzsch, Diallo, Schütt, Bensch, El Baff, Opitz, Hecking. EACL Findings 2024. [paper](https://aclanthology.org/2024.findings-eacl.52/)
   Note: extends contrastive extraction to stylistic attributes.

5. **Improving Instruction-Following in Language Models through Activation Steering**
   Stolfo, Balachandran, Yousefi, Horvitz, Nushi. ICLR 2025. [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c3262a4c965ba9888f120d4f9e13478-Abstract-Conference.html)
   Note: steering vectors for instruction-following behavior.

6. **Comparing Bottom-Up and Top-Down Steering Approaches on In-Context Learning Tasks**
   Brumley, Kwon, Krueger, Krasheninnikov, Anwar. NeurIPS MINT Workshop 2024. [paper](https://openreview.net/forum?id=VMiWNaWVJ5)
   Note: compares function vectors (bottom-up) with in-context vectors (top-down), finding different strengths on precise tasks versus behavioral shifts.

7. **Improving Activation Steering in Language Models with Mean-Centring**
   Jorgensen, Cope, Schoots, Shanahan. arXiv 2023. [paper](https://arxiv.org/abs/2312.03813)
   Note: subtracts background mean activations from target activations to improve behavioral steering and function-vector extraction.

8. **In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering (ICV)**
   Liu, Ye, Xing, Zou. ICML 2024. [paper](https://proceedings.mlr.press/v235/liu24bx.html)
   Note: extracts vectors from demonstrations and shifts hidden states with adjustable magnitude; supports vector composition without retaining demonstrations in the prompt.

9. **Genesis Steering: Better Activation Steering Through Training Data**
   Bodin, Aubinais, Torre, Cheng. arXiv 2026. [paper](https://arxiv.org/abs/2610.19114)
   Note: constructs contrastive vectors from the model's fine-tuning data, or membership-inference-selected proxy examples, without updating model weights during steering.

### Supervised Probe Directions

10. **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)**
    Li, Patel, Viégas, Pfister, Wattenberg. NeurIPS 2023. [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html)
    Note: uses truthfulness probes to select attention heads, then shifts their activations during inference; intervention strength controls the truthfulness-helpfulness trade-off.

11. **Spectral Editing of Activations for Large Language Model Alignment (SEA)**
    Qiu, Zhao, Ziser, Korhonen, Ponti, Cohen. NeurIPS 2024. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/684c59d614fe6ae74a3be8c3ef07e061-Abstract-Conference.html)
    Note: projects activations toward directions with high covariance with positive demonstrations and low covariance with negative demonstrations; includes nonlinear feature mappings.

### Sparse Features and Concept Dictionaries

Includes SAE feature interventions and explicit concept dictionaries; PaCE uses sparse coding over a concept dictionary rather than an SAE.

12. **Improving Steering Vectors by Targeting Sparse Autoencoder Features (SAE-TS)**
    Chalnev, Siu, Conmy. arXiv 2024. [paper](https://arxiv.org/abs/2411.02193)
    Note: uses SAE as a lens for measuring and reducing off-target side effects.

13. **Interpretable Steering of Large Language Models with Feature Guided Activation Additions (FGAA)**
    Samuel Soo, Chen Guang, Chandrasekaran Balaganesh, Wesley Teng, Tan Guoxian, Yan Ming. ICLR BuildingTrust Workshop 2025. [paper](https://arxiv.org/abs/2501.09929) [venue](https://openreview.net/forum?id=swRxS7s4rB)
    Note: optimizes steering coefficients in SAE feature space for steering/coherence trade-off.

14. **CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features**
    Cho, Wu, Koshiyama. ICML 2026. [paper](https://proceedings.mlr.press/v306/cho26h.html)
    Note: selects SAE features by correlating generation-time activations with sample correctness and derives steering coefficients from their average activations.

15. **LF-Steering: Latent Feature Activation Steering for Enhancing Semantic Consistency in Large Language Models**
    Yang, Li, Wang, Zhou, Feng, Peng. arXiv 2025. [paper](https://arxiv.org/abs/2501.11036)
    Note: targets inconsistency-related SAE features to reduce semantic inconsistency.

16. **PaCE: Parsimonious Concept Engineering for Large Language Models**
    Luo, Ding, Chan, Thaker, Chattopadhyay, Callison-Burch, Vidal. NeurIPS 2024. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b3cca813dcd78fe75e4d4df2e6a0b1a7-Abstract-Conference.html)
    Note: sparsely codes activations over an explicit semantic concept dictionary and removes components labeled undesirable; uses sparse coding rather than an SAE.

17. **Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering (SpARE)**
    Zhao, Devoto, Hong, Du, Gema, Wang, He, Wong, Minervini. NAACL 2025. [paper](https://aclanthology.org/2025.naacl-long.264/)
    Note: mutual-information-based feature selection for knowledge control.

18. **SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs (DSG)**
    Muhamed, Bonato, Diab, Smith. COLM 2025. [paper](https://openreview.net/forum?id=kaPAalWAp3)
    Note: dynamic classifier for selective suppression of unwanted knowledge.

19. **Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering (SDCV)**
    Zhao, Wu, Yang, Shen, Liu, Du. EACL Findings 2026. [paper](https://aclanthology.org/2026.findings-eacl.40/)
    Note: filters task-irrelevant features from concept vectors via SAE denoising.

20. **Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement (SAE-RSV)**
    Wang, Wu, Shu, Ma, Liu. arXiv 2025. [paper](https://arxiv.org/abs/2509.23799)
    Note: semantically denoises vectors extracted from small datasets and augments them with related SAE features missing from those datasets.

21. **SAIF: A Sparse Autoencoder Framework for Interpreting and Steering Instruction Following of Language Models**
    He, Zhao, Qiao, Yang, Payani, Ma, Du. arXiv 2025. [paper](https://arxiv.org/abs/2502.11356)
    Note: SAE-based interpretation and steering of instruction-following behavior.

22. **SCAR: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs**
    Härle, Friedrich, Brack, Deiseroth, Schramowski, Kersting. NeurIPS SoLaR Workshop 2024. [paper](https://openreview.net/forum?id=o6pgbB1ijo)
    Note: conditional SAE architecture for concept-specific steering.

23. **Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms (STA)**
    Wang, Xu, Mao, Deng, Tu, Chen, Zhang. ACL 2025. [paper](https://aclanthology.org/2025.acl-long.1139/)
    Note: decomposes steering targets into atomic SAE features for robust control.

24. **Exploring the Personality Traits of LLMs through Latent Features Steering**
    Yang, Zhu, Liu, Hu, Li, Wang. arXiv 2024. [paper](https://arxiv.org/abs/2410.10863)
    Note: steers personality dimensions via SAE feature manipulation.

25. **A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering**
    Xie. arXiv 2025. [paper](https://arxiv.org/abs/2510.01246)
    Note: shows single top-1 SAE latents with token-wise decay can outperform top-k and mean activation difference.

26. **Analyze Feature Flow to Enhance Interpretation and Steering in Language Models**
    Laptev, Balagansky, Aksenov, Gavrilov. ICML 2025. [paper](https://proceedings.mlr.press/v267/laptev25a.html)
    Note: traces how SAE features evolve and compose across layers.

27. **Disentangling Steering Vectors**
    Hiramatsu, Atarashi, Takeuchi, Kashima. arXiv 2026. [paper](https://arxiv.org/abs/2609.07037)
    Note: trains an SAE on instance-level contrastive activation differences to decompose composite steering directions into separately controllable features.

### Transport-Based Methods

Methods that transform activation distributions or integrate a steering field while keeping base-model weights fixed. Their intervention location is latent, even when the learning objective matches distributions.

28. **Controlling Language and Diffusion Models by Transporting Activations (AcT)**
    Rodriguez, Blaas, Klein, Zappella, Apostoloff, Cuturi, Suau. ICLR 2025. [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/df4f6e43446b1ee29c5a33d32c279f83-Abstract-Conference.html)
    Note: uses optimal-transport maps between source and target activations for adjustable toxicity, concept, and truthfulness control; also studies image generation.

29. **LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss**
    Rodriguez, Klein, Gualdoni, Maiorca, Blaas, Zappella, Cuturi, Suau. NeurIPS 2025. [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/767ef009ca91f00d1e256bc6f3fc7969-Abstract-Conference.html) [code](https://github.com/apple/ml-lineas)
    Note: jointly learns layer-wise affine activation maps from unpaired data, with sparse regularization and a continuous application-strength control.

30. **Beyond Steering Vector: Flow-based Activation Steering for Inference-Time Intervention (FLAS)**
    Jin, Deng, Wang, Shen, Zhang. NeurIPS 2026 (accepted). [paper](https://neurips.cc/virtual/2026/poster/155564)
    Note: learns a concept-conditioned velocity field on a frozen LM; integration time controls strength along curved, token-dependent activation trajectories.

31. **Conditional Optimal Bridge for Riemannian Activation Steering (COBRAS)**
    Dalili, Sridhar, Narayanan, Mahdavi. NeurIPS 2026 (accepted). [paper](https://neurips.cc/virtual/2026/poster/153803) [code](https://github.com/arshandalili/cobras)
    Note: derives query-adaptive steering from a Schrödinger bridge on the residual-stream hypersphere, with strength gating and abstention away from training support.

32. **Steering by Influence: Curvature Aware Data Weighting for Activation Steering**
    Dixon, Roberts, Quinzan. arXiv 2026. [paper](https://arxiv.org/abs/2610.06383)
    Note: weights reference examples using curvature-aware influence functions before activation transport to reduce unrelated concept and token effects.

### Stability-Oriented Methods

33. **Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection**
    Dang, Ngo. ACL Findings 2026. [paper](https://aclanthology.org/2026.findings-acl.529/)
    Note: combines a norm-preserving rotation with discriminative layer selection to control refusal while limiting capability degradation.

34. **ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment**
    Zhao, Sun, Kong, Li, Wang, Jiang, Zhu, Abdelzaher, Choi, Li, Shao. ICLR 2026. [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/72136a6372df75ddc3b1cd7f27c63712-Abstract-Conference.html)
    Note: models activation steering as an ODE guided by a log-density-ratio barrier function, enabling multiple adaptive intervention steps.

35. **Angular Steering: Behavior Control via Rotation in Activation Space**
    Vu, Nguyen. NeurIPS 2025 (Spotlight). [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b0223cad0e73b793f31eb6cc41cefceb-Abstract-Conference.html)
    Note: rotates activations toward or away from a behavior direction; its adaptive variant selectively rotates activations aligned with the target feature.

36. **Steering Without Side Effects: Improving Post-Deployment Control of Language Models (KL-then-Steer)**
    Stickland, Lyzhov, Pfau, Mahdi, Bowman. NeurIPS SafeGenAI Workshop 2024. [paper](https://openreview.net/forum?id=tfXIZ8P4ZU)
    Note: hybrid preparation: first fine-tunes the base LM against steering side effects, then applies runtime activation steering; the preparation stage changes base weights.

37. **Towards Understanding Steering Strength**
    Taimeskhanov, Vaiter, Garreau. ICML 2026. [paper](https://proceedings.mlr.press/v306/taimeskhanov26a.html)
    Note: mathematical laws governing the relationship between steering magnitude and behavioral effect.

38. **Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control**
    Skifstad, Yang, Chou. ICML 2026. [paper](https://proceedings.mlr.press/v306/skifstad26a.html) [code](https://github.com/trustworthyrobotics/lqr-activation-steering)
    Note: uses layer-wise Jacobians and linear-quadratic feedback control to track semantic activation setpoints without offline training.

39. **Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions (GCAD)**
    Kang, Liu, Ma, Huang, Tan, Jiang. NeurIPS 2026 (accepted). [paper](https://neurips.cc/virtual/2026/poster/154174)
    Note: extracts attention deltas from system-prompt contributions and gates their application per token to reduce KV-cache contamination in multi-turn steering.

40. **GEOSTEER: Geodesic Optimization for Activation Steering in Large Language Models**
    Ngo, Vo, Le. arXiv 2026. [paper](https://arxiv.org/abs/2609.10658)
    Note: follows multiple geodesic steps guided by a learned nonlinear probe while preserving activation norms; total path length controls steering strength.

41. **One Token Can Be Enough: Bridging Prompting and Activation Steering with Prefix Steering**
    Zhu, Zhu. arXiv 2026. [paper](https://arxiv.org/abs/2610.04967)
    Note: studies the control-capability trade-off when existing activation interventions act only over an initial token span; this is activation steering, distinct from learned prefix tuning.

### Multi-Property and Adaptive Steering

42. **Multi-property Steering of Large Language Models with Dynamic Activation Composition**
    Scalena, Sarti, Nissim. BlackboxNLP 2024. [paper](https://aclanthology.org/2024.blackboxnlp-1.34/)
    Note: information-theoretic weighting for composing multiple steering vectors.

43. **Steer2Adapt: Dynamically Composing Steering Vectors Elicits Efficient Adaptation of LLMs**
    Han, Xu, Xuan, Song, Ouyang, Tian, Jiang, Qian, Jiang, Sun, Cui, Zhong, Liu, Han, You. ICLR Re-Align Workshop 2026. [paper](https://openreview.net/forum?id=LkPXxVBwyZ)
    Note: adapts to new tasks from a small example set by composing steering vectors in a reusable semantic prior subspace.

44. **Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors (SADI)**
    Wang, Yang, Peng. ICLR 2025. [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c4d26a95fd83f8e590f81c54ae670b5d-Abstract-Conference.html)
    Note: selects steering direction based on input semantics.

45. **Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment**
    Gadgil, Lin, Lee. EMNLP 2026 (accepted). [paper](https://arxiv.org/abs/2604.03867)
    Note: selects intervention layer adaptively based on input features.

46. **HyperSteer: Activation Steering at Scale with Hypernetworks**
    Sun, Baskaran, Wu, Sklar, Potts, Geiger. arXiv 2025. [paper](https://arxiv.org/abs/2506.03292)
    Note: generates steering vectors directly via hypernetworks for scalable multi-task steering.

47. **Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training (RICE)**
    Wang, Chen, Wang, He, Xu, Liang, Liu, Yao, Wang, Ma, Mi, Zhang, Tu, Li, Yu. NeurIPS 2025. [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cdcfd67988fea858bb22317497105053-Abstract-Conference.html)
    Note: steers MoE expert selection to reinforce reasoning effort.

48. **Programming Refusal with Conditional Activation Steering (CAST)**
    Lee, Padhi, Natesan Ramamurthy, Miehling, Dognin, Nagireddy, Dhurandhar. ICLR 2025. [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html) [code](https://github.com/IBM/activation-steering)
    Note: gates behavior vectors using activation-based conditions, enabling composable refusal rules without weight optimization.

49. **Dynamically Scaled Activation Steering (DSAS)**
    Ferrando, Suau, Gonzàlez, Rodriguez. TMLR 2026. [paper](https://openreview.net/forum?id=EXKv1KMNtR)
    Note: adapts the strength of a steering transformation to each layer and input, reducing interventions when the target behavior is already satisfied.

50. **FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models**
    Weng, Zhang, Cai, Li, Wang, Tian. ACL 2026. [paper](https://aclanthology.org/2026.acl-long.852/) [code](https://github.com/YukinoAsuna/FineSteer)
    Note: combines subspace-guided conditional steering with a mixture of steering experts to synthesize query-specific vectors while preserving utility.

51. **GAPS: Dimension-Level Gates for Conditional Activation Steering**
    Fereidouni, Haider, Sajjad, Siddique. arXiv 2026. [paper](https://arxiv.org/abs/2609.01878)
    Note: combines static neuron-separability and dynamic posterior gates to select which hidden dimensions need steering; complements token-level conditional methods.

52. **Adaptive Multi-Value Control in LLMs via Causal Activation Steering (AIMES)**
    Bhattacharjee, Tandon. arXiv 2026. [paper](https://arxiv.org/abs/2609.30405)
    Note: uses intermediate vocabulary readouts as online observers to adjust multiple moral-value steering strengths at each generation step.

53. **Kernelized Activation Steering (KAS)**
    Abdullaev, Pham, Do, Than, Nguyen. NeurIPS 2026 (accepted). [paper](https://neurips.cc/virtual/2026/poster/154267)
    Note: generalizes difference-in-means steering through kernel-defined, activation-dependent nonlinear fields; a linear kernel recovers the standard direction.

### Human Preference Alignment

54. **Toward Preference-Aligned Large Language Models via Residual-Based Model Steering (PaLRS)**
    La Cava, Tagarelli. IJCAI 2026. [paper](https://www.ijcai.org/proceedings/2026/642)
    Note: residual-based steering for preference alignment without retraining.

55. **Self-Improving Model Steering (SIMS)**
    Zhu, Wang, Jiang, Liang, Wang. arXiv 2025. [paper](https://arxiv.org/abs/2507.08967)
    Note: iterative self-improvement loop for steering vector refinement.

56. **Improved Representation Steering for Language Models (RePS)**
    Wu, Yu, Arora, Manning, Potts. NeurIPS 2025 (Spotlight). [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb1ef82926376d252dde00d5dd909f4b-Abstract-Conference.html)
    Note: reference-free bi-directional preference steering.

57. **From Weights to Activations: Is Steering the Next Frontier of Adaptation?**
    Ostermann, Gurgurov, Baeumel, Hedderich, Lapuschkin, Samek, Schmitt. ACL 2026. [paper](https://aclanthology.org/2026.acl-long.1377/)
    Note: unified framework positioning steering within the broader adaptation landscape.

58. **Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization (BiPO)**
    Cao, Zhang, Cao, Yin, Lin, Ma, Chen. NeurIPS 2024. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/58cbe393b4254da8966780a40d023c0b-Abstract-Conference.html) [code](https://github.com/CaoYuanpu/BiPO)
    Note: optimizes external activation vectors on preference pairs while freezing the LM, then adjusts their sign and magnitude for personalized inference-time control.

59. **Persona Vectors: Monitoring and Controlling Character Traits in Language Models**
    Chen, Arditi, Sleight, Evans, Lindsey. NeurIPS 2026 (accepted). [paper](https://neurips.cc/virtual/2026/poster/152713)
    Note: automatically extracts trait directions for monitoring and inference-time suppression of behaviors such as sycophancy; its separate preventative-training intervention is outside core scope.

---

## Logit-Level Methods

Methods that modify output scores $\ell_t$ directly. Because there are no downstream layers to destabilize, these are more stable than latent interventions but less expressive.

60. **DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts**
    Liu, Sap, Lu, Swayamdipta, Bhagavatula, Smith, Choi. ACL 2021. [paper](https://aclanthology.org/2021.acl-long.522/)
    Note: contrastive expert/anti-expert logit bias; cancels shared linguistic structure to amplify attribute signals.

61. **GeDi: Generative Discriminator Guided Sequence Generation**
    Krause, Gotmare, McCann, Keskar, Joty, Socher, Rajani. EMNLP Findings 2021. [paper](https://aclanthology.org/2021.findings-emnlp.424/)
    Note: Bayes-factor ratio from class-conditional LMs for generative discrimination.

62. **FUDGE: Controlled Text Generation With Future Discriminators**
    Yang, Klein. NAACL 2021. [paper](https://aclanthology.org/2021.naacl-main.276/)
    Note: future-attribute discriminator predicts downstream satisfaction from partial generations.

63. **Steering Language Models Before They Speak: Logit-Level Interventions**
    An, Park, Jin, Han. arXiv 2026. [paper](https://arxiv.org/abs/2601.10960)
    Note: z-normalized log-odds biases from labeled corpora; training-free multi-task control.

64. **DoubtLess: Training-Free Contrastive Decoding for Efficient Reasoning**
    Luo, Chen. arXiv 2026. [paper](https://arxiv.org/abs/2610.18862) [code](https://github.com/leoluolol/DoubtLess)
    Note: per-step logit steering adds a tunable brief-minus-verbose logit difference from the same small model, conditioned on the current chain of thought, to a larger model's thinking logits; all model weights remain fixed.

65. **Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP**
    Schick, Udupa, Schütze. TACL 2021. [paper](https://aclanthology.org/2021.tacl-1.84/)
    Note: derives token penalties from auxiliary prompts describing undesired behavior and reweights next-token probabilities with adjustable decay, without training or weight changes.

66. **Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model (RAD)**
    Deng, Raffel. EMNLP 2023. [paper](https://aclanthology.org/2023.emnlp-main.721/)
    Note: adds token-candidate reward scores to base logits with a tunable coefficient; a cached unidirectional reward model reduces guidance overhead.

67. **Tuning Language Models by Proxy**
    Liu, Han, Wang, Tsvetkov, Choi, Smith. COLM 2024 (Spotlight). [paper](https://openreview.net/forum?id=dribhnhm1i)
    Note: shifts a fixed large LM's logits using tuned-minus-untuned small-model predictions; only the auxiliary proxy is fine-tuned.

68. **DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning**
    Li, Wang. ACL Findings 2026. [paper](https://aclanthology.org/2026.findings-acl.953/)
    Note: extracts a task vector from few-shot versus zero-shot output logits and injects it during generation, avoiding hidden-state hooks and additional query-time demonstrations.

---

## Distribution-Level Methods

Methods that interpolate the model's output distribution with an external distribution after the forward pass. The base model's computation is unmodified.

69. **Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM)**
    Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. ICLR 2020. [paper](https://openreview.net/forum?id=HklBjCEKvH)
    Note: retrieves nearest neighbors from a datastore to interpolate with model distribution; domain adaptation by swapping datastore.

70. **Improving Neural Language Models with a Continuous Cache**
    Grave, Joulin, Usunier. ICLR 2017. [paper](https://openreview.net/forum?id=B184E5qee)
    Note: stores recent hidden states and next tokens, then interpolates a cache-derived token distribution with the base LM using a tunable mixing weight.

71. **Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models**
    Cao, Wang, Wei, Guo, Chen, Zhou, Lin. NeurIPS 2025. [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a7bc7d298e4b509bfc3936995d22d828-Abstract-Conference.html)
    Note: parametric module replacing explicit retrieval; faster inference while maintaining distribution-interpolation interface.

72. **MLP Memory: A Retriever-Pretrained Memory for Large Language Models**
    Wei, Cao, Wang, Kai, Guo, Zhou, Lin. ICLR 2026. [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d72ae75abaa70a3b19c5d4f436c680d1-Abstract-Conference.html)
    Note: MLP-based parametric memory replacing kNN retrieval for faster inference.

73. **Efficient Nearest Neighbor Language Models**
    He, Neubig, Berg-Kirkpatrick. EMNLP 2021. [paper](https://aclanthology.org/2021.emnlp-main.461/)
    Note: improves kNN-LM inference through adaptive retrieval, datastore pruning, and dimensionality reduction while retaining next-token distribution interpolation.

74. **Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval (RetoMaton)**
    Alon, Xu, He, Sengupta, Roth, Neubig. ICML 2022. [paper](https://proceedings.mlr.press/v162/alon22a.html) [code](https://github.com/neulab/retomaton)
    Note: uses a retrieval automaton to reuse datastore transitions and reduce nearest-neighbor searches; mixes retrieved token probabilities with the fixed LM's distribution.

---

## Decoding-Level Methods (Boundary)

Methods that modify the token selection rule $\mathcal{D}$ rather than the predictive state. Included as boundary cases.

75. **Contrastive Decoding: Open-ended Text Generation as Optimization**
    Li, Holtzman, Fried, Liang, Eisner, Hashimoto, Zettlemoyer, Lewis. ACL 2023. [paper](https://aclanthology.org/2023.acl-long.687/)
    Note: favors tokens where expert disagrees with amateur model.

76. **NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics**
    Lu, Welleck, West, Jiang, Kasai, Khashabi, Le Bras, Qin, Yu, Zellers, Smith, Choi. NAACL 2022. [paper](https://aclanthology.org/2022.naacl-main.57/)
    Note: enforces hard lexical constraints via lookahead search.

---

## Safety and Robustness

77. **Refusal in Language Models Is Mediated by a Single Direction**
    Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. NeurIPS 2024. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html)
    Note: shows refusal behavior is mediated by a single removable direction; highlights safety fragility.

78. **Steering Externalities: Benign Activation Steering Unintentionally Increases Jailbreak Risk for Large Language Models**
    Xiong, He, Chen, Ko, Ho. NeurIPS 2026 (accepted). [paper](https://neurips.cc/virtual/2026/poster/156102)
    Note: demonstrates that even benign steering vectors can increase jailbreak vulnerability as a side effect.

79. **Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions**
    Goyal, Daumé III. EACL 2026. [paper](https://aclanthology.org/2026.eacl-long.268/)
    Note: evaluates robustness of steering under distribution shift; reveals brittleness of common approaches.

80. **Toward Universal Steering and Monitoring of AI Models**
    Beaglehole, Radhakrishnan, Boix-Adserà, Belkin. Science 2026. [paper](https://doi.org/10.1126/science.aea6792)
    Note: universal linear-representation-based steering and monitoring framework.

81. **Jailbreaking the Matrix: Nullspace Steering for Controlled Model Subversion (HMNS)**
    Pramanik, Maliha, Jha, Jha. ICLR 2026. [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3760d0ea4709a913a4804f4b4c073836-Abstract-Conference.html)
    Note: nullspace steering jailbreak attack exploiting activation geometry.

82. **The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness**
    Abdelnabi, Salem. NeurIPS 2025 (Spotlight). [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cf42f133f355e0e07a8957b508b26a1b-Abstract-Conference.html)
    Note: models alter behavior under evaluation; steering for test-awareness control.

83. **DSO: Direct Steering Optimization for Bias Mitigation**
    Monteiro Paes, Sivakumar, Wang, Fedzechkina, Theobald, Zappella, Apostoloff. CVPR 2026. [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Paes_DSO_Direct_Steering_Optimization_for_Bias_Mitigation_CVPR_2026_paper.html)
    Note: uses reinforcement learning to optimize linear activation transformations for an inference-time fairness-capability trade-off in LLMs and VLMs.

84. **Steered LLM Activations are Non-Surjective**
    Mishra, Khashabi, Liu. ICLR Workshops 2026 (Re-Align, Sci4DL). [paper](https://openreview.net/forum?id=NRrbJGp2Vx)
    Note: formalizes that steering can produce residual-stream states unreachable by any discrete prompt.

85. **Activation Steering Induces Emergent Misalignment: Steering Vector Construction and Comprehensive Evaluation**
    Cao, Lou, Liu, Feng, Li, Luu. arXiv 2026. [paper](https://arxiv.org/abs/2606.08682)
    Note: evaluates how vectors constructed from narrow unsafe tasks can induce broader misalignment, including sensitivity to strength, model family, and intervention layer.

86. **CounterSteer: Suppressing Indirect Prompt Injection with Activation Steering**
    Russinovich. arXiv 2026. [paper](https://arxiv.org/abs/2609.36570)
    Note: subtracts a causally screened residual-stream direction from tool-result tokens during prefill to suppress instruction takeover without fine-tuning the served model.

---

## Evaluation and Benchmarks

87. **AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders**
    Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts. ICML 2025. [paper](https://proceedings.mlr.press/v267/wu25a.html)
    Note: benchmarks concept detection and steering on Gemma-2-2B/9B; prompting outperforms the tested steering baselines, while ReFT-r1 improves representation-based control.

88. **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability**
    Karvonen, Rager, Lin, Tigges, Bloom, Chanin, Lau, Farrell, McDougall, Ayonrinde, Till, Wearden, Conmy, Marks, Nanda. ICML 2025. [paper](https://proceedings.mlr.press/v267/karvonen25a.html)
    Note: evaluates SAEs across eight metrics and finds that proxy-metric gains do not reliably imply better practical performance, including on unlearning tasks.

89. **Evaluating Steering Techniques Using Human Similarity Judgments**
    Studdiford, Rogers, Suresh, Mukherjee. COLM 2026. [paper](https://openreview.net/forum?id=HkbLTLiJDK)
    Note: evaluates steering with human triadic similarity judgments; prompting best matches the tested size/kind judgments among the compared methods.

90. **How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities**
    Xu, Xu, Xu, Hong, Huang, Xue, Zhang, Shen, Zheng, Chen, Deng. ACL 2026. [paper](https://aclanthology.org/2026.acl-long.1443/)
    Note: introduces a hierarchical evaluation of language features, sentiment, and personality, testing whether control survives increasingly specific behavioral requirements.

91. **CLaS-Bench: A Cross-Lingual Alignment and Steering Benchmark**
    Gurgurov, Al Ghussin, Baeumel, Chou, Schramowski, Mosbach, van Genabith, Ostermann. ACL Findings 2026. [paper](https://aclanthology.org/2026.findings-acl.1086/)
    Note: evaluates language steering across 32 languages using both language adherence and semantic relevance, with prompting and multiple activation-based baselines.

92. **What Models Express, Suppress, and Resist: Auditing Open-Weight LLMs with Persona Vectors**
    Zeng, Emami, Choi. arXiv 2026. [paper](https://arxiv.org/abs/2607.13162)
    Note: audits 53 traits and their composition to distinguish default, steerable, and resistant behaviors, exposing failures when combining latent traits.

---

## Representation Engineering and Interpretability

93. **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**
    Bricken, Templeton, Batson, Chen, Jermyn, et al. Transformer Circuits 2023. [paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
    Note: foundational work on SAE dictionary learning for interpretable feature decomposition.

94. **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet**
    Templeton, Conerly, Marcus, Lindsey, et al. Transformer Circuits 2024. [paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
    Note: scales SAE feature extraction to Claude 3 Sonnet and demonstrates behavioral control by clamping individual features.

95. **Scaling and Evaluating Sparse Autoencoders**
    Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike, Wu. ICLR 2025. [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/42ef3308c230942d223c411adf182c88-Abstract-Conference.html)
    Note: scaling laws and evaluation methodology for SAEs.

96. **Not All Language Model Features Are One-Dimensionally Linear**
    Engels, Michaud, Liao, Gurnee, Tegmark. ICLR 2025. [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d3221cdb27e49d9c1cd35ad254feccfe-Abstract-Conference.html)
    Note: some concepts are multi-dimensional manifolds (circles), limiting contrastive-vector methods.

97. **The Linear Representation Hypothesis and the Geometry of Large Language Models**
    Park, Choe, Veitch. ICML 2024. [paper](https://proceedings.mlr.press/v235/park24c.html)
    Note: theoretical grounding for why linear directions encode semantic concepts.

98. **The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets**
    Marks, Tegmark. COLM 2024 (Spotlight). [paper](https://openreview.net/forum?id=aajyHYjjsk)
    Note: empirical evidence for linear truth representations enabling steering.

99. **Discovering Latent Knowledge in Language Models Without Supervision**
    Burns, Ye, Klein, Steinhardt. ICLR 2023. [paper](https://openreview.net/forum?id=ETKGuby0hcs)
    Note: unsupervised discovery of truth-related directions in activation space.

100. **Brain-Grounded Axes for Reading and Steering LLM States**
     Andric. arXiv 2025. [paper](https://arxiv.org/abs/2512.19399)
     Note: uses brain-activity-derived axes for interpretable steering directions.

101. **Steering Large Language Models to Evaluate and Amplify Creativity**
     Olson, Ratzlaff, Hinck, Tseng, Lal. NeurIPS Creativity & Generative AI Workshop 2024 (Spotlight). [paper](https://arxiv.org/abs/2412.06060)
     Note: contrastive steering for creativity evaluation and amplification.

102. **The Logical Implication Steering Method for Conditional Interventions on Transformer Generation (LIMS)**
     Kalajdzievski. ICML 2025. [paper](https://proceedings.mlr.press/v267/kalajdzievski25a.html)
     Note: logic-rule-based conditional steering via logical implications.

103. **DEAL: Disentangling Transformer Head Activations for LLM Steering**
     Zhan, Liu, Lu, Feng, Xie, Cao, Wu. ACL KnowFM Workshop 2025 (non-archival). [paper](https://openreview.net/forum?id=HvXDK8Ryst)
     Note: uses disentangled VQ-AE representations to select and weight attention heads for steering; an earlier workshop version of REAL.

104. **REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering**
     Zhan, Liu, Feng, Xie, Cao, Wu. ICLR 2026. [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d2c50c5b2e3fcb33e8554d44b84eb52f-Abstract-Conference.html)
     Note: VQ-AE-based precise localization of steering targets.

105. **Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions (CDAS)**
     Bao, Zhang, Chen, Su, Cai, Peng, Sun, Weng, Yan, Yin. ICLR 2026. [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/704357127afabbd5a6eb979a57810767-Abstract-Conference.html)
     Note: distribution-matching approach for faithful bi-directional control.

106. **Function Vectors in Large Language Models**
     Todd, Li, Sen Sharma, Mueller, Wallace, Bau. ICLR 2024. [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html)
     Note: identifies attention heads carrying task representations and causally injects their function vectors into new contexts; studies vector composition.

107. **In-Context Learning Creates Task Vectors**
     Hendel, Geva, Globerson. EMNLP Findings 2023. [paper](https://aclanthology.org/2023.findings-emnlp.624/)
     Note: demonstrates that patching demonstration-derived task vectors into hidden states can reproduce ICL behavior; a mechanistic foundation for activation-based task transfer.

---

## Multimodal Steering

Related extension beyond the survey's text-generation scope; includes both latent interventions and output-score guidance in vision-language models.

108. **Automating Steering for Safe Multimodal Large Language Models**
     Wu, Wang, Xu, Cao, Oo, Hooi, Deng. EMNLP 2025. [paper](https://aclanthology.org/2025.emnlp-main.41/)
     Note: automated steering for safety in vision-language models.

109. **Learning to Steer: Input-Dependent Steering for Multimodal LLMs**
     Parekh, Khayatan, Shukor, Dapogny, Newson, Cord. NeurIPS 2025. [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ea491e2d1c46686b8db5cd11154f5d2c-Abstract-Conference.html)
     Note: input-dependent steering applied to multimodal architectures.

110. **Analyzing Finetuning Representation Shift for Multimodal LLMs Steering**
     Khayatan, Shukor, Parekh, Dapogny, Cord. ICCV 2025. [paper](https://openaccess.thecvf.com/content/ICCV2025/html/Khayatan_Analyzing_Finetuning_Representation_Shift_for_Multimodal_LLMs_Steering_ICCV_2025_paper.html)
     Note: analyzes how finetuning shifts representations and implications for steering in multimodal LLMs.

111. **GGSS: Geodesic-Gated Spherical Steering for Inference-Time Debiasing of Generative Vision-Language Models**
     Sun, Chen, Wei, Hsieh. EMNLP 2026 (accepted). [paper](https://arxiv.org/abs/2608.25375) [code](https://github.com/dukesun99/GGSS)
     Note: norm-preserving geodesic rotation of visual-token activations with an adaptive per-token gate for demographic debiasing of frozen VLMs.

112. **Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding (VCD)**
     Leng, Zhang, Chen, Li, Lu, Miao, Bing. CVPR 2024. [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html) [code](https://github.com/DAMO-NLP-SG/VCD)
     Note: contrasts logits from original and distorted images with tunable strength to reduce object hallucination without model training.

---

## Related Surveys

113. **A Survey of Controllable Text Generation Using Transformer-based Pre-trained Language Models**
     Zhang, Song, Li, Zhou, Song. ACM Computing Surveys 2024 (online 2023). [paper](https://doi.org/10.1145/3617680)
     Note: broad controllable generation covering training-time and inference-time approaches.

114. **A Comprehensive Study of Knowledge Editing for Large Language Models**
     Zhang, Yao, Tian, Wang, Deng, Wang, Xi, Mao, Zhang, Ni, Cheng, Xu, Xu, Gu, Jiang, Xie, Huang, Liang, Zhang, Zhu, Zhou, Chen. arXiv 2024. [paper](https://arxiv.org/abs/2401.01286)
     Note: surveys external-knowledge, parameter-merging, and weight-editing approaches; broader context beyond reversible predictive-state steering.

115. **Representation Engineering for Large-Language Models: Survey and Research Challenges**
     Bartoszcze, Munshi, Sukidi, Yen, Yang, Williams-King, Le, Asuzu, Maple. arXiv 2025. [paper](https://arxiv.org/abs/2502.17601)
     Note: surveys methods for detecting and editing high-level activation representations, their risks, alternatives, and open research challenges.

116. **Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models**
     Zhang, Zhang, Wang, et al. Computer Science Review 2026. [paper](https://doi.org/10.1016/j.cosrev.2026.101011)
     Note: organized around Interpretable Objects (what is targeted) rather than intervention location.

117. **A Survey on LLM Inference-Time Self-Improvement**
     Dong, Teleki, Caverlee. arXiv 2024. [paper](https://arxiv.org/abs/2412.14352)
     Note: broader inference-time improvement including search, verification, refinement.

118. **Exploring Controllable Text Generation Techniques**
     Prabhumoye, Black, Salakhutdinov. COLING 2020. [paper](https://aclanthology.org/2020.coling-main.1/)
     Note: earlier pipeline-based survey of controllable generation; cited in the survey's related work but previously absent from this list.

---

## Toolkits and Frameworks

119. **EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models**
     Xu, Wang, Xu, Xu, Wang, Deng, Yao, Zheng, Chen, Zhang. EMNLP Demos 2025. [paper](https://aclanthology.org/2025.emnlp-demos.38/)
     Note: unified toolkit supporting multiple steering methods with a common interface.

120. **pyvene: A Library for Understanding and Improving PyTorch Models via Interventions**
     Wu, Geiger, Arora, Huang, Wang, Goodman, Manning, Potts. NAACL Demos 2024. [paper](https://aclanthology.org/2024.naacl-demo.16/) [code](https://github.com/stanfordnlp/pyvene)
     Note: provides configurable static and trainable interventions on model internals for steering, causal analysis, and related model-editing research.

121. **AI Steerability 360: A Toolkit for Steering Large Language Models**
     Miehling, Natesan Ramamurthy, Venkateswaran, Ko, Dognin, Singh, Pedapati, Balakrishnan, Riemer, Wei, Vejsbjerg, Daly, Varshney. ACL Demos 2026. [paper](https://aclanthology.org/2026.acl-demo.43/) [code](https://github.com/IBM/AISteer360)
     Note: composes and benchmarks steering pipelines across model control surfaces; its input and structural controls extend beyond this list's core eligibility criteria.

---

## Background and Foundations

122. **Attention is All You Need**
     Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. NeurIPS 2017. [paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
     Note: introduces the Transformer architecture used by many of the language models in this list.

123. **A Mathematical Framework for Transformer Circuits**
     Elhage, Nanda, Olsson, et al. Transformer Circuits 2021. [paper](https://transformer-circuits.pub/2021/framework/index.html)
     Note: residual stream formalization enabling principled activation interventions.

124. **Locating and Editing Factual Associations in GPT (ROME)**
     Meng, Bau, Andonian, Belinkov. NeurIPS 2022. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)
     Note: localization of factual associations to MLP layers; foundational for steering targets.

125. **Transformer Feed-Forward Layers Are Key-Value Memories**
     Geva, Schuster, Berant, Levy. EMNLP 2021. [paper](https://aclanthology.org/2021.emnlp-main.446/)
     Note: MLP layers as key-value stores; motivates feed-forward interventions.

126. **Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small**
     Wang, Variengien, Conmy, Shlegeris, Steinhardt. ICLR 2023. [paper](https://openreview.net/forum?id=NpsVSN6o4ul)
     Note: mechanistic circuit discovery enabling targeted interventions.

127. **Towards Best Practices of Activation Patching in Language Models: Metrics and Methods**
     Zhang, Nanda. ICLR 2024. [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html)
     Note: best practices for activation patching; directly relevant to steering methodology.

128. **Knowledge Neurons in Pretrained Transformers**
     Dai, Dong, Hao, Sui, Chang, Wei. ACL 2022. [paper](https://aclanthology.org/2022.acl-long.581/)
     Note: identifies neurons associated with factual recall in BERT; contextual localization work already cited in the paper, including editing experiments outside core steering scope.

129. **Curvature Tuning: Provable Training-Free Model Steering from a Single Parameter**
     Hu, Gamba, Balestriero. NeurIPS 2025. [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/57cc9268e06500c7681e58829dca4a07-Abstract-Conference.html)
     Note: related non-text work on controlling activation-function curvature; evaluates vision classifiers and includes both fixed-hyperparameter steering and a trainable variant.

---

## Boundary Cases (Not Steering, but Related)

Included for context. These approaches use input-side control, learned parameter adaptation, search, or hybrid optimization rather than satisfying the full predictive-state steering definition. Decoding-only boundary methods are listed [separately](#decoding-level-methods-boundary).

130. **Compositional Steering of Large Language Models with Steering Tokens**
     Radevski, Gashteovski, Hong, Lawrence, Glavaš. ACL 2026. [paper](https://aclanthology.org/2026.acl-long.1435/)
     Note: learns compositional behavior tokens in input space; related multi-behavior control rather than direct predictive-state intervention.

131. **Neologism Learning as a Parameter-Efficient Alternative to Fine-Tuning for Model Steering**
     Park, Ramamurthi, Terry. arXiv 2025. [paper](https://arxiv.org/abs/2512.18551)
     Note: learns new token embeddings and controls behavior through prompts containing those tokens; an input-side alternative to activation steering.

132. **Steering Information Utility in Key-Value Memory for Language Model Post-Training (InfoSteer)**
     Deng, Chang, Chen. NeurIPS 2025. [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/93b11b5128ced940120f41ce9b216f39-Abstract-Conference.html)
     Note: applies FFN-memory interventions or entropy regularization during supervised fine-tuning; its reported post-training procedure updates base weights.

133. **CTRL: A Conditional Transformer Language Model for Controllable Generation**
     Keskar, McCann, Varshney, Xiong, Socher. arXiv 2019. [paper](https://arxiv.org/abs/1909.05858)
     Note: uses control codes learned during pretraining and supplied in the input; a conditional-generation baseline rather than a direct predictive-state intervention.

134. **Plug and Play Language Models: A Simple Approach to Controlled Text Generation (PPLM)**
     Dathathri, Madotto, Lan, Hung, Frank, Molino, Yosinski, Liu. ICLR 2020. [paper](https://openreview.net/forum?id=H1edEyBKDS)
     Note: performs iterative gradient updates to cached activations while freezing LM weights; retained as the survey's hybrid generation-time optimization boundary case.

135. **Prefix-Tuning: Optimizing Continuous Prompts for Generation**
     Li, Liang. ACL 2021. [paper](https://aclanthology.org/2021.acl-long.353/)
     Note: learns task-specific continuous prefixes while freezing LM weights; a learned-conditioning baseline in the survey's boundary category.

136. **The Power of Scale for Parameter-Efficient Prompt Tuning**
     Lester, Al-Rfou, Constant. EMNLP 2021. [paper](https://aclanthology.org/2021.emnlp-main.243/)
     Note: learns soft input prompts through backpropagation while freezing the model; a task-adaptation baseline rather than the survey's core steering mechanism.

137. **ReFT: Representation Finetuning for Language Models**
     Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts. NeurIPS 2024. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html)
     Note: learns task-specific interventions on hidden representations of a frozen LM; retained as a representation-finetuning boundary case under the survey's definition.

138. **SPARK: Adaptive Parametric Knowledge Participation for Faithful Retrieval-Augmented Generation**
     Li, Huang, Liu, Yan, Xie, Gu, Yu, Sun. arXiv 2026. [paper](https://arxiv.org/abs/2606.27786) [code](https://github.com/OpenBMB/SHIFT)
     Note: learns a lightweight policy that rescales each FFN sublayer's contribution per decoding step on a frozen LM, trained with a faithfulness reward to reduce reliance on conflicting parametric knowledge in RAG; a learned intervention module rather than a per-request tunable steering signal.

139. **LoRA: Low-Rank Adaptation of Large Language Models**
     Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. ICLR 2022. [paper](https://openreview.net/forum?id=nZeVKeeFYf9)
     Note: learns low-rank weight updates; a parameter-adaptation baseline, distinct from external interventions on predictive states.

140. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
     Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou. NeurIPS 2022. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
     Note: input-side reasoning-prompting baseline without an explicit predictive-state intervention.

141. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**
     Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan. NeurIPS 2023. [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html)
     Note: search-based prompting baseline.

---

## Citation

If you find this resource useful, please cite our survey:

```bibtex
@article{xue2026inferencetimesteering,
  title={Inference-Time Model Steering via Predictive-State Intervention: A Survey},
  author={Xue, Renhao and Wang, Rui and Wang, Yawei and Cui, Yueying and Qian, Yiyue and Vaddamanu, Praneetha and Song, Huan and Marlowe, Hannah},
  year={2026},
  journal={Research Square (Preprint)},
  doi={10.21203/rs.3.rs-9849186/v1},
  url={https://www.researchsquare.com/article/rs-9849186/v1}
}
```

## Contributing

We welcome contributions! Please submit a pull request with papers in the format shown above. Papers should satisfy the two defining criteria of inference-time steering:

1. **Tunability at serving time** via an external per-request signal applied to the fixed base computation
2. **Reversibility** — interventions are applied per-request and removed without state changes to the base model

Please identify the intervention location and runtime control, and follow the [scope and eligibility guidance](#scope-and-eligibility) when distinguishing core methods from boundary cases or contextual work. Use a primary paper or proceedings link and a verified venue; add code links when an author-maintained implementation is available.
