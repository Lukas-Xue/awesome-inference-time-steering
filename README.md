# Awesome Inference-Time Steering

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated paper list for **inference-time model steering** — methods that control large language models by intervening on their internal computations during generation, with strength tunable and the intervention reversible per request.

Based on our survey: [Inference-Time Model Steering via Predictive-State Intervention: A Survey](paper.pdf)

Feel free to suggest relevant papers in the following format.

```markdown
**Paper Title**
Authors. Venue Year. [paper](url) [code](url)
Note: brief description of contribution.
```

### Table of Contents
- [Latent-Level Methods](#latent-level-methods)
  - [Training-Free Contrastive Extraction](#training-free-contrastive-extraction)
  - [Supervised Probe Directions](#supervised-probe-directions)
  - [Learned Sparse Features (SAE-Based)](#learned-sparse-features-sae-based)
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

---

## Latent-Level Methods

Methods that modify hidden states $h_{\ell,t}$ inside the residual stream. Perturbations propagate through all subsequent layers, making these the most expressive but also carrying the largest side-effect surface.

### Training-Free Contrastive Extraction

1. **Steering Language Models With Activation Engineering (ActAdd)**
   Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid. arXiv 2024. [paper](https://arxiv.org/abs/2308.10248)
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
   Stolfo, Balachandran, Yousefi, Horvitz, Nushi. ICLR 2025. [paper](https://arxiv.org/abs/2410.12877)
   Note: steering vectors for instruction-following behavior.

6. **Comparing Bottom-Up and Top-Down Steering Approaches on In-Context Learning Tasks**
   Brumley, Kwon, Krueger, Krasheninnikov, Anwar. NeurIPS Workshop 2024. [paper](https://arxiv.org/abs/2411.07213)
   Note: compares prompt-derived vs. representation-derived steering on ICL tasks.

### Supervised Probe Directions

7. **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)**
   Li, Patel, Viegas, Pfister, Wattenberg. NeurIPS 2023. [paper](https://arxiv.org/abs/2306.03341)
   Note: trains linear probes on labeled truthfulness data; selects top-K heads; achieves strong steering at small perturbation ratio.

8. **Spectral Editing of Activations for Large Language Model Alignment (SEA)**
   Qiu, Zhao, Ziser, Korhonen, Ponti, Cohen. NeurIPS 2024. [paper](https://arxiv.org/abs/2405.09719)
   Note: identifies subspace of relevant directions via principal axes of activation differences.

### Learned Sparse Features (SAE-Based)

9. **Improving Steering Vectors by Targeting Sparse Autoencoder Features (SAE-TS)**
   Chalnev, Siu, Conmy. arXiv 2024. [paper](https://arxiv.org/abs/2411.02193)
   Note: uses SAE as a lens for measuring and reducing off-target side effects.

10. **Interpretable Steering of Large Language Models with Feature Guided Activation Additions (FGAA)**
    Soo, Guang, Teng, Balaganesh, Guoxian, Ming. ICLR Workshop 2025. [paper](https://arxiv.org/abs/2501.09929)
    Note: optimizes steering coefficients in SAE feature space for steering/coherence trade-off.

11. **CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features**
    Cho, Wu, Koshiyama. ICML 2026. [paper](https://arxiv.org/abs/2508.12535)
    Note: adaptive feature selection via correlation with task-success signal; introduces Side Effect Ratio (SER) metric.

12. **LF-Steering: Latent Feature Activation Steering for Enhancing Semantic Consistency in Large Language Models**
    Yang, Li, Wang, Zhou, Feng, Peng. arXiv 2025. [paper](https://arxiv.org/abs/2501.11036)
    Note: targets inconsistency-related SAE features to reduce semantic inconsistency.

13. **PaCE: Parsimonious Concept Engineering for Large Language Models**
    Luo, Ding, Chan, Thaker, Chattopadhyay, Callison-Burch, Vidal. NeurIPS 2024. [paper](https://openreview.net/forum?id=lOMHt16T8R)
    Note: disentangled concept removal via projection in SAE space.

14. **Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering (SpARE)**
    Zhao, Devoto, Hong, Du, Gema, Wang, He, Wong, Minervini. NAACL 2025. [paper](https://aclanthology.org/2025.naacl-long.264/)
    Note: mutual-information-based feature selection for knowledge control.

15. **SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs (DSG)**
    Muhamed, Bonato, Diab, Smith. ICML Workshop 2025. [paper](https://openreview.net/forum?id=8gFO7ebDLT)
    Note: dynamic classifier for selective suppression of unwanted knowledge.

16. **Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering (SDCV)**
    Zhao, Wu, Yang, Shen, Liu, Du. EACL Findings 2026. [paper](https://aclanthology.org/2026.findings-eacl.40/)
    Note: filters task-irrelevant features from concept vectors via SAE denoising.

17. **Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement (SAE-RSV)**
    Wang, Wu, Shu, Ma, Liu. arXiv 2025. [paper](https://arxiv.org/abs/2509.23799)
    Note: iterative feature refinement for improved steering success rates.

18. **SAIF: A Sparse Autoencoder Framework for Interpreting and Steering Instruction Following of Language Models**
    He, Zhao, Qiao, Yang, Payani, Ma, Du. arXiv 2025. [paper](https://arxiv.org/abs/2502.11356)
    Note: SAE-based interpretation and steering of instruction-following behavior.

19. **SCAR: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs**
    Härle, Friedrich, Brack, Deiseroth, Schramowski, Kersting. NeurIPS Workshop 2024. [paper](https://arxiv.org/abs/2411.07122)
    Note: conditional SAE architecture for concept-specific steering.

20. **Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms (STA)**
    Wang, Xu, Mao, Deng, Tu, Chen, Zhang. ACL 2025. [paper](https://aclanthology.org/2025.acl-long.1139/)
    Note: decomposes steering targets into atomic SAE features for robust control.

21. **Exploring the Personality Traits of LLMs through Latent Features Steering**
    Yang, Zhu, Liu, Hu, Li, Wang. arXiv 2024. [paper](https://arxiv.org/abs/2410.10863)
    Note: steers personality dimensions via SAE feature manipulation.

22. **A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering**
    Xie. arXiv 2025. [paper](https://arxiv.org/abs/2510.01246)
    Note: shows single top-1 SAE latents with token-wise decay can outperform top-k and mean activation difference.

23. **Analyze Feature Flow to Enhance Interpretation and Steering in Language Models**
    Laptev, Balagansky, Aksenov, Gavrilov. ICML 2025. [paper](https://arxiv.org/abs/2502.03032)
    Note: traces how SAE features evolve and compose across layers.

### Stability-Oriented Methods

24. **Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection**
    Dang, Ngo. arXiv 2026. [paper](https://arxiv.org/abs/2601.19375)
    Note: rotates hidden states preserving norm by construction; zero perplexity violations across 9 tested models.

25. **ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment**
    Zhao, Sun, Kong, Li, Wang, Jiang, Zhu, Abdelzaher, Choi, Li, Shao. ICLR 2026. [paper](https://arxiv.org/abs/2602.17560)
    Note: multi-step integration with barrier constraints; +5.7% TruthfulQA, +2.4% toxicity reduction over single-step baselines.

26. **Angular Steering: Behavior Control via Rotation in Activation Space**
    Vu, Nguyen. NeurIPS 2025 (Spotlight). [paper](https://arxiv.org/abs/2510.26243)
    Note: calibrated rotation angle tuned to task; norm-preserving by construction.

27. **Steering Without Side Effects: Improving Post-Deployment Control of Language Models (KL-then-Steer)**
    Stickland, Lyzhov, Pfau, Mahdi, Bowman. NeurIPS Workshop 2024. [paper](https://arxiv.org/abs/2406.15518)
    Note: fine-tunes LM to minimize KL divergence between steered/unsteered outputs on benign inputs.

28. **Towards Understanding Steering Strength**
    Taimeskhanov, Vaiter, Garreau. ICML 2026. [paper](https://arxiv.org/abs/2602.02712)
    Note: mathematical laws governing the relationship between steering magnitude and behavioral effect.

### Multi-Property and Adaptive Steering

29. **Multi-property Steering of Large Language Models with Dynamic Activation Composition**
    Scalena, Sarti, Nissim. BlackboxNLP 2024. [paper](https://aclanthology.org/2024.blackboxnlp-1.34/)
    Note: information-theoretic weighting for composing multiple steering vectors.

30. **Compositional Steering of Large Language Models with Steering Tokens**
    Radevski, Gashteovski, Hong, Lawrence, Glavas. ACL 2026. [paper](https://arxiv.org/abs/2601.05062)
    Note: learns compositional tokens for multi-behavior control in input space.

31. **Steer2Adapt: Dynamically Composing Steering Vectors Elicits Efficient Adaptation of LLMs**
    Han, Xu, Xuan, Song, Ouyang, Tian, Jiang, Qian, Jiang, Sun, Cui, Zhong, Liu, Ge, Han, You. ICLR Workshop 2026. [paper](https://arxiv.org/abs/2602.07276)
    Note: learns input-dependent weights for gated composition; reduces interference between properties.

32. **Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors (SADI)**
    Wang, Yang, Peng. ICLR 2025. [paper](https://arxiv.org/abs/2410.12299)
    Note: selects steering direction based on input semantics.

33. **Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment**
    Gadgil, Lin, Lee. arXiv 2026. [paper](https://arxiv.org/abs/2604.03867)
    Note: selects intervention layer adaptively based on input features.

34. **HyperSteer: Activation Steering at Scale with Hypernetworks**
    Sun, Baskaran, Wu, Sklar, Potts, Geiger. arXiv 2025. [paper](https://arxiv.org/abs/2506.03292)
    Note: generates steering vectors directly via hypernetworks for scalable multi-task steering.

35. **Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training (RICE)**
    Wang, Chen, Wang, He, Xu, Liang, Liu, Yao, Wang, Ma, Mi, Zhang, Tu, Li, Yu. NeurIPS 2025. [paper](https://arxiv.org/abs/2505.14681)
    Note: steers MoE expert selection to reinforce reasoning effort.

36. **Curvature Tuning: Provable Training-Free Model Steering from a Single Parameter**
    Hu, Gamba, Balestriero. NeurIPS 2025. [paper](https://arxiv.org/abs/2502.07783)
    Note: provable training-free steering via a single curvature parameter.

37. **Steering Information Utility in Key-Value Memory for Language Model Post-Training (InfoSteer)**
    Deng, Chang, Chen. NeurIPS 2025. [paper](https://arxiv.org/abs/2507.05158)
    Note: steers FFN key-value memory for post-training control.

### Human Preference Alignment

38. **Toward Preference-Aligned Large Language Models via Residual-Based Model Steering (PaLRS)**
    La Cava, Tagarelli. arXiv 2025. [paper](https://arxiv.org/abs/2509.23982)
    Note: residual-based steering for preference alignment without retraining.

39. **Self-Improving Model Steering (SIMS)**
    Zhu, Wang, Jiang, Liang, Wang. arXiv 2025. [paper](https://arxiv.org/abs/2507.08967)
    Note: iterative self-improvement loop for steering vector refinement.

40. **Improved Representation Steering for Language Models (RePS)**
    Wu, Yu, Arora, Manning, Potts. NeurIPS 2025 (Spotlight). [paper](https://arxiv.org/abs/2505.20809)
    Note: reference-free bi-directional preference steering.

41. **From Weights to Activations: Is Steering the Next Frontier of Adaptation?**
    Ostermann, Gurgurov, Baeumel, Hedderich, Lapuschkin, Samek, Schmitt. ACL 2026. [paper](https://arxiv.org/abs/2604.14090)
    Note: unified framework positioning steering within the broader adaptation landscape.

---

## Logit-Level Methods

Methods that modify output scores $\ell_t$ directly. Because there are no downstream layers to destabilize, these are more stable than latent interventions but less expressive.

42. **DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts**
    Liu, Sap, Lu, Swayamdipta, Bhagavatula, Smith, Choi. ACL 2021. [paper](https://aclanthology.org/2021.acl-long.522/)
    Note: contrastive expert/anti-expert logit bias; cancels shared linguistic structure to amplify attribute signals.

43. **GeDi: Generative Discriminator Guided Sequence Generation**
    Krause, Gotmare, McCann, Keskar, Joty, Socher, Rajani. EMNLP Findings 2021. [paper](https://aclanthology.org/2021.findings-emnlp.424/)
    Note: Bayes-factor ratio from class-conditional LMs for generative discrimination.

44. **FUDGE: Controlled Text Generation With Future Discriminators**
    Yang, Klein. NAACL 2021. [paper](https://aclanthology.org/2021.naacl-main.276/)
    Note: future-attribute discriminator predicts downstream satisfaction from partial generations.

45. **Steering Language Models Before They Speak: Logit-Level Interventions**
    An, Park, Jin, Han. arXiv 2026. [paper](https://arxiv.org/abs/2601.10960)
    Note: z-normalized log-odds biases from labeled corpora; training-free multi-task control.

---

## Distribution-Level Methods

Methods that interpolate the model's output distribution with an external distribution after the forward pass. The base model's computation is unmodified.

46. **Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM)**
    Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. ICLR 2020. [paper](https://openreview.net/forum?id=HklBjCEKvH)
    Note: retrieves nearest neighbors from a datastore to interpolate with model distribution; domain adaptation by swapping datastore.

47. **Improving Neural Language Models with a Continuous Cache**
    Grave, Joulin, Usunier. ICLR 2017. [paper](https://openreview.net/forum?id=B184E5qee)
    Note: stores recent hidden activations; learned gate interpolates cache distribution with model output.

48. **Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models**
    Cao, Wang, Wei, Guo, Chen, Zhou, Lin. NeurIPS 2025. [paper](https://arxiv.org/abs/2508.09874)
    Note: parametric module replacing explicit retrieval; faster inference while maintaining distribution-interpolation interface.

49. **MLP Memory: A Retriever-Pretrained Memory for Large Language Models**
    Wei, Cao, Wang, Kai, Guo, Zhou, Lin. ICLR 2026. [paper](https://arxiv.org/abs/2508.01832)
    Note: MLP-based parametric memory replacing kNN retrieval for faster inference.

---

## Decoding-Level Methods (Boundary)

Methods that modify the token selection rule $\mathcal{D}$ rather than the predictive state. Included as boundary cases.

50. **Contrastive Decoding: Open-ended Text Generation as Optimization**
    Li, Holtzman, Fried, Liang, Eisner, Hashimoto, Zettlemoyer, Lewis. ACL 2023. [paper](https://aclanthology.org/2023.acl-long.687/)
    Note: favors tokens where expert disagrees with amateur model.

51. **NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics**
    Lu, Welleck, West, Jiang, Kasai, Khashabi, Le Bras, Qin, Yu, Zellers, Smith, Choi. NAACL 2022. [paper](https://aclanthology.org/2022.naacl-main.57/)
    Note: enforces hard lexical constraints via lookahead search.

---

## Safety and Robustness

52. **Refusal in Language Models Is Mediated by a Single Direction**
    Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. NeurIPS 2024. [paper](https://arxiv.org/abs/2406.11717)
    Note: shows refusal behavior is mediated by a single removable direction; highlights safety fragility.

53. **Steering Externalities: Benign Activation Steering Unintentionally Increases Jailbreak Risk for Large Language Models**
    Xiong, He, Chen, Ko, Ho. arXiv 2026. [paper](https://arxiv.org/abs/2602.04896)
    Note: demonstrates that even benign steering vectors can increase jailbreak vulnerability as a side effect.

54. **Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions**
    Goyal, Daumé III. EACL 2026. [paper](https://arxiv.org/abs/2602.06256)
    Note: evaluates robustness of steering under distribution shift; reveals brittleness of common approaches.

55. **Toward Universal Steering and Monitoring of AI Models**
    Beaglehole, Radhakrishnan, Boix-Adsera, Belkin. Science 2026. [paper](https://www.science.org/doi/abs/10.1126/science.aea6792)
    Note: universal linear-representation-based steering and monitoring framework.

56. **Jailbreaking the Matrix: Nullspace Steering for Controlled Model Subversion (HMNS)**
    Pramanik, Maliha, Jha, Jha. arXiv 2026. [paper](https://arxiv.org/abs/2604.10326)
    Note: nullspace steering jailbreak attack exploiting activation geometry.

57. **The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness**
    Abdelnabi, Salem. NeurIPS 2025 (Spotlight). [paper](https://arxiv.org/abs/2505.14617)
    Note: models alter behavior under evaluation; steering for test-awareness control.

58. **DSO: Direct Steering Optimization for Bias Mitigation**
    Monteiro Paes, Sivakumar, Wang, Fedzechkina, Theobald, Zappella, Apostoloff. CVPR 2026. [paper](https://arxiv.org/abs/2512.15926)
    Note: RL + subspace optimization for bias mitigation via steering.

59. **Steered LLM Activations are Non-Surjective**
    Mishra, Khashabi, Liu. ICLR Workshop 2026. [paper](https://arxiv.org/abs/2604.09839)
    Note: formalizes that steering can produce residual-stream states unreachable by any discrete prompt.

---

## Evaluation and Benchmarks

60. **AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders**
    Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts. ICML 2025. [paper](https://openreview.net/forum?id=K2CckZjNy0)
    Note: prompting outperforms representation-based steering on concept-level tasks (Gemma-2-2B/9B).

61. **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability**
    Karvonen, Rager, Lin, Tigges, Bloom, Chanin, Lau, Farrell, McDougall, Ayonrinde, Till, Wearden, Conmy, Marks, Nanda. ICML 2025. [paper](https://openreview.net/forum?id=qrU3yNfX0d)
    Note: SAE proxy metrics (reconstruction loss, sparsity) do not reliably predict steering performance.

62. **Evaluating Steering Techniques Using Human Similarity Judgments**
    Studdiford, Rogers, Suresh, Mukherjee. arXiv 2025. [paper](https://arxiv.org/abs/2505.19333)
    Note: human evaluation corroborates AxBench findings on steering effectiveness.

---

## Representation Engineering and Interpretability

63. **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**
    Anthropic. 2023. [paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
    Note: foundational work on SAE dictionary learning for interpretable feature decomposition.

64. **Scaling and Evaluating Sparse Autoencoders**
    Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike, Wu. ICLR 2025. [paper](https://arxiv.org/abs/2406.04093)
    Note: scaling laws and evaluation methodology for SAEs.

65. **Not All Language Model Features Are One-Dimensionally Linear**
    Engels, Michaud, Liao, Gurnee, Tegmark. ICLR 2025. [paper](https://arxiv.org/abs/2405.14860)
    Note: some concepts are multi-dimensional manifolds (circles), limiting contrastive-vector methods.

66. **The Linear Representation Hypothesis and the Geometry of Large Language Models**
    Park, Choe, Veitch. arXiv 2023. [paper](https://arxiv.org/abs/2311.03658)
    Note: theoretical grounding for why linear directions encode semantic concepts.

67. **The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets**
    Marks, Tegmark. arXiv 2023. [paper](https://arxiv.org/abs/2310.06824)
    Note: empirical evidence for linear truth representations enabling steering.

68. **Discovering Latent Knowledge in Language Models Without Supervision**
    Burns, Ye, Klein, Steinhardt. ICLR 2023. [paper](https://arxiv.org/abs/2212.03827)
    Note: unsupervised discovery of truth-related directions in activation space.

69. **Brain-Grounded Axes for Reading and Steering LLM States**
    Andric. arXiv 2025. [paper](https://arxiv.org/abs/2512.19399)
    Note: uses brain-activity-derived axes for interpretable steering directions.

70. **Steering Large Language Models to Evaluate and Amplify Creativity**
    Olson, Ratzlaff, Hinck, Tseng, Lal. NeurIPS Workshop 2024 (Spotlight). [paper](https://arxiv.org/abs/2412.06060)
    Note: contrastive steering for creativity evaluation and amplification.

71. **The Logical Implication Steering Method for Conditional Interventions on Transformer Generation (LIMS)**
    Kalajdzievski. ICML 2025. [paper](https://arxiv.org/abs/2502.03618)
    Note: logic-rule-based conditional steering via logical implications.

72. **DEAL: Disentangling Transformer Head Activations for LLM Steering**
    Zhan, Liu, Lu, Feng, Xie, Cao, Wu. ACL Workshop 2025. [paper](https://openreview.net/forum?id=HvXDK8Ryst)
    Note: causal attribution for disentangling per-head contributions to steering.

73. **REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering**
    Zhan, Liu, Xie, Cao, Wu. ICLR 2026. [paper](https://arxiv.org/abs/2506.08359)
    Note: VQ-AE-based precise localization of steering targets.

74. **Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions (CDAS)**
    Bao, Zhang, Chen, Su, Cai, Peng, Sun, Weng, Yan, Yin. ICLR 2026. [paper](https://arxiv.org/abs/2602.05234)
    Note: distribution-matching approach for faithful bi-directional control.

75. **Neologism Learning as a Parameter-Efficient Alternative to Fine-Tuning for Model Steering**
    Park, Ramamurthi, Terry. arXiv 2025. [paper](https://arxiv.org/abs/2512.18551)
    Note: learns steering via neologism tokens in input space.

---

## Multimodal Steering

76. **Automating Steering for Safe Multimodal Large Language Models**
    Wu, Wang, Xu, Cao, Oo, Hooi, Deng. EMNLP 2025. [paper](https://aclanthology.org/2025.emnlp-main.41/)
    Note: automated steering for safety in vision-language models.

77. **Learning to Steer: Input-Dependent Steering for Multimodal LLMs**
    Parekh, Khayatan, Shukor, Dapogny, Newson, Cord. NeurIPS 2025. [paper](https://arxiv.org/abs/2508.12815)
    Note: input-dependent steering applied to multimodal architectures.

78. **Analyzing Finetuning Representation Shift for Multimodal LLMs Steering**
    Khayatan, Shukor, Parekh, Dapogny, Cord. ICCV 2025. [paper](https://openaccess.thecvf.com/content/ICCV2025/html/Khayatan_Analyzing_Finetuning_Representation_Shift_for_Multimodal_LLMs_Steering_ICCV_2025_paper.html)
    Note: analyzes how finetuning shifts representations and implications for steering in multimodal LLMs.

---

## Related Surveys

79. **A Survey of Controllable Text Generation Using Transformer-based Pre-trained Language Models**
    Zhang, Song, Li, Zhou, Song. ACM Computing Surveys 2023. [paper](https://arxiv.org/abs/2201.05337)
    Note: broad controllable generation covering training-time and inference-time approaches.

80. **A Comprehensive Study of Knowledge Editing for Large Language Models**
    Zhang, Yao, Tian, Wang, Deng, Wang, Xi, Mao, Zhang, Ni, Cheng, Xu, Xu, Gu, Jiang, Xie, Huang, Liang, Zhang, Zhu, Zhou, Chen. arXiv 2024. [paper](https://arxiv.org/abs/2401.01286)
    Note: persistent factual update/weight editing (excluded from steering but related).

81. **Representation Engineering for Large-Language Models: Survey and Research Challenges**
    Bartoszcze, Munshi, Sukidi, Yen, Yang, Williams-King, Le, Asuzu, Maple. arXiv 2025. [paper](https://arxiv.org/abs/2502.17601)
    Note: latent representation engineering survey; does not cover logit/distribution-level methods.

82. **Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models**
    Zhang, Zhang, Wang, et al. arXiv 2026. [paper](https://arxiv.org/abs/2601.14004)
    Note: organized around Interpretable Objects (what is targeted) rather than intervention location.

83. **A Survey on LLM Inference-Time Self-Improvement**
    Dong, Teleki, Caverlee. arXiv 2024. [paper](https://arxiv.org/abs/2412.14352)
    Note: broader inference-time improvement including search, verification, refinement.

---

## Toolkits and Frameworks

84. **EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models**
    Xu, Wang, Xu, Xu, Wang, Deng, Yao, Zheng, Chen, Zhang. EMNLP Demos 2025. [paper](https://aclanthology.org/2025.emnlp-demos.38/)
    Note: unified toolkit supporting multiple steering methods with a common interface.

---

## Background and Foundations

85. **Attention is All You Need**
    Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. NeurIPS 2017.
    Note: transformer architecture underlying all steering methods.

86. **A Mathematical Framework for Transformer Circuits**
    Elhage, Nanda, Olsson, et al. Anthropic 2021. [paper](https://transformer-circuits.pub/2021/framework/index.html)
    Note: residual stream formalization enabling principled activation interventions.

87. **Locating and Editing Factual Associations in GPT (ROME)**
    Meng, Bau, Andonian, Belinkov. NeurIPS 2022. [paper](https://arxiv.org/abs/2202.05262)
    Note: localization of factual associations to MLP layers; foundational for steering targets.

88. **Transformer Feed-Forward Layers Are Key-Value Memories**
    Geva, Schuster, Berant, Levy. EMNLP 2021. [paper](https://arxiv.org/abs/2012.14913)
    Note: MLP layers as key-value stores; motivates feed-forward interventions.

89. **Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small**
    Wang, Variengien, Conmy, Shlegeris, Steinhardt. ICLR 2023. [paper](https://arxiv.org/abs/2211.00593)
    Note: mechanistic circuit discovery enabling targeted interventions.

90. **Towards Best Practices of Activation Patching in Language Models: Metrics and Methods**
    Zhang, Nanda. ICLR 2024. [paper](https://arxiv.org/abs/2309.16042)
    Note: best practices for activation patching; directly relevant to steering methodology.

---

## Boundary Cases (Not Steering, but Related)

Included for context. These methods violate at least one defining criterion of steering (per-request tunability or reversibility without retraining).

91. **CTRL: A Conditional Transformer Language Model for Controllable Generation**
    Keskar, McCann, Varshney, Xiong, Socher. arXiv 2019. [paper](https://arxiv.org/abs/1909.05858)
    Note: control codes at pretraining; not tunable at serving time.

92. **Plug and Play Language Models: A Simple Approach to Controlled Text Generation (PPLM)**
    Dathathri, Madotto, Lan, Hung, Frank, Molino, Yosinski, Liu. ICLR 2020. [paper](https://openreview.net/forum?id=H1edEyBKDS)
    Note: hybrid; uses gradient updates during generation.

93. **Prefix-Tuning: Optimizing Continuous Prompts for Generation**
    Li, Liang. ACL 2021. [paper](https://aclanthology.org/2021.acl-long.353/)
    Note: learned continuous prefixes; requires retraining to change behavior.

94. **The Power of Scale for Parameter-Efficient Prompt Tuning**
    Lester, Al-Rfou, Constant. EMNLP 2021. [paper](https://aclanthology.org/2021.emnlp-main.243/)
    Note: soft prompt tuning; learned parameters not adjustable at serving time.

95. **ReFT: Representation Finetuning for Language Models**
    Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts. NeurIPS 2024. [paper](https://arxiv.org/abs/2404.03592)
    Note: learns intervention parameters via gradient descent; not runtime-adjustable.

96. **LoRA: Low-Rank Adaptation of Large Language Models**
    Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. ICLR 2022. [paper](https://openreview.net/forum?id=nZeVKeeFYf9)
    Note: PEFT baseline; adapter swap not per-request tunable.

97. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
    Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou. NeurIPS 2022. [paper](https://openreview.net/forum?id=_VjQlMeSB_J)
    Note: prompting baseline; no internal state modification.

98. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**
    Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan. NeurIPS 2023. [paper](https://openreview.net/forum?id=5Xc1ecxO1h)
    Note: search-based prompting baseline.

---

## Citation

If you find this resource useful, please cite our survey:

```bibtex
@article{xue2026inferencetimesteering,
  title={Inference-Time Model Steering via Predictive-State Intervention: A Survey},
  author={Xue, Renhao and Wang, Rui and Wang, Yawei and Cui, Yueying and Qian, Yiyue and Vaddamanu, Praneetha and Song, Huan and Marlowe, Hannah R},
  year={2026}
}
```

## Contributing

We welcome contributions! Please submit a pull request with papers in the format shown above. Papers should satisfy the two defining criteria of inference-time steering:
1. **Tunability at serving time** via an external per-request signal applied to the fixed base computation
2. **Reversibility** — interventions are applied per-request and removed without state changes to the base model
