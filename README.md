# Beyond-Static-Artifacts: An Evolutionary Framework for Synthetic Claim Generation
With the generative capabilities of large language models (LLMs) reshaping the information ecosystem, the concern with the sociological validity of claim detection benchmarks is increasing. Current claim detection benchmarks predominantly treat claims as static textual artifacts, overlooking the sociological etiology of how information naturally emerges and mutates. In this paper, we propose an evolutionary paradigm that models claims as socially evolving entities. Specifically, we introduce a socially generative framework for synthetic claim generation, a multi-agent simulation grounded in the Open Claims Model. By decomposing claims into context, utterance, and proposition, our approach enables the precise simulation of unmitigated propagation to capture truth decay, and intervened propagation with multi-auditor oversight for targeted generation. Furthermore, we propose the background-user-perspective (BUP) framework, which reformulates check-worthiness as a condition-dependent probability rooted in the social environment. Experiments on our datasets verify the data quality and reveal how network topology and user attributes systematically shape veracity drift.
## Architecture
![Beyond-Static-Artifacts banner](assets/figure2.png)
A social simulation framework for claim evolution and customized check-worthiness detection.
## Premilinary
### Open Claims Model (OCM)
OCM represents a claim through three layers—context, utterance, and proposition—to disentangle narrative background, linguistic form, and factual content. It further parameterizes these layers with interpretable dimensions (e.g., stance, causality, veracity) to trace and control how claims evolve across social networks. Details can be found in the `assets` folder.
### Simulated Social Environment
This environment simulates claim evolution in a social environment with heterogeneous agents modeled from real users’ psychometric traits and sociocultural groups. Claims propagate over different network topologies (random, clustered, scale-free) to study how social structure drives information mutation and veracity drift. The details of the three types of networks and user information can be found in the `assets` folder.
## Methodology
### Context-Anchored Initialization

### Socialized Propagation and Evolution
- **Unmitigated Propagation:** 
- **Intervened Propagation:**
### Customized Check-Worthiness Evaluation

### Quality Control

