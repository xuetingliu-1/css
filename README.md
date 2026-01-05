# Do Prompting Strategies Generalize to Medical Domain? A Comparative Analysis and an Order Invariant Clinical Scenario Simulation Method



We propose Clinical Scenario Simulation (CSS), a two-stage inference framework that decouples reasoning exploration from answer selection via counterfactual generation and order-invariant aggregation. This method enables general-purpose LLMs (e.g., Llama-3, Ministral) to rival or outperform specialized medical LLMs (e.g., Med42, UltraMedical) on benchmarks like MedQA and MedBullets.

Repository Structure
The codebase is organized by model family to accommodate specific chat templates and tokenization requirements.

```text
.
├── medbullets/       # Data scripts and files for MedBullets (Op4/Op5)
├── medqa/            # Data scripts and files for MedQA (USMLE)
├── src_llama/        # Inference scripts for Llama-3.1-8B (General)
├── src_mistral/      # Inference scripts for Ministral-3-8B (General)
├── src_qwen/         # Inference scripts for Qwen2.5-7B (General)
├── src_medins/       # Inference scripts for MMed-Llama-3 (Medical)
├── src_meerkat/      # Inference scripts for Meerkat-Llama-3 (Medical)
├── src_ultra/        # Inference scripts for UltraMedical (Medical)
└── README.md

