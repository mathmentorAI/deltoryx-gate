# Scaling Adaptive Agents in Regulated Environments (`Harness²` + `Deltoryx Gate`)

Recursive agent frameworks such as [`google-research/harness2`](https://github.com/google-research/harness2) optimize **how an agent operates**—iteratively refining prompts, scripts, tools, memory, and retrieval strategies through a *Propose → Probe → Compose* loop without modifying model weights.

In regulated environments—such as high-risk AI use cases under the **EU AI Act (RIA)** or regulated financial/legal workflows—that adaptability creates a fundamental tension if the same loop also controls what counts as a valid answer.

---

## 1. The Verification Bottleneck

Generating outputs is no longer the primary bottleneck for AI agents; **verifying them at the same pace** is:

- If an adaptive harness makes an agent 10× more productive while human reviewers still have to audit every output, human oversight becomes the bottleneck (or degrades into rubber-stamping).
- Conversely, if the adaptive loop is allowed to rewrite its own acceptance rules to make probes pass, compliance guarantees erode silently.

**Deltoryx Gate** resolves this by making verification **deterministic, cheap, and operator-owned**, so verification scales alongside agent generation:

- Clear, contract-compliant outputs are authorized automatically (`PASS`) with a verifiable receipt (`verify_receipt`).
- Extraction or grounding failures are returned to the agent immediately with structured diagnostics (`RETRY`, zero human wait).
- Only genuine ambiguities, contradictions, unrepresentable conditions, or out-of-policy values escalate to a human (`ESCALATE`).

---

## 2. Separation of Powers: Adaptive Layer vs. Normative Boundary

| Layer | Role | Mutable by Agent / `Harness²`? | Key Components |
| :--- | :--- | :--- | :--- |
| **Adaptive Harness** (`Harness²`) | Search, chunking, context recovery, extraction prompting, tool orchestration | **Yes** (optimizes capability) | `Propose → Probe → Compose`, Experience Library, `LLMProposalReader` |
| **Normative Boundary** (`Deltoryx Gate v4`) | Domain schema, fidelity checks, scope/precedence resolution, authorization gate | **No** (frozen by operator) | `Vocabulary`, `AcceptancePolicy`, `ClauseRelation`, `verify_extraction_fidelity()`, `authorize_operation()`, `verify_receipt()` |

In addition, `Deltoryx Gate v4` gives `Harness²` an **objective, deterministic signal during its *Probe* step**: the composer can measure which prompt or retrieval edit reduces `UNCAPTURED_CONDITION`, `POLARITY_MISMATCH`, or `DIRECTION_OF_CHANGE_MISMATCH` findings without ever loosening the acceptance threshold.

---

## 3. Technical Alignment with Regulated Frameworks (e.g., EU AI Act / RIA)

When building systems for high-risk or regulated domains (credit, employment, essential services, structured finance, commercial contracts), `Harness²` + `Deltoryx Gate v4` provides a concrete technical control surface:

- **Effective Human Oversight (e.g., EU AI Act Art. 14):** `ESCALATE` (`requires_human=True`) and `RETRY` enforce explicit abstention on unread documents, `unknown_condition`, `unrepresentable_condition`, or equal-tier contradictions—preventing silent automation over incomplete evidence.
- **Record-Keeping & Traceability (Arts. 12–13):** Every authorization decision binds back to exact character offsets (`EvidenceRef`, `FieldEvidence`), explicit `Vocabulary` and `AcceptancePolicy` versions, and HMAC-authenticated receipts (`verify_receipt(..., allow_unkeyed=False)`).
- **Risk Management & Accuracy Separation (Arts. 9, 15):** The 3-layer evaluation harness (`benchmarks/b2b_live_eval.py`) separates deterministic reasoning over typed facts, extraction fidelity (including dual-reader shared-error measurement via `evaluate_dual_readers`), and end-to-end risk-coverage (`selective_risk` vs. `automation_coverage`).

---

## 4. Honest Scope & Boundaries

Using `Harness²` with `Deltoryx Gate` **does not by itself grant legal or regulatory certification** under the EU AI Act (RIA), SEC rules, or any other statute:

- Regulatory compliance also requires organizational governance, data/privacy controls (e.g., GDPR), cybersecurity, bias/fairness assessments, conformity assessments where applicable, and empirical validation on live domain documents (`--live`).
- What `Deltoryx Gate` provides is the **deterministic verification and authorization boundary** that allows teams to adopt adaptive agent harnesses without surrendering auditability, policy control, or abstention guarantees.
