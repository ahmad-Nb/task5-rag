# Task 5: RAG System Notes

## 1. Theory

* **Embedding:** A numerical representation of text in a high-dimensional vector space that captures semantic meaning.
* **RAG vs. Prompting All Docs:** Putting all documents in the prompt causes high token costs, slower processing, and "lost in the middle" context degradation. RAG is cheaper, faster, and maintains focused context.
* **RAG vs. Fine-Tuning:** RAG retrieves external, dynamic knowledge without altering the model's underlying parameters. Fine-tuning permanently adjusts the model's internal weights to teach specific behaviors, tones, or output structures. Use RAG for updating facts; use fine-tuning for consistent formatting or domain-specific styles.
* **RAG Failure Modes:** 
  1. **Retrieval Failure:** The vector database misses the correct document. 
  2. **Generation Hallucination:** The LLM ignores retrieved facts or fabricates information.

## 2. Results Table

| Q | AI Answer | Source Cited | Correct? |
| :--- | :--- | :--- | :--- |
| 1 | 24 months | warranty.md | Yes |
| 2 | 30 days under v2.0 | returns_policy_v2.0.md | Yes |
| 3 | 7–10 business days | refunds.md | Yes |
| 4 | No, Jordan only | shipping_and_delivery.md | Yes |
| 5 | No, unless defective | returns_policy.md, returns_policy_v1.3.md | Yes |
| 6 | Report 48hrs, pickup 2 days | damaged_or_wrong_items.md | Yes |
| 7 | 5 JOD, 2-3 business days | shipping_and_delivery.md | Yes |
| 8 | Refused (no info) | None | Yes |
| 9 | 12 months only | warranty.md | Yes |
| 10 | Refused (no info) | None | Yes |
| 11 | Closed on Friday | stores_and_service_center.md | Yes |
| 12 | Refused (no info) | returns_policy.md | No |

## 3. Retrieval Check

**Ground Truth Documents:**
1. `warranty.md`
2. `returns_policy_v2.0.md`
3. `refunds.md`
4. `shipping_and_delivery.md`
5. `returns_policy_v2.0.md`
6. `damaged_or_wrong_items.md`
7. `shipping_and_delivery.md`
8. None (unanswerable)
9. `warranty.md`
10. None (unanswerable)
11. `stores_and_service_center.md`
12. `novapoints_loyalty.md`

**Hit@3 Performance (10 evaluable queries):**
* **Chunk size 500 (overlap 50):** 6/10 (60%)
* **Chunk size 250 (overlap 30):** 6/10 (60%)

**Selection & Justification:** 
**500 characters.** It prevents splitting policy tables from their immediate exceptions (e.g., a warranty length separated from a battery limitation), preserving complete logical context for the LLM.

## 4. Problems & Solutions

* **Hallucinations on Out-of-Domain Queries:** The model initially guessed answers for unanswerable questions (Q8, Q10). 
  * *Solution:* Applied strict negative-constraint prompting to force a standardized refusal.
* **Retrieval Failure on Compound Rules:** Missed the specific Gold member return policy (Q12) because `k=3` filled the context window with general return policies before grabbing the loyalty document. 
  * *Solution:* Acknowledged that a future iteration requires increasing the `k` limit or implementing a re-ranker.