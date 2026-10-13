# NovaTech RAG System Notes

## 1. Theory 

* **What is an embedding?** 
An embedding is a numerical representation of text within a high-dimensional vector space. It goes beyond simple keyword matching by capturing the semantic meaning and context of the words. Phrases that mean similar things (like "refund" and "money back") are mathematically mapped closer together in this vector space, allowing the system to find relevant documents even if the exact vocabulary doesn't match the user's question.

* **Why use RAG instead of putting all the documents into the prompt?**
Injecting all 14 documents into the LLM prompt for every single question is highly inefficient. It consumes massive amounts of tokens (costing more money and taking longer to process) and often leads to context degradation, where the LLM forgets or skips over rules buried in the middle of a massive text block. RAG solves this by using a vector search to pull only the top 3 most relevant paragraphs, keeping the LLM prompt focused, cheap, and fast.

* **What is the difference between RAG and fine-tuning? When would you use each?**
RAG gives an LLM access to external, changing facts at runtime without altering the model itself. Fine-tuning actually updates the neural network's internal weights by training it on a specific dataset. 
*Use RAG* when the model needs to reference strict, factual data that changes frequently (like store hours or policy updates). 
*Use fine-tuning* when you need the model to adopt a specific writing style, tone, or output a strict format (like complex JSON structures).

* **Name two ways a RAG system can give a wrong answer.**
1. **Retrieval Failure:** The vector database fails to pull the correct document chunk into the top results (e.g., the chunk size split a rule in half, or the question didn't semantically trigger the right file). If the LLM doesn't get the facts, it can't answer.
2. **Generation Hallucination:** The correct document is retrieved, but the LLM ignores it, misinterprets the text, or blends it with its own pre-trained knowledge to make up an incorrect answer.

---

## 2. Results Table

| Question | AI Answer | Source Cited | Correct? (Yes/No) |
| :--- | :--- | :--- | :--- |
| 1. Warranty on TV? | 24 months | warranty.md | Yes |
| 2. Days to return? | 30 days under v2.0 | returns_policy_v2.0.md | Yes |
| 3. Bank transfer refund time? | 7–10 business days | refunds.md | Yes |
| 4. Deliver to Kuwait? | No, Jordan only | shipping_and_delivery.md | Yes |
| 5. Opened earbuds return? | No, hygiene seal broken unless defective | returns_policy.md, returns_policy_v1.3.md | Yes |
| 6. Broken order arrived? | Report within 48 hours, free courier pickup in 2 days | damaged_or_wrong_items.md | Yes |
| 7. Delivery to Irbid (40 JOD)? | 5 JOD, 2-3 business days | shipping_and_delivery.md | Yes |
| 8. CEO of NovaTech? | Refused to answer (no info) | None | Yes |
| 9. Laptop battery warranty? | 12 months only | warranty.md | Yes |
| 10. Student discount? | Refused to answer (no info) | None | Yes |
| 11. Irbid store open Friday? | Closed on Friday | stores_and_service_center.md | Yes |
| 12. Gold member headphones? | Refused to answer (no info) | returns_policy.md | **No** (Retrieval Failure) |

---

## 3. Retrieval Check (hit@3)

* **Chunk Size 500 characters:** The correct document was in the top 3 chunks 6 out of 10 times (60.0%).
* **Chunk Size 250 characters:** The correct document was in the top 3 chunks 6 out of 10 times (60.0%).
* **Conclusion:** I selected a chunk size of 500 characters with an overlap of 50. While both sizes yielded a 60% hit rate for the specific test queries, the 500-character chunks proved superior for generation. Many of NovaTech's policies contain tables (like warranty lengths) followed by immediate exceptions (like battery limits). The 250-character chunks severed these exceptions from their categories, while the 500-character chunks preserved the compound context necessary for the LLM to provide accurate answers.

---

## 4. Problems Found & Solutions

* **Problem 1: Hallucinations on Out-of-Domain Queries.** Initially, the system attempted to guess or fabricate answers for queries not covered in the documents, such as the CEO of NovaTech (Q8) and student discounts (Q10).
* **Solution:** I implemented strict negative-constraint prompting, forcing the LLM to output a standardized refusal ("The provided documents do not contain information to answer this question.") if the retrieved context lacked explicit facts.

* **Problem 2: Multi-Document Retrieval Failure (Question 12).** The system failed to answer if a Gold member could return headphones after 40 days. 
* **Solution / Observation:** This occurred because the `k=3` retrieval limit pulled the standard return policies but missed the `novapoints_loyalty.md` document, which contains the 45-day Gold tier exception. To fix this in a future iteration, the `k` value (number of chunks retrieved) should be increased to 5 or 7, or a re-ranking model could be implemented to better prioritize overlapping policy documents.