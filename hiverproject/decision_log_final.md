# Decision Log — SpotifyCares Support Agent

1. **Chose SpotifyCares over larger brands (Amazon, Apple)** as the target brand. Rationale: mid-size volume (43,265 replies) — large enough for real patterns, small enough to iterate quickly under a tight time budget. Smaller brands also tend to have more templated, repeatable reply patterns, which suits a retrieval-grounded approach.

2. **Scoped to single-message classification, not full multi-turn threads.** Early data exploration showed a large fraction of "customer messages" were actually mid-thread fragments (device info replies, "same issue as here," follow-up confirmations) rather than standalone complaints. Classifying these as first-class intents would have been meaningless ("what is the intent of 'Windows 10, Chrome'?"). Scoped the system to the initial/support-seeking message only.

3. **Defined 11 intents from direct data inspection, not assumption.** Started with a 30-message manual read, then validated candidate categories against keyword frequency across the full 41,697-message corpus before finalizing, to avoid missing major categories present only in the larger dataset.

4. **Used AI-assisted labeling for the 200-message golden set, with human review rather than fully manual labeling.** Given the time budget, all 200 messages were labeled using consistent LLM-applied judgment (not fully manual), with the first batch explicitly reviewed and approved before the same standard was applied to the rest. This is disclosed as a deliberate scope tradeoff, not presented as fully independent human labeling.

5. **Used few-shot prompting (Groq, `gpt-oss-20b`) instead of training a custom classifier.** No time budget for fine-tuning; an LLM with reference examples per intent gave a fast, reasonable-accuracy baseline (77.51%) suitable for the timeframe.

6. **Switched from `llama-3.1-8b-instant` to `gpt-oss-20b` mid-project** after discovering Groq retired the former model (2026-08-16). Required adding an explicit `reasoning_effort="low"` parameter and increasing `max_tokens`, since `gpt-oss` models silently return empty completions if their internal reasoning budget consumes the entire token allowance.

7. **Subsampled the retrieval knowledge base to 6,000 pairs (later 5,966) rather than using the full ~41,000.** The assignment explicitly permits and expects subsampling; this kept embedding generation and iteration fast without materially harming retrieval quality.

8. **Discovered and fixed data leakage between the knowledge base and evaluation set.** Both were independently sampled from the same message pool, causing ~20% of test messages to also exist verbatim in the retrieval index (producing misleadingly perfect similarity=1.00 retrievals). Fixed by explicitly excluding test-set messages from the knowledge base before final evaluation.

9. **Escalation rule combines two independent signals (intent risk category + retrieval confidence) rather than a single score.** A single confidence threshold would not have caught category-level risk (e.g., a confidently-classified billing message still shouldn't be auto-replied to), so risk category and grounding confidence are treated as independent, both-must-pass gates.

10. **Chose to always escalate Billing & Payment, Account & Login, and Premium & Subscription intents regardless of confidence.** These involve money or account access, where an incorrect auto-reply carries outsized real-world cost compared to, e.g., a wrong reply about playback bugs.

11. **Set retrieval similarity threshold at 0.75 for auto-handling**, after explicitly testing and reverting a lower 0.70 threshold. Kept the more conservative value deliberately rather than optimizing purely for a higher auto-handle rate, prioritizing safety over throughput for a first version.

12. **Only draft LLM replies for messages the escalation rule marks AUTO_HANDLE, not all messages.** Saves API quota and reflects realistic system design — no reason to have an LLM draft a full reply for a message a human will handle anyway.

13. **Used semantic embedding similarity to the REAL historical reply as an automated reply-quality proxy**, in addition to LLM-as-judge scoring. This gave a free, reference-based metric not dependent on further LLM calls, though it was explicitly noted as underweighting replies that are functionally correct but differently worded.

14. **Iterated on the LLM-judge's groundedness rubric after finding weak human agreement (r=0.39).** Added concrete fabrication examples and a falsifiability test to the prompt rather than discarding the LLM-judge approach entirely, which raised agreement to r=0.85 — treated as evidence that judge design, not just judge presence, matters for evaluation validity.

15. **Built both a trivial and a simple baseline for classification AND for reply drafting** (4 baselines total, not 2), to isolate how much value each pipeline stage (LLM classification over keyword rules; LLM rewriting over pure retrieval) actually adds, rather than only benchmarking the system as a whole.
