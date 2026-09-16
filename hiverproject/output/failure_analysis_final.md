# Failure Analysis — SpotifyCares Support Agent

## Failure Mode 1: Premium ↔ Billing intent confusion (recall problem)

**Example:** *"y'alls system for management is crap I got charged for our family plan I expect my family to be able to listen. Fix ur system"*
True intent: Premium & Subscription → Predicted: Billing & Payment

**Metric evidence:** Premium & Subscription had the classifier's lowest recall (0.39) — of all intents, it's most often missed, usually mislabeled as Billing & Payment.

**Hypothesis:** Premium/subscription problems frequently surface *through* a billing symptom (a charge, a failed payment), so the surface vocabulary overlaps heavily with genuine billing complaints. The few-shot classifier likely keys on payment-related keywords without distinguishing "the charge itself is wrong" (Billing) from "I paid correctly but the subscription isn't functioning" (Premium).

---

## Failure Mode 2: Operational-status hallucination in auto-handled replies

**Example:** *"Spotify is DOWN omgggg, what am I gonna do"* → drafted reply: *"Our tech team is on it right now. Try logging out, restarting your device..."*

**Metric evidence:** This message scored 0.82 retrieval similarity and passed as AUTO_HANDLE — meaning the escalation safety net did not catch it.

**Hypothesis:** The retrieved historical examples contained vague reassurance ("hopefully soon"), and the LLM extrapolated that into a specific, confident operational claim ("our tech team is on it") that has no support in the retrieved evidence. High retrieval similarity measures *topical* match, not *factual* safety — a message can be topically well-matched while still inviting a confidently fabricated response.

---

## Failure Mode 3: Data leakage between knowledge base and evaluation set

**Finding:** 34 of 169 test messages (20.1%) also existed verbatim in the initial 6,000-message retrieval knowledge base, since both were randomly sampled from the same 41,697-message pool.

**Hypothesis:** Independent random sampling from a shared finite pool, without an explicit exclusion constraint, produces overlap proportional to the sampling fractions (~14% expected here, 20% observed — within normal variance). This is a pipeline design flaw, not a data quality issue — caught and fixed before final results by excluding test-set messages from the knowledge base.

---

## Failure Mode 4: Generic triage instead of direct answers to specific factual questions

**Examples:**
- *"@115888 where is the perfect velvet😔"* (asking about a specific album) → drafted: *"we'll have it available to you as soon as it's available to us"* (non-answer)
- *"what's the presale code for @26720???"* → drafted: *"codes are sent out via email to a select group"* (fabricated + doesn't answer)

**Metric evidence:** LLM-judge actionability score was the lowest of the 4 rubric criteria (3.33–3.42/5), with repeated notes citing "no clear next step."

**Hypothesis:** The knowledge base is dominated by generic troubleshooting/DM-redirect replies (the most common historical pattern), so retrieval and drafting are biased toward that generic pattern even when a customer's question calls for a specific factual answer the system doesn't actually have. The system has no mechanism to recognize "I don't know the specific answer" versus "I can give a generic helpful response."

---

## Failure Mode 5: LLM-as-judge unreliable at detecting hallucination without concrete anchoring

**Finding:** Initial LLM-judge groundedness scores correlated with human judgment at only r=0.39 (vs. 0.76–0.86 on other criteria). After rewriting the rubric with concrete fabrication examples and a falsifiability test ("could this sentence get someone in trouble for being wrong?"), correlation improved to r=0.85.

**Hypothesis:** An abstract instruction ("avoid inventing facts") gives the judge model no concrete anchor for what counts as fabrication versus safe generic language — both can "sound" equally confident. Concrete positive/negative examples closed most of the gap, but this suggests groundedness judgments from an LLM judge should not be trusted without this kind of calibration, and ideally without ongoing human spot-checks.

---

## Cross-cutting pattern

Three of these five failures (#2, #4, #5) point to the same underlying issue: **the system is good at matching topic and tone, but has no reliable mechanism for distinguishing "safe to say" from "sounds right but isn't verified."** This is the single most important limitation to carry into the "what's misleading about my headline number" section of the report.
