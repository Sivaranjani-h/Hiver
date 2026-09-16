# SpotifyCares AI Support Agent — Report

## 1. Problem Framing

**Brand chosen:** SpotifyCares (43,265 historical support replies — large enough for real
patterns, small enough to iterate on within the project timeframe).

**What "good" means for this brand:** SpotifyCares handles high-volume, largely repetitive
support traffic (playback bugs, billing questions, account access, content availability,
feature requests). A "good" system for this brand is one that:
- Correctly routes the *majority* of low-stakes, repetitive traffic (technical
  troubleshooting, generic feedback, content questions) without human involvement,
- Never auto-sends a reply that makes an unverified factual or financial claim,
- Escalates anything touching money or account access to a human by default, regardless
  of how confident the system is,
- Is honest about *which* traffic it can safely handle versus which it can't yet.

**What we chose not to build:**
- **Full multi-turn conversation handling.** Early data exploration showed a large share of
  "customer messages" in the raw dataset are actually mid-thread fragments (device info
  replies, "same issue as here," follow-up confirmations), not standalone complaints.
  Classifying these as independent intents is not a meaningful problem
  ("what is the intent of 'Windows 10, Chrome'?"). We scoped to the initial/support-seeking
  message only — a deliberate, documented scope cut (see Decision Log, #2).
- **A trained/fine-tuned classifier.** Given the timeframe, we used few-shot LLM
  classification instead of training a custom model from scratch.
- **Fully manual hand-labeling of all 200 golden examples.** Labeling was AI-assisted with
  human review of the process and spot-checks, not independently hand-labeled end to end —
  disclosed explicitly rather than presented as fully manual (Decision Log, #4).
- **A single end-to-end confidence score.** We deliberately kept classification confidence,
  retrieval confidence, and risk category as separate, inspectable signals rather than
  collapsing them into one opaque score, so the escalation decision stays explainable.

## 2. System Architecture

1. **Intent classification** — few-shot LLM classification (Groq, `gpt-oss-20b`) across
   11 intents defined from direct inspection of the SpotifyCares data.
2. **Reply drafting** — retrieval-augmented generation: embed the incoming message
   (`all-MiniLM-L6-v2`), retrieve the most similar historical customer↔reply pairs, and
   prompt an LLM to draft a reply grounded in those real examples.
3. **Auto-handle vs. escalate** — a message is only auto-handled if BOTH (a) its predicted
   intent is not in a fixed risk list (Billing & Payment, Account & Login, Premium &
   Subscription) AND (b) retrieval similarity to the best historical match exceeds 0.75.
   Otherwise it is escalated with a stated, specific reason.

## 3. Results vs. Baselines

### Classification

| Approach | Accuracy |
|---|---|
| Trivial (always predict majority class "Noise / Unclassifiable") | 33.73% |
| Simple (keyword/regex rules) | 47.93% |
| **Our system (few-shot LLM)** | **77.51%** (131/169) |

Per-intent precision/recall/F1 shows strong performance on Billing & Payment, Account &
Login, and Content Availability (F1 > 0.88), and weaker performance on Premium &
Subscription (F1 = 0.50, driven by low recall — frequently confused with Billing & Payment)
and Feature Request / Feedback (F1 = 0.61, driven by low precision — over-applied to vague
messages).

### Reply Quality (semantic similarity to the real historical reply that was actually sent)

| Approach | Similarity |
|---|---|
| Trivial (identical canned reply for every message) | 0.406 |
| Simple (retrieval only, reply copied verbatim, no LLM rewriting) | 0.505 |
| **Our system (retrieval + LLM grounding)** | **0.520** |

The gap between simple and our system is smaller than expected on this metric — but the
metric itself doesn't penalize the simple baseline for copying a *different* customer's
specific details verbatim (wrong handle, wrong device, wrong specifics), which is a real
correctness risk the LLM step corrects for but that semantic similarity doesn't capture
well. See Section 5.

### Escalation Behavior

Of 169 test messages: **86 auto-handled (50.9%), 83 escalated (49.1%)** — 32 escalated for
risk category (Billing/Account/Premium), 51 for low retrieval confidence.

## 4. Evaluation Harness

- **Automated metrics:** classification accuracy/precision/recall/F1; semantic similarity
  of drafted replies to real historical replies.
- **LLM-as-judge:** 4-criterion rubric (Relevance, Tone Fit, Groundedness, Actionability,
  1–5 each) scored on a 20-message sample via Groq.
- **Human agreement:** the same 20 messages were independently scored by a human against
  the identical rubric. Initial agreement was strong on Relevance (r=0.86), Tone Fit
  (r=0.76), and Actionability (r=0.81), but weak on Groundedness (r=0.39) — the LLM judge
  was unreliable specifically at detecting fabricated claims. After sharpening the
  groundedness rubric with concrete fabrication examples and a falsifiability test
  ("could this exact sentence get someone in trouble for being wrong?"), agreement rose to
  r=0.85, with a small tradeoff (Relevance and Tone Fit agreement dropped slightly, to
  r=0.75 and r=0.68 respectively).

## 5. Failure Analysis (Top 5)

**1. Premium ↔ Billing intent confusion (recall problem).**
*"y'alls system for management is crap I got charged for our family plan..."* — true
intent Premium & Subscription, predicted Billing & Payment. Premium/subscription issues
frequently surface through a billing symptom, so surface vocabulary overlaps heavily;
the classifier appears to key on payment keywords without distinguishing "the charge is
wrong" from "I paid correctly but the subscription doesn't work."

**2. Operational-status hallucination that bypasses the safety net.**
*"Spotify is DOWN omgggg, what am I gonna do"* → drafted reply invented *"Our tech team is
on it right now."* This message scored 0.82 retrieval similarity and was marked
AUTO_HANDLE — the escalation rule caught category risk and low grounding, but not
content-level fabrication. High retrieval similarity measures topical match, not factual
safety.

**3. Data leakage between the knowledge base and evaluation set.**
34 of 169 test messages (20.1%) also existed verbatim in the initial knowledge base sample,
since both were drawn independently from the same 41,697-message pool. This produced
misleadingly perfect (1.00) retrieval scores for those cases. Found and fixed before final
results by excluding test-set messages from the knowledge base.

**4. Generic triage instead of direct answers to specific factual questions.**
*"where is the perfect velvet"* and *"what's the presale code for [artist]"* both received
generic non-answers ("we'll have it available as soon as it's available," a fabricated
distribution claim) instead of direct engagement with the specific question asked. The
knowledge base is dominated by generic troubleshooting/DM-redirect replies, biasing
retrieval and drafting toward that pattern even when it doesn't fit.

**5. LLM-as-judge unreliable at detecting hallucination without concrete anchoring.**
See Section 4 — groundedness agreement with human judgment was weak (r=0.39) until the
rubric was given concrete positive/negative examples, after which it improved to r=0.85.
This suggests groundedness judgments from an LLM judge should not be trusted without this
kind of calibration.

**Cross-cutting pattern:** three of five failures (#2, #4, #5) share a root cause — the
system reliably matches *topic and tone*, but has no dependable mechanism for
distinguishing "safe, generic, and true" from "confident-sounding but unverified."

## 6. What Is Misleading About My Headline Number?

The 77.51% classification accuracy and the 50.9% auto-handle rate both sound like clean,
positive results in isolation — but combined, they overstate what the system reliably
delivers end-to-end:

- The *true* throughput number is not 77.51% (classification) or even 50.9%
  (auto-handle rate) in isolation — it's their **intersection**: only messages that are
  BOTH correctly classified AND correctly routed AND safely drafted actually reach a
  customer without human review. We did not compute this compounded number directly, but
  it is necessarily lower than either headline figure alone.
- Within the 86 messages marked "safe to auto-handle," at least one (the outage message)
  still contained a fabricated operational claim. The escalation rule reduces risk; it does
  not eliminate it.
- The reply-quality similarity score (0.520) sits close to a naive retrieval-only baseline
  (0.505), which could misleadingly suggest the LLM grounding step adds little value — when
  in fact the metric doesn't capture the correctness risk of copying another customer's
  specific details verbatim, which the LLM step measurably avoids.
- The golden evaluation set itself was AI-assisted, not fully independently hand-labeled,
  which the assignment's accuracy numbers inherit as an upstream assumption.

## 7. What We'd Do With One More Week

1. **Add a dedicated hallucination/fact-check pass** on drafted replies before the
   auto-handle decision — a second, narrowly-scoped LLM call whose only job is "does this
   reply state anything not present in the retrieved examples," rather than relying on the
   retrieval-similarity threshold alone to proxy for factual safety.
2. **Expand and rebalance the golden evaluation set**, especially for intents with very
   few examples (Playlist & Library had only 3 total, leaving zero left for testing after
   few-shot examples were drawn).
3. **Have a second independent human fully re-label the golden set from scratch**
   (rather than reviewing AI-drafted labels) to get a true inter-annotator agreement
   figure, strengthening the "hand-labelled" claim.
4. **Build a real multi-turn thread handler**, now that we've explicitly scoped it out,
   using the fragment/continuation detection this project surfaced as a starting point.
5. **Run the LLM-judge groundedness check against a larger human-scored sample** (50+
   rather than 20) to get a more statistically robust agreement estimate.
