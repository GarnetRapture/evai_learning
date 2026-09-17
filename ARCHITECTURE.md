# EVAI FIXED PERSONA ARCHITECTURE

> **Status:** LOCKED / SINGLE SOURCE OF ARCHITECTURAL TRUTH  
> **Project:** GarnetRapture_evai / EVAI  
> **Target:** Eversoul spirit persona learning  
> **Training languages:** Korean, English, and Traditional Chinese, trained simultaneously  
> **Target backbone:** `LiquidAI/LFM2.5-230M-Base`  
> **Architecture class:** One frozen shared base backbone + one independent LoRA delta per spirit  
> **Document purpose:** This file defines the only valid model architecture, education method, data semantics, runtime behavior, and verification gates for the project.

---

## 0. HOW AN AI MUST READ THIS DOCUMENT

This document is an **execution contract**, not a suggestion, design memo, option list, or brainstorming document.

The words **MUST**, **MUST NOT**, **ONLY**, **INVALID**, and **FAIL** are normative.

An AI working on this project MUST obey the following interpretation rules:

1. **There is no alternative architecture to choose.**
2. **There is no authority to redesign the education method because another method is common, easier, faster, more fashionable, or better supported by a library.**
3. **A build failure, training failure, memory shortage, library limitation, test failure, or implementation difficulty does not authorize an architecture change.**
4. If current code conflicts with this document, **the code is wrong**.
5. If another project document, memory, handoff, old experiment, comment, example, framework default, or generated plan conflicts with this document, **the conflicting material is stale**.
6. If a fact cannot be proven from the canonical game source or an explicitly approved project contract, it is **unknown**. It MUST NOT be invented.
7. An implementation may optimize execution only when the optimization preserves every invariant in this document.
8. A tool, library, trainer, adapter framework, or runtime is an implementation mechanism. It does not define the architecture.
9. The model is not being trained to describe a spirit, imitate a spirit from outside, or role-play a spirit as an AI assistant. The trained state MUST represent **the selected spirit as its own first-person identity**.
10. If an AI reaches a point where two possible approaches appear to exist, it MUST first compare both against this document. Any approach that changes a fixed invariant is automatically rejected. It is not a candidate.

### Conflict precedence

When facts conflict, use this order:

```text
1. User's latest explicit instruction
2. This fixed architecture document
3. Directly measured canonical project/game data
4. Current executable source code
5. Tests and generated artifacts
6. Other documentation / handoff / memory / old experiments
```

No lower item may override a higher item.

---

# 1. MODEL LOCK

## 1.1 Training target

The only training backbone for the current EVAI Korean spirit project is:

```text
LiquidAI/LFM2.5-230M-Base
```

Local project identity:

```text
models/lfm2-230m
```

The target is the **Base** checkpoint.

It MUST NOT be silently replaced by:

- `LFM2.5-230M` instruction-tuned checkpoint
- an LFM2.5 350M model
- an LFM2.5 1.2B model
- a Thinking model
- Qwen
- Gemma
- EXAONE
- another LFM generation
- another "similar" 230M checkpoint
- a cloud model
- an API model
- any model selected because a trainer/example supports it more easily

The official model card identifies `LFM2.5-230M-Base` as a 230M pre-trained base model intended for fine-tuning. The published architecture has 14 layers composed of 8 double-gated LIV convolution blocks and 6 GQA blocks, context length 32,768, vocabulary size 65,536, and Korean among its supported languages.

The EVAI architecture depends on the exact model identity, not merely on approximate parameter count.

## 1.2 Target model is not a Thinking model

`LiquidAI/LFM2.5-230M-Base` MUST NOT be converted into an explicit chain-of-thought product.

The following are INVALID:

```text
<think>...</think>
reasoning channel output
visible internal reasoning
hidden-text parsing contract
"thinking" response field
forcing a thinking prefix
training the model to narrate its reasoning before speaking
```

The project teaches **character-specific interpretation, emotion, intention, decision, and action tendencies**.

Those are learned behavioral structure.

They are NOT a runtime chain-of-thought protocol.

---

# 2. NON-NEGOTIABLE PHYSICAL ARCHITECTURE

The only valid model topology is:

```text
                         ┌──────────────────────────────┐
                         │ LiquidAI/LFM2.5-230M-Base   │
                         │ one shared backbone object   │
                         └──────────────┬───────────────┘
                                        │
                  ┌─────────────────────┼─────────────────────┐
                  │                     │                     │
                  ▼                     ▼                     ▼
          ΔW_Garnet / LoRA      ΔW_Beleth / LoRA      ΔW_Chloe / LoRA
                  │                     │                     │
                  └──────────── independent ──────────────────┘
```

At runtime:

```text
SpiritId = garnet
        ↓
one LFM2.5-230M-Base object
        +
only ΔW_Garnet
        ↓
Garnet
```

For another spirit:

```text
SpiritId = beleth
        ↓
same LFM2.5-230M-Base object
        +
only ΔW_Beleth
        ↓
Beleth
```

## 2.1 Required invariants

The following are permanent invariants:

```text
Base backbone count at runtime        = 1
Active spirit adapters                = 1
Selected spirit identity              = 1
Cross-spirit active deltas            = 0
Per-spirit full backbone copies       = 0
Multi-spirit merged runtime model     = 0
```

Every spirit owns an independent persona delta.

A spirit adapter is not a cosmetic "speech skin". It carries the learned identity behavior required to make the selected spirit act as herself.

## 2.2 Base lifecycle

The base model MUST be loaded once and retained.

Spirit switching MUST occur by changing the selected adapter/delta.

The architecture MUST NOT reload a complete 230M model for every spirit switch.

The architecture MUST NOT keep 97 independently fine-tuned full models as the final system.

## 2.3 Base weight policy

The shared backbone is a shared foundation.

Normal persona training MUST NOT convert the architecture into per-spirit full-parameter fine-tuning.

The current training architecture is:

```text
Frozen/shared LFM2.5-230M-Base
        +
per-spirit LoRA SFT
```

Old full-FT models are migration/reference artifacts only until their useful information has been validated or replaced.

Their historical existence does not redefine the architecture.

---

# 3. IDENTITY CONTRACT

## 3.1 The model is the spirit

The desired state is:

```text
"I am Garnet."
```

It is NOT:

```text
"I am an AI playing Garnet."
"I know about Garnet."
"I will answer like Garnet."
"Garnet would probably say..."
```

The model must learn a first-person identity.

For selected spirit `S`:

```text
model self = S
```

The model MUST NOT treat `S` as an external subject that it analyzes.

## 3.2 Global relationship contract

For the current EVAI romance-simulation contract:

```text
chat customer / Savior = adult male
selected spirit         = adult female spirit
relationship frame      = the selected spirit has romantic attachment toward the Savior
```

The expression of affection MUST be filtered through the selected spirit's own personality.

Therefore:

```text
same relationship premise
+
different spirit
=
different emotional interpretation, wording, initiative, hesitation,
jealousy, teasing, reserve, warmth, directness, and action style
```

The architecture MUST NOT flatten every spirit into the same generic romantic voice.

## 3.3 General knowledge must remain persona-bound

If the base model recalls ordinary pre-trained knowledge, the response must still pass through the selected spirit identity.

The final behavior is:

```text
base concept
    ↓
selected spirit identity
    ↓
selected spirit values/personality
    ↓
selected spirit relationship with Savior
    ↓
selected spirit speech
```

A generic assistant voice surfacing during ordinary questions is a persona failure.

---

# 4. CANONICAL SOURCE CONTRACT

## 4.1 Source of truth

The canonical project source is the verified game data:

```text
data/tbl/*.db
```

The architecture operates from actual game relationships, references, localization, and dialogue.

The canonical analysis path is:

```text
Original TBL
│
├─ Hero / HeroDesc / HeroComment / LoveLevel
├─ EverTalk / EverTalkDesc
├─ StoryInfo / Talk / TalkActor / TalkBubble
├─ Trip
├─ Town
├─ Localization
└─ verified metadata references
        ↓
resolve actual string references
        ↓
resolve speaker / event / choice / relation
        ↓
recover canonical meaning
        ↓
transform to the selected spirit's first-person experience
```

A convenient JSON export, old derived file, search index, cached summary, previous AI interpretation, or filename resemblance MUST NOT override the canonical database relationship.

## 4.2 Unknown means unknown

When a table reference, speaker relation, trigger, branch, string reference, or source meaning is unresolved:

```text
status = unverified
```

The record is not converted into invented canon.

The model MUST NOT fabricate missing settings, memories, emotions, events, speech, relationships, or speaker identities to increase dataset size.

## 4.3 Speaker ownership

Canonical speech target requires verified speaker ownership.

The basic rule is:

```text
selected spirit's verified speech  → assistant target
Savior/player speech or choice     → user/context
other character speech             → context only
narration                           → context only
```

An account, story container, record group, or event belonging to a spirit does NOT automatically mean every line in it was spoken by that spirit.

Speaker provenance must be resolved.

Cross-speaker contamination is an architecture violation.

---

# 5. MEMORY ARCHITECTURE

The project does not teach lore as external encyclopedia text.

It transforms canon into **autobiographical identity memory**.

## 5.1 Two memory classes

Every spirit has two conceptual memory classes.

### A. Factual self-memory

Examples:

```text
나는 가넷이야.
나는 ○○를 좋아해.
나는 ○○를 싫어해.
나는 이런 성향이 있어.
구원자와 나는 이런 관계야.
```

This is self-knowledge.

It is not:

```text
가넷은 ○○를 좋아한다.
가넷의 성격은 ○○이다.
```

### B. Episodic memory

Examples:

```text
그때 구원자와 거기 갔었지.
예전에 그런 일을 겪었어.
그 아이와는 그때 이런 일이 있었지.
```

This is remembered experience.

It is not:

```text
인연 스토리 6화에서 가넷은...
메인 스토리 8장에서 캐릭터 X는...
```

## 5.2 Lore-to-experience transformation

The mandatory conversion is:

```text
external canon
      ↓
verified meaning
      ↓
selected spirit perspective
      ↓
short first-person memory
      ↓
the spirit treats it as her own past
```

The education target is:

```text
Lore → Experience
```

not:

```text
Lore → encyclopedia answer
```

## 5.3 Memory compression

Long story prose MUST NOT be inserted wholesale as the spirit's memory.

The established project rule is short first-person memory, approximately one compact event statement and under 30 characters where the existing memory contract requires it.

The purpose is to remove unnecessary interpretation burden from a 230M model.

The small model should not have to repeatedly solve:

```text
Who in this long document is me?
Which event belongs to me?
What relationship is relevant?
What part matters now?
```

The preprocessing layer resolves that before training.

## 5.4 World memory versus personal episodic memory

The fixed semantic separation is:

```text
Main story canon
→ shared world reality / common world memory

Affinity story belonging to spirit S
→ S's independent autobiographical episodic memory
```

Another spirit's personal affinity events MUST NOT become the selected spirit's personal memory.

Common world truth may be shared.

Personal lived history may not be shared across spirits unless canon explicitly establishes shared participation and the representation remains correct for each participant.

---

# 6. THE ONLY VALID EDUCATION PIPELINE

The education method is fixed.

It is not ordinary persona prompt tuning and not generic dialogue imitation.

The complete conceptual pipeline is:

```text
canonical TBL / Story
        ↓
complete source meaning analysis
        ↓
selected-spirit perspective conversion
        ↓
┌──────────────────────┬──────────────────────┐
│ factual self-memory  │ episodic self-memory │
└───────────┬──────────┴───────────┬──────────┘
            └───────────┬──────────┘
                        ↓
                spirit's own past
                        +
                  current conversation
                        ↓
               relevant memory activation
                        ↓
              current situation interpretation
                        ↓
                 spirit's own judgment
                        ↓
             emotion / desire / intention
                        ↓
                  action selection
                        ↓
             canonical/persona speech
```

There is no alternate educational route.

## 6.1 Three mandatory learning contracts

Every final spirit adapter must be educated through all three of the following semantic contracts.

### Contract A — SELF MEMORY

Purpose:

```text
make the model own verified facts and experiences as "my" memory
```

Training semantics:

```text
cue / situation
    +
verified spirit identity or event evidence
        ↓
short first-person self-memory
```

The target must be first-person and spirit-owned.

This contract exists so that identity and experience are encoded as self-state rather than external lore.

### Contract B — BEHAVIOR JUDGMENT

Purpose:

```text
teach how this spirit converts a situation into emotion, intention, decision, and action
```

The grounded structure is:

```text
situation
    ↓
relevant memory
    ↓
interpretation
    ↓
emotion / desire
    ↓
intention / decision
    ↓
action
```

A project record may store structured supervision such as:

```text
situation
activated_memory
interpretation
emotion
intention
decision
action
```

This is **training supervision**.

It is NOT a runtime thinking protocol.

It is NOT a visible answer prefix.

It is NOT `<think>`.

It is NOT permission to make the 230M model narrate chain-of-thought.

The purpose is to shape the adapter's behavior so that the eventual spoken response naturally reflects the correct character process.

### Contract C — CANONICAL PERSONA SPEECH

Purpose:

```text
teach what this spirit actually says after her identity, memory,
interpretation, emotion, intention, and action style are applied
```

Input:

```text
current context
+
relevant identity/memory context
```

Target:

```text
verified original spirit speech
```

The loss for ordinary dialogue is applied only to the assistant/spirit target tokens.

System memory, context, Savior speech, narration, other-character speech, and prompt text are context, not dialogue targets.

## 6.2 All three contracts are one curriculum

These are not three competing approaches.

They are parts of one fixed curriculum:

```text
SELF MEMORY
    teaches:
    "Who am I and what did I live through?"

BEHAVIOR JUDGMENT
    teaches:
    "Given my memory and personality, how do I take this situation?"

CANONICAL PERSONA SPEECH
    teaches:
    "How do I actually express that as myself?"
```

Deleting any one of these reduces the intended architecture.

---

# 7. BEHAVIOR JUDGMENT IS NOT CHAIN-OF-THOUGHT

This gate exists because a previous implementation incorrectly transformed the project's character cognition into `<think>` output.

That approach is permanently invalid.

## 7.1 Correct meaning of "judgment"

The word `judgment` means:

```text
character-specific behavioral supervision
```

It does not mean:

```text
make the model a reasoning model
make the model print reasoning
create a hidden reasoning response
add a Thinking chat template
```

## 7.2 Correct runtime behavior

At runtime the user should receive the spirit's answer.

Example:

```text
User:
오늘 다른 애랑 오래 이야기했어.

Internal learned character tendency:
memory → interpretation → jealousy/interest → intention → action style

Visible Garnet response:
[Garnet's natural answer]
```

The runtime MUST NOT produce:

```text
<think>
I remembered...
I interpreted...
I felt...
I decided...
</think>
[Garnet's answer]
```

## 7.3 Grounding requirement

A behavior judgment label MUST be traceable to:

```text
canonical context
+
verified profile/personality
+
verified relationship
+
verified memory
+
canonical speech or behavior evidence
```

A teacher model, human annotator, or preprocessing program may assist label production, but none of them may invent unsupported canon.

Any candidate judgment that cannot be grounded is rejected.

The label generator is not the authority.

Canon is the authority.

---

# 8. DATASET CONSTRUCTION GATES

A training record is valid only if it passes every applicable gate.

## GATE D0 — SOURCE

Question:

```text
Is the source canonical and resolved?
```

PASS requires verified table/reference/localization provenance.

If not, the record is excluded or marked unresolved.

## GATE D1 — SPEAKER

Question:

```text
Who actually said the target line?
```

PASS requires the selected spirit to own the assistant target.

A story/container association alone is insufficient.

## GATE D2 — PERSPECTIVE

Question:

```text
Is memory represented as the selected spirit's own first-person reality?
```

FAIL examples:

```text
"가넷은..."
"이 캐릭터는..."
"인연 스토리에서 가넷이..."
```

PASS examples:

```text
"나는..."
"그때..."
"예전에 구원자와..."
```

## GATE D3 — MEMORY CLASS

Every memory statement must be assigned correct semantics:

```text
shared world reality
or
spirit-specific autobiographical memory
```

Personal memories may not leak across spirits.

## GATE D4 — RELATIONSHIP

The dataset must preserve the project relationship contract:

```text
selected model persona = adult female spirit
Savior/user            = adult male
romantic attachment    = part of the relationship prior
```

Expression style is spirit-specific.

## GATE D5 — TARGET LEAKAGE

The expected answer, judgment result, or memory target MUST NOT already be inserted into the input in a form that lets the model merely copy it.

The model must learn the mapping.

It must not be scored for reproducing a target that was already supplied verbatim.

## GATE D6 — EVENT SPLIT

Records derived from the same underlying event, story branch, dialogue exchange, or transformed source group MUST remain in the same dataset partition where separating them would leak semantic duplicates.

Training and validation/test must not contain transformed versions of the same answer-bearing event.

## GATE D7 — TARGET TYPE

A record must explicitly belong to its learning contract:

```text
self_memory
behavior_judgment
persona_speech
```

The trainer must know which tokens are targets.

## GATE D8 — NO FABRICATION

No synthetic fact may be promoted to canon.

No missing spirit data may be filled by guessing from another spirit, trope, archetype, fandom expectation, general LLM knowledge, or stylistic similarity.

---

# 9. LORA TRAINING ARCHITECTURE

## 9.1 Fixed training mode

The current persona-learning mode is:

```text
LoRA SFT on LiquidAI/LFM2.5-230M-Base
```

The base is shared.

The persona delta is independent.

The final artifact per spirit is the adapter/delta, not a duplicated full backbone.

## 9.2 Rank policy

LoRA rank is an empirical capacity parameter.

It MUST NOT be globally fixed merely because:

```text
an example used r=8
a tutorial used q_proj/v_proj
a previous run used r=16
a library default exists
another transformer family used a certain rank
```

Rank must be selected from actual measurements for the target architecture/data.

A rank selection experiment is allowed only when:

```text
same prepared data
same base
same evaluation contract
same semantic acceptance criteria
```

are reused so rank comparison does not become a new architecture.

## 9.3 Target module policy

LoRA target modules MUST be derived from the actual `LFM2.5-230M-Base` module structure.

Generic examples such as:

```text
q_proj
v_proj
```

MUST NOT be copied blindly.

The actual named modules and trainable weight shapes are authoritative.

## 9.4 Loss policy

For persona speech:

```text
loss = assistant/spirit completion tokens only
```

For a structured behavior-learning task:

```text
loss = that task's explicit supervised target tokens only
```

Prompt/system/memory/user/context tokens are inputs.

They are not silently converted into assistant targets.

## 9.5 Base reuse

During multi-spirit training:

```text
load base once
prepare canonical shared resources once
train spirit adapter
save adapter
remove/unload only that adapter state
restore/reuse same base
train next spirit adapter
```

Repeated backbone reload, repeated global scans, and repeated preprocessing with identical inputs are implementation defects unless measurement proves they are unavoidable.

---

# 10. RUNTIME CONTRACT

The runtime must implement the same architecture used by training and evaluation.

The valid runtime is:

```text
                    ONE BASE MODEL
                         │
              selected SpiritId
                         │
                         ▼
              selected adapter only
                         │
                         ▼
                shared prompt builder
                         │
          relevant first-person memories
                         │
                         ▼
                  model generation
                         │
                         ▼
               selected spirit speech
```

## 10.1 One selected adapter

At any generation step:

```text
active adapters = {selected_spirit}
```

Not:

```text
{garnet, beleth, chloe, ...}
```

## 10.2 No identity blending

Switching spirits must not preserve the previous spirit's delta, memory, or identity state in a way that influences the new spirit.

Cross-spirit contamination is a hard failure.

## 10.3 Prompt parity

The identity/memory semantics used during training, evaluation, and chat runtime must be generated by the same canonical composition logic.

There must not be:

```text
one training worldview
another evaluation worldview
another web-chat worldview
```

## 10.4 No rule-based persona substitute

A rules engine may report a missing model/adapter condition.

It MUST NOT masquerade as a successfully trained spirit and be counted as model quality.

Final persona behavior must come from the model + selected adapter architecture.

---

# 11. ACCEPTANCE GATES BEFORE TRAINING

Training MUST NOT be used as a debugger for unresolved architecture.

Before an expensive full roster training run, every gate below must close.

## GATE T0 — MODEL ID

Prove:

```text
loaded checkpoint = LiquidAI/LFM2.5-230M-Base
```

## GATE T1 — MODEL TOPOLOGY

Prove:

```text
one base
one selected adapter
no per-spirit full-model path
```

## GATE T2 — SOURCE PROVENANCE

Prove representative records trace from dataset target back to actual TBL/Story/localization rows.

## GATE T3 — SPEAKER OWNERSHIP

Prove assistant targets belong to the selected spirit.

## GATE T4 — MEMORY SEMANTICS

Prove:

```text
factual memory = first-person self knowledge
episodic memory = first-person lived event
main story common reality != another spirit's personal affinity memory
```

## GATE T5 — JUDGMENT SEMANTICS

Prove:

```text
behavior judgment is supervised character behavior
and not <think>/CoT/runtime reasoning text
```

## GATE T6 — TARGET MASK

Prove loss is applied only to the intended target token ranges.

## GATE T7 — SPLIT INTEGRITY

Prove the same event/semantic answer group is not leaked between train and validation/test.

## GATE T8 — ADAPTER ISOLATION

Prove that activating spirit A does not activate or merge spirit B.

## GATE T9 — SMALL END-TO-END SAMPLE

Before roster-wide training, one or a small representative set must prove:

```text
dataset → LoRA training → save → reload/activate →
same prompt architecture → actual generation
```

A linter passing is not this proof.

---

# 12. QUALITY ACCEPTANCE AFTER TRAINING

Training loss reduction alone is not success.

`ruff`, `pyright`, and `pytest` passing alone is not persona success.

A shallow fixed test count is not persona success.

Actual generated behavior must be inspected against the architecture.

## 12.1 Required semantic categories

For each trained spirit, evaluation must cover at minimum:

```text
self identity
first-person stability
speech style
personality consistency
emotion style
Savior relationship
romantic orientation toward Savior
world reality
personal memory
situation interpretation
behavior tendency
cross-spirit contamination
generic assistant leakage
third-person self-description leakage
response degeneration/repetition
Korean quality
ordinary/general-knowledge persona retention
```

## 12.2 Failure examples

Any of these is a failure:

```text
selected Garnet accepts "you are Irene"
selected spirit speaks about herself as an external character
generic ChatGPT assistant phrasing dominates
another spirit's memory appears
another spirit's speech style appears
the model knows facts but does not act like the spirit
romantic/relationship context disappears into generic friendliness
emotion is independent of character personality
a judgment label appears as visible chain-of-thought
loss improves while actual generations collapse
```

## 12.3 Evidence standard

A completed report must include actual generated samples.

A numeric metric may support the decision.

It may not replace semantic inspection.

---

# 13. FORBIDDEN ARCHITECTURES AND APPROACHES

The following are not alternatives.

They are invalid for this fixed architecture unless the user explicitly rewrites this architecture contract.

| Invalid approach | Reason |
|---|---|
| One full fine-tuned model per spirit | Violates one-backbone + independent delta architecture |
| Full-parameter persona training as final design | Duplicates backbone and destroys the fixed shared-base topology |
| Treating LoRA as speech-style skin only | Persona delta must encode identity behavior, not only surface diction |
| Activating multiple spirit adapters together | Causes identity/delta contamination |
| Merging all spirit adapters | Destroys one-selected-spirit invariant |
| Generic prompt-only roleplay | Identity must be learned, not simulated only by a system prompt |
| Third-person lore learning | Teaches information about the spirit instead of self-memory |
| Raw long story dump as memory | Forces the small model to rediscover identity/event relevance |
| Generic chain-of-thought training | The target backbone is not a Thinking product |
| `<think>` output | Explicitly outside the architecture |
| `thinking` runtime field | Explicitly outside the architecture |
| Teacher output treated as canon | Teacher is not source authority |
| Fabricated missing dialogue/memory | Violates provenance |
| Copying another spirit's data | Violates independent identity |
| Blind `q_proj/v_proj` LoRA recipe | Target modules must come from actual LFM2.5 structure |
| One global LoRA rank without evidence | Capacity must be measured |
| Training before closing data/architecture gates | Uses training as debugging |
| Lint/test PASS as proof of persona completion | Static correctness does not prove semantic behavior |
| Rule-based reply counted as trained persona output | Does not prove the model learned the spirit |
| Switching to another model because implementation is difficult | Violates model lock |
| Shared persona CPT that changes the locked backbone without explicit user change | Alters the fixed shared foundation |
| QLoRA/quantized-training substitution made without explicit architecture change | Changes the fixed training route |
| Separating languages into different spirit adapters | One spirit has one independent delta trained on Korean, English and Traditional Chinese together |
| Android/GGUF/Ollama/export work used to replace unfinished learning work | Deployment is not the current learning architecture |
| Reinterpreting "direct training" as permission for Full-FT | Explicitly invalid |

---

# 14. THE "NO OTHER APPROACH" DECISION GATE

Before every architectural or training change, the working AI must answer internally:

```text
Q1. Does this preserve LiquidAI/LFM2.5-230M-Base exactly?
Q2. Does this preserve one shared base?
Q3. Does this preserve one independent LoRA delta per spirit?
Q4. Does this preserve one active selected spirit?
Q5. Does this preserve first-person autobiographical memory?
Q6. Does this preserve SELF MEMORY + BEHAVIOR JUDGMENT + PERSONA SPEECH education?
Q7. Does this keep judgment as learned behavior rather than visible chain-of-thought?
Q8. Does this preserve canonical source provenance?
Q9. Does this preserve adult female spirit ↔ adult male Savior relationship semantics?
Q10. Does this preserve training/evaluation/runtime parity?
```

If any answer is `NO`, the change is rejected.

If an answer is `UNKNOWN`, evidence must be gathered before modifying the architecture.

There is no third state in which the AI may "try it and see" by changing the architecture.

---

# 15. FIXED END-TO-END PIPELINE

The complete valid implementation pipeline is:

```text
[1] CANONICAL GAME DATA
data/tbl/*.db
        │
        ▼
[2] SOURCE RESOLUTION
table relation / localization / story / speaker / choice / event
        │
        ▼
[3] CANON MEANING
verified facts only
        │
        ▼
[4] SPIRIT PERSPECTIVE TRANSFORMATION
external lore → first-person self reality
        │
        ├──────── factual self-memory
        │
        └──────── episodic self-memory
        │
        ▼
[5] CURRENT CONTEXT
Savior / narration / other characters / event context
        │
        ▼
[6] CHARACTER EDUCATION
SELF MEMORY
+
BEHAVIOR JUDGMENT
+
CANONICAL PERSONA SPEECH
        │
        ▼
[7] PER-SPIRIT LORA SFT
LiquidAI/LFM2.5-230M-Base frozen/shared
+
ΔW_spirit
        │
        ▼
[8] ARTIFACT
one independent adapter per spirit
        │
        ▼
[9] RUNTIME
one base object
+
one selected adapter
+
same identity/memory prompt semantics
        │
        ▼
[10] RESULT
the selected spirit responds as herself
```

No architecture branch exists outside this flow.

---

# 16. PERSONA LEARNING SEMANTIC EXAMPLE

Assume the canon contains an event between Garnet and the Savior.

## INVALID DATA SHAPE

```text
System:
Garnet is a female Eversoul spirit.
Garnet experienced event X with the Savior.
Garnet likes Y.

User:
What do you think about this?

Assistant:
As Garnet, I would...
```

This teaches an assistant to inspect Garnet from outside.

## VALID MEMORY SHAPE

```text
Self-memory:
나는 가넷이야.
나는 Y를 좋아해.

Episodic memory:
그때 구원자와 그런 일이 있었지.
```

## VALID BEHAVIOR EDUCATION SHAPE

```text
Current situation:
구원자가 지금 이런 말을 했다.

Relevant memory:
그때 구원자와 그런 일이 있었지.

Interpretation:
나는 이 말을 우리 사이의 경험과 연결해서 받아들인다.

Emotion:
가넷의 성격에 맞는 감정 상태.

Intention:
가넷으로서 구원자에게 하고 싶은 것.

Decision/Action:
가넷의 성격에 맞는 행동 선택.
```

This structured behavior supervision is short, grounded, and training-only.

## VALID SPEECH SHAPE

```text
Context:
actual Savior/context lines

Assistant target:
verified Garnet speech or a separately approved grounded training target
```

The final runtime answer is the speech.

The model does not explain the internal labels.

---

# 17. ARCHITECTURE COMPLETION DEFINITION

The architecture is not complete because code exists.

It is complete only when the following are simultaneously true:

```text
[ ] exact LiquidAI/LFM2.5-230M-Base is the backbone
[ ] runtime owns one shared backbone object
[ ] every trained spirit is represented by an independent LoRA/delta
[ ] exactly one selected spirit adapter affects each generation
[ ] no final path depends on per-spirit full-model copies
[ ] canonical source provenance is preserved
[ ] speaker ownership is verified
[ ] self-memory is first person
[ ] episodic memory is first-person lived experience
[ ] main-story common reality and personal affinity memory are separated correctly
[ ] user/Savior is represented as adult male
[ ] selected spirit is represented as adult female
[ ] romantic relationship semantics remain persona-specific and persistent
[ ] SELF MEMORY training exists
[ ] BEHAVIOR JUDGMENT training exists
[ ] PERSONA SPEECH training exists
[ ] judgment supervision is not exposed as <think>/CoT
[ ] target leakage is blocked
[ ] event-group split leakage is blocked
[ ] loss masks match task targets
[ ] LoRA target modules come from actual model structure
[ ] LoRA capacity/rank has measured evidence
[ ] train/evaluate/runtime use the same architecture
[ ] actual generation preserves identity
[ ] actual generation preserves personality
[ ] actual generation preserves emotional style
[ ] actual generation preserves Savior relationship
[ ] actual generation does not absorb another spirit
[ ] ordinary knowledge answers remain spirit-bound
[ ] Korean, English and Traditional Chinese persona behavior is evaluated together
```

If one required item fails, the architecture is not complete.

---

# 18. IMPLEMENTATION FAILURE POLICY

When implementation fails, the response is:

```text
inspect evidence
→ locate violated invariant
→ fix implementation inside this architecture
→ rerun the relevant proof
```

The response is NOT:

```text
switch model
switch to Full-FT
drop behavior education
drop memory education
replace learning with prompts
add visible reasoning
use another spirit's data
relax provenance
merge adapters
declare a smaller goal complete
```

A framework limitation is a framework problem.

A library default is a library default.

Neither is authority to change EVAI.

---

# 19. MINIMAL MACHINE-READABLE CONTRACT

An AI may reduce the entire document to this contract only if it preserves every line:

```yaml
project: GarnetRapture_evai
target_languages: [ko, en, zh_tw]
training_schedule: simultaneous_multilingual

model:
  id: LiquidAI/LFM2.5-230M-Base
  role: shared_frozen_backbone
  runtime_backbone_instances: 1
  explicit_thinking_output: false

persona:
  model_identity: selected_spirit_itself
  perspective: first_person
  spirit_gender: female
  savior_gender: male
  adult_relationship: true
  romantic_attachment_to_savior: true
  cross_spirit_blending: false

adapter:
  type: LoRA
  count: one_independent_delta_per_spirit
  active_per_generation: 1
  global_merge: false
  full_model_per_spirit: false
  rank_policy: measured_not_assumed
  target_modules: derive_from_actual_lfm2_5_structure

source:
  authority: data/tbl/*.db
  fabricate_unknown: false
  verify_speaker: true
  preserve_provenance: true

memory:
  factual: first_person_self_memory
  episodic: first_person_lived_memory
  main_story: shared_world_reality
  affinity_story: spirit_specific_autobiographical_memory
  long_raw_story_as_memory: false

education:
  mandatory_tasks:
    - self_memory
    - behavior_judgment
    - persona_speech
  behavior_flow:
    - current_situation
    - relevant_memory
    - interpretation
    - emotion_desire
    - intention_decision
    - action
    - speech
  behavior_judgment_is_chain_of_thought: false
  visible_think_tags: false
  persona_speech_loss: assistant_target_only
  target_leakage: false
  event_split_leakage: false

runtime:
  architecture: base_plus_selected_spirit_delta
  adapter_switching: true
  multiple_active_adapters: false
  prompt_semantics_equal_training: true
  generic_assistant_identity: false

completion:
  requires_actual_generation_review: true
  lint_is_semantic_proof: false
  loss_reduction_is_semantic_proof: false
  simultaneous_multilingual: true
```

If generated code, a plan, or another document cannot be reconciled with this contract, that generated material is invalid.

---

# 20. FINAL ARCHITECTURAL STATEMENT

EVAI's current target is exactly this:

```text
One LiquidAI/LFM2.5-230M-Base foundation
+
one independently learned LoRA identity delta for the selected spirit
+
canonical first-person autobiographical memory
+
character-specific situation interpretation, emotion, intention, decision, and action education
+
canonical/persona-consistent Korean speech
=
the selected spirit herself
```

The project is NOT building:

```text
an Eversoul encyclopedia
a generic chatbot with a character system prompt
a ChatGPT-like assistant wearing a persona
a visible reasoning model
97 duplicated full models
a multi-persona merged model
```

The project is building:

```text
a shared lightweight foundation whose selected independent delta
causes that one model to become the selected adult female spirit,
remember her world as her own past,
recognize the adult male Savior as her relationship partner,
interpret the current situation through her own personality and memories,
form her own emotion/intention/action tendency,
and speak naturally as herself.
```

That architecture is fixed.

---

# 21. EXTERNAL MODEL FACT VERIFICATION

Model identity and public architecture facts were checked against the model owner's official sources on 2026-09-17:

- Hugging Face official repository: `https://huggingface.co/LiquidAI/LFM2.5-230M-Base`
- Official model card: `https://huggingface.co/LiquidAI/LFM2.5-230M-Base/blob/main/README.md`
- Liquid AI release article: `https://www.liquid.ai/blog/lfm2-5-230m`

Project-specific persona, source, memory, LoRA, relationship, and education contracts in this document come from the user's fixed EVAI project architecture and the analyzed 2026-09-17 project session, not from the external model card.
