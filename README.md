# star-schema-llm-context

Research and documentation of patterns for using star schema (Kimball-style) dimensional modeling as a data-driven flywheel for LLMs and agents (and whatever we call them next) - or, business rules and state persistence layers for LLM agent loops. This has been something that's been evolving in my head for 3+ years -- the idea keeps evolving but the core intuition hasn't changed: abstract the data, not the behavior.

In-context learning is the most powerful thing LLMs bring us - we have universal transformation engines / universal functions. What we need is a way to use our data to drive them in a more automated yet predictable way.

LLM-driven coding also changes the economics of data modeling itself. If your grain changes or business rules change or you have too much bloat and it's not working, refactor it. That's how you get faster feedback loops - LLMs will create bloat, that is how they work as of today. Your job is to define the outcomes and constraints that will help you figure out if what it built is what you wanted. If it isn't, keep iterating as your code base gets more bloat. Then once you have the outcome you want - reduce, refactor, whatever. Expand and compress, right? ...or middle out?

What would've been a rewrite or pure tech debt before is now a conversation with your agent. Schemas can be fuzzy when you need to adapt, constrained when you need predictability, and **signifgicantly** easier to change when the business rules shift. The hard part of software has always been adapting to changing business rules - but we should still be capturing them in a structured way so we can analyze, transform, and borrow patterns across processes. "This new process looks a lot like that one over there and also like this one, but the difference is xyz" - that kind of reasoning across business contexts is what structured data modeling enables.

LLM frameworks that abstract interaction patterns (chains, agents, retrievers) break when research moves faster than the abstraction, and AI labs are optimizing for the harness/model combo - you can't win there, you can only adapt your strategy toward where it's going. Dimensional modeling abstracts what happened (facts) in what context (dimensions) - patterns that have managed to hold up pretty well over time.

Here's where I've evolved on this: a useful data model for data. It's the business rules layer that drives what it does next. A source URL in `dim_source` is a monitoring rule. A skill description in `dim_skill` is a routing rule. SCD Type 2 versions these rules over time. Facts capture outcomes - what worked, what didn't, what humans labeled as helpful, what the agent tried and abandoned. The agent reads this data back to make routing decisions. Which skills to load, which sources to trust, which execution paths produce good outcomes - derived from labeled data rather than hardcoded logic.

Because dimensions are mutable (SCD Type 2), you can fix routing by updating the data, not the code. Change a skill description in `dim_skill`, update a source URL in `dim_source` - the agent picks up the change on next read without a redeploy. The schema is the stable layer; behavior changes by changing what's in it.

The part that gets me most excited: this creates a **data-driven** flywheel. The [AI engineering data flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/) describes the process of continuously improving data through evaluation, monitoring, and human/synthetic feedback. While by no means the only method to do so, dimensional modeling feels very much like a natural fit for making the agentic loops driven by data rather than conditional logic (which does not scale) - and it makes the flywheel output actionable - execution paths and routing decisions driven by the data the flywheel produces. You don't need custom workflow code for everything, nor do you want it, because by the time you write it, it will be either out of date or will not perform nearly as well as the simpler solution. 

You improve the agent by improving the data it reads. The data gets better as humans label and as it runs more often with more feedback.

When every consumer generates keys the same way, handles change detection the same way, and writes metadata the same way, the transformation patterns become consistent enough to build on top of. Consistent transformation of data-as-rules is what makes it possible to abstract upward without breaking what's below.

I took detours into interpretability and SAEs along the way. [Facebook's Large Concept Model](https://github.com/facebookresearch/large_concept_model) reinforced the idea of semantic abstractions within the inference process (not semantic embeddings, but semantic structure). [RLMs (Recursive Language Models)](https://arxiv.org/abs/2512.24601v1) remain an active research area for me -- primarily for their simplicity and potential as a verifier/routing model.

A star schema is almost certainly not the only way to do this - I don't pretend it is, and didn't during the data era either. There's never one ultimate solution, and there's inherent bias leaking through from my data background. But it's one that makes sense to me (and maybe others?): natural filtering, point-in-time data through SCD Type 2, explicit grain definition for decisions and outcomes, easy roll-ups at the lowest levels of granularity to aggregate patterns, etc. The bet is that the patterns transfer, not that the specific implementation is the only viable one.

If we can enable transformations and abstractions at a more consistent, structured level, then we can create a path for business users to leverage the "single interface into work" (that every AI lab is trying to compete for) into deeper levels of automation, enabling them to focus on the business logic and the data they want to drive a given outcome.

## what's in this repo
- Research ideas and directions that will likely evolve and change many times
- Things that seem like decent ideas but are not yet fully fleshed out or tested
- Links to other repos that are related in my universe, such as a [pattern of Skills](https://github.com/fblissjr/fb-claude-skills/blob/main/skill-maintainer/scripts/store.py) to see if it can work in various contexts.

This is a research and ideation library, not a code library.

## what's here

```
docs/
  library_design.md          Reference implementation spec
                             Core primitives, schema patterns, DAG hierarchy model,
                             consumer import patterns, design decisions with rationale
```

## quick primer for those unfamiliar with the core ideas and patterns of dimensional / star schema modeling in general

### key generation

MD5 hash surrogate keys from natural key components. Deterministic, NULL-safe, composable. No sequences, no coordination.

```python
import hashlib

def dimension_key(*natural_keys) -> str:
    parts = [str(k) if k is not None else "-1" for k in natural_keys]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

def hash_diff(**attributes) -> str:
    parts = [f"{k}={v}" for k, v in sorted(attributes.items()) if v is not None]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
```

### SCD Type 2 dimensions

Every dimension table: hash_key (no PK), business columns, effective_from/effective_to, is_current, hash_diff, record_source, session_id, created_at. Multiple rows per entity -- the whole point of SCD Type 2.

### fact tables

No PKs, no sequences. Grain = composite dimension keys + event timestamp. Append-only event logs. Metadata: inserted_at, record_source, session_id.

### meta tables

meta_schema_version (schema evolution tracking) and meta_load_log (operational visibility).

## current research areas and use cases

### primary: data-driven agent routing

The star schema as the business rules layer for agent loops. Dimensions encode routing rules (which skills, which sources, which approaches). Facts capture outcomes and labels. The agent queries this data to make routing decisions. Human and synthetic feedback improves the data, which improves the routing. Execution paths driven by data from the flywheel process.

### secondary: agent task decomposition DAG

Track agent execution as a data pipeline DAG. Goals decompose to tasks, tasks route to subagents, subagents execute tool chains, tool chains produce outputs that get synthesized. The five invariant operations (decompose, route, prune, synthesize, verify) become the fact table grain.

### tertiary: skill quality lifecycle

Track how skills evolve from creation to maturity. Trigger accuracy, validation trends, reference freshness, abstraction levels, complexity.

### future: cross-project pattern mining

Identify which conventions work across projects, surface candidates for promotion to shared skills.

See [docs/library_design.md](docs/library_design.md) for full schema designs and the expansion roadmap.

## related

- [fb-claude-skills](https://github.com/fblissjr/fb-claude-skills) -- Random collection (including one for this repo's concept) of Skills for Claude and others that have adopted the SKILLS.md paraidgm
- [ccutils](https://github.com/fblissjr/ccutils) -- client application that leverages session analytics
