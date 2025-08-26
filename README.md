# Modeling Context and Memory as a Star Schema

## Overview

This is a pure prototype of seeing what happens when you take a star schema model + an OLAP database (or any relational db) and create business-level abstractions in the form of dimensions and facts.

## Background

This project came about from three places:

1. My anchoring on 20+ years in the data space, where star schema models reign king
2. 3 years of thinking about ways to better model data for consumption by LLMs. Initially, this was for retrieval use cases and embeddings, which was further validated by Meta's LCM paper, which was not a star schema, but had similar goals as my own research (Large Concept Models)[https://github.com/facebookresearch/large_concept_model]
3. Better data access controls and row-level security - star schemas make row-level security significantly easier to manage and extend and scale

## Problem Statement

Like computers of the 80s and 90s, working with LLMs today (or any transformer model) requires not just managing context length, but also managing the actual *content* that goes into that context. Those of us working with LLMs for the past few years take for granted how intuitive this is to us, but for many others, a single wrong input can cascade into a downward spiral - and that's bad for adoption and bad for consistency and bad for security.

Perhaps more importantly, reusing context - such as the app you built a few weeks ago, synthesized with a new idea, synthesized with new data inputs, synthesized with a colleague's context or another's team's context - is something that few people know you can do effectively. That means having the LLM you're using to build the 'new' thing rewrite the 'old' thing in it's native "language". And, of course, pulling in the right data at the right time.

So we should aim to make it easier to manage and reuse context. Some more thoughts on the "why" explore storing context in a data lake / database:

- **Context Across Sessions Across Teams**: It's frustrating enough to lose your context on your own hobbyist project - imagine what's possible for a whole team.
- **Token Size Limitations and Context Confusion**: Loading entire codebases causes a good chunk of (what's in the middle to be lost)[https://arxiv.org/abs/2307.03172]. More importantly, though, flooding the LLM with tokens that are unhelpful at best or misleading at worst will lead you down a path of wasted time and less than wonderful results.
- **Tracking Versions, Evals, and Auditing**: Can't track what was previously understood, can't audit what was used for inputs, and can't evaluate how things are getting better, worse, same, etc.
- **File-based Scalability**: Linear scanning of files doesn't scale. Hopping between markdown files is painful. As wonderful as grep is for Claude Code, it can't hop between relationships
- **Relationship Understanding**: A star schema creates connections between code elements or other context that look like your business or domain
- **Data Access Controls**: This won't solve prompt injections or other LLM security issues, but it does move us in the right step toward ensuring that users leveraging these tools have access to only the data they are permissioned to have. This makes having your context in a data lake much more reliable than hitting the source systems directly - which lines up nicely with the same best practices prior to LLMs.

## What This Repo's For

This is a *research-oriented* repo - not a production one. If anyone's even looking at this, it's mostly intended for my own uses and explorations, building in open to stay accountable.
