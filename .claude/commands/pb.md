Run the PitchBook Observer Agent against the query provided by the user.

## What this agent does

The PitchBook Observer Agent answers questions about public deal activity — funding rounds, acquisitions, IPOs, and investor behavior — by:

1. Parsing intent from the query
2. Retrieving relevant chunks from the indexed knowledge base (hybrid vector + keyword search)
3. Reranking results using cross-encoder scoring
4. Augmenting with episodic memory context (SmartCard)
5. Generating a grounded answer with citations

## How to invoke

Use the Bash tool to run the agent:

```bash
cd /path/to/PB
python -m pipeline.agents.runtime_agent "$ARGUMENTS"
```

If no query is provided in $ARGUMENTS, ask the user what they want to know about deals, funding, or market activity.

## Output format

Present the response with:
- The answer text
- Confidence level (high / medium / low)
- Top citations (source, title, relevance score)
- Timing breakdown (retrieval, reranking, generation)
- Memory augmentation status (active / inactive)

## Examples

- `/pb what funding rounds were announced this week?`
- `/pb which investors are most active in AI?`
- `/pb tell me about recent acquisitions in fintech`
- `/pb compare NeuralPath AI and QuantumLeap Computing`
