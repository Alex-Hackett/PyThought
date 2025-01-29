# Linguistic Analysis Tool

A Python tool designed to analyze and compare the linguistic characteristics between an AI model's "thinking" process output and its final response. Specifically built to examine how models like DeepSeek transform their intermediate reasoning steps into polished final outputs, measuring changes in complexity, formality, and tone between the two stages of generation.

For example, an AI's "thinking" output might show more informal, step-by-step reasoning, while its final output could be more polished and formal. This tool helps quantify these differences through various linguistic metrics.

## Quick Start

```bash
pip install spacy textstat pandas textblob
python -m spacy download en_core_web_sm
```

```python
from text_analyzer import analyze_and_save

results, csv_file = analyze_and_save(thinking_output, final_output, prompt)
```

## Metrics Explained

### Readability Scores
- **Flesch Reading Ease**: 0-100 scale; higher scores mean easier to read. 60-70 is ideal for general audience
- **Flesch-Kincaid Grade**: US grade level required to understand the text
- **Gunning Fog**: Estimates years of formal education needed to understand text
- **SMOG Index**: Predicts reading grade level, optimized for healthcare materials
- **Coleman-Liau**: Grade level based on characters per word instead of syllables

### Complexity Metrics
- **Lexical Diversity**: Ratio of unique words to total words; higher values indicate more varied vocabulary
- **Dependency Distance**: Average distance between syntactically related words; higher values suggest more complex sentence structure
- **Noun-Verb Ratio**: Higher values typically indicate more formal, academic writing
- **Average Sentence Length**: Longer sentences often indicate more complex writing

### Formality Indicators
- **Personal Pronouns**: Higher usage suggests more informal, personal tone
- **Passive Voice**: Higher usage typically indicates more formal, academic writing
- **Nominalizations**: Verbs/adjectives converted to nouns (e.g., "discover" → "discovery"); more common in formal writing
- **Academic Vocabulary**: Longer, more specialized words; indicates technical or formal content

### Sentiment Analysis
- **Polarity**: -1 (negative) to 1 (positive)
- **Subjectivity**: 0 (objective) to 1 (subjective)
- **Sentiment Variance**: How much sentiment varies between sentences; higher values indicate more emotional range

## Output

Results are saved to timestamped CSV files with:
- Original texts and prompt
- All metrics for both texts
- Calculated differences
- Analysis timestamp
