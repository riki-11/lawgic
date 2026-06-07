# lawgic_tos_schema

This defines the structure of the JSON output produced by the LLM after ingesting a new Terms of Service document. Every ingested ToS — regardless of source — must conform to this schema exactly before it is written to the data store or consumed by the frontend.

---

```json
{
  "$schema_version": "1.0.0",

  "service_name": "Name of the service being analyzed (e.g. 'YouTube')",
  "service_id": "Stable, lowercase-slugified unique ID (e.g. 'youtube')",
  "service_url": "Canonical URL or list of URLs associated with this service's ToS",

  "document": {
    "tos_version": "Version string if stated in the document (e.g. 'v4.2', null if absent)",
    "tos_effective_date": "ISO 8601 date the ToS came into effect (e.g. '2024-01-15', null if absent)",
    "tos_last_updated": "ISO 8601 date the ToS was last updated as stated in the document (null if absent)",
    "document_language": "ISO 639-1 language code of the document (e.g. 'en', 'de', 'fr')",
    "word_count": 12400,
    "full_text": "Full raw text of the ToS in plain Markdown format. No HTML."
  },

  "ingestion": {
    "ingested_at": "ISO 8601 UTC timestamp of when this document was processed (e.g. '2025-06-01T10:32:00Z')",
    "source": "user_upload | scraped",
    "source_format": "pdf | docx | html | plain_text",
    "document_hash": "SHA-256 hash of the raw input document. Used for deduplication.",
    "llm_model_used": "Model identifier used for extraction (e.g. 'claude-sonnet-4-20250514')"
  },

  "analysis": {
    "overall_risk_score": -0.4,
    "risk_label": "high | moderate | low",
    "plain_english_summary": "A 2–3 sentence LLM-generated plain-language summary of the ToS as a whole, written for a non-legal audience.",
    "topic_breakdown": [
      {
        "topic_id": "data_sharing",
        "topic_name": "Data Sharing with Third Parties",
        "score": -1,
        "risk_label": "bad",
        "clause_count": 3
      },
      {
        "topic_id": "account_deletion",
        "topic_name": "Account Deletion",
        "score": 0,
        "risk_label": "neutral",
        "clause_count": 1
      }
    ]
  },

  "notable_clauses": [
    {
      "clause_id": "youtube_001",
      "clause_title": "Short, human-readable title for the clause (e.g. 'Right to terminate your account without notice')",
      "clause_text": "The verbatim text of the clause as it appears in the ToS.",
      "clause_summary": "Plain-language explanation of what this clause means for the user. Written for a non-legal audience.",
      "topic_ids": [
        "account_termination",
        "dispute_resolution"
      ],
      "primary_topic_id": "account_termination",
      "score": -1,
      "score_rationale": "The LLM's explanation for why this score was assigned, grounded in the scoring criteria for the primary topic.",
      "llm_confidence": 0.91,
      "clause_text_start": 4821,
      "clause_text_end": 5103
    }
  ]
}
```

---

## Field Reference

### Top Level

| Field | Type | Description |
|---|---|---|
| `$schema_version` | `string` | Version of this schema. Increment on breaking changes to enable migration logic. |
| `service_name` | `string` | Display name of the service. |
| `service_id` | `string` | Stable slug ID. Must be unique per service across all documents. |
| `service_url` | `string` | Canonical ToS URL(s). |

---

### `document`

Metadata about the ToS document itself, distinct from when/how it was ingested.

| Field | Type | Description |
|---|---|---|
| `tos_version` | `string \| null` | Version string as stated in the document. |
| `tos_effective_date` | `string \| null` | ISO 8601. When the ToS officially came into effect. |
| `tos_last_updated` | `string \| null` | ISO 8601. Last updated date as written in the document. |
| `document_language` | `string` | ISO 639-1 language code. |
| `word_count` | `number` | Approximate word count of `full_text`. Useful for chunking decisions. |
| `full_text` | `string` | Full raw text in Markdown. No HTML tags. |

---

### `ingestion`

Provenance metadata. Tracks how and when the document entered the pipeline.

| Field | Type | Description |
|---|---|---|
| `ingested_at` | `string` | ISO 8601 UTC timestamp of pipeline processing. |
| `source` | `enum` | `user_upload` or `scraped`. |
| `source_format` | `enum` | `pdf`, `docx`, `html`, or `plain_text`. |
| `document_hash` | `string` | SHA-256 of the raw input. Used to prevent duplicate processing. |
| `llm_model_used` | `string` | Model identifier for reproducibility and debugging. |

---

### `analysis`

Service-level aggregate output. Powers the overview/dashboard layer of the React Flow UI.

| Field | Type | Description |
|---|---|---|
| `overall_risk_score` | `number` | Weighted average score across all clauses, factoring in `severity_weight` from topics. Range: `-1.0` to `1.0`. |
| `risk_label` | `enum` | `high`, `moderate`, or `low`. Derived from `overall_risk_score`. |
| `plain_english_summary` | `string` | 2–3 sentence LLM-generated summary written for a non-legal audience. |
| `topic_breakdown` | `array` | Per-topic rollup: score, label, and clause count. Used by the UI topic overview. |

---

### `notable_clauses[]`

Individual clause-level extractions. Each entry maps to a node in the React Flow canvas.

| Field | Type | Description |
|---|---|---|
| `clause_id` | `string` | Unique ID. Recommended format: `{service_id}_{zero-padded-index}` (e.g. `youtube_001`). |
| `clause_title` | `string` | Short, human-readable label for the clause. Used as the node title in the UI. |
| `clause_text` | `string` | Verbatim quote from the ToS. |
| `clause_summary` | `string` | Plain-language interpretation for non-legal users. |
| `topic_ids` | `string[]` | Array of all applicable topic IDs from `lawgic_tos_topics.json`. Min 1, no upper limit. |
| `primary_topic_id` | `string` | The single most relevant topic. Used for primary node classification and scoring. |
| `score` | `number` | `-1`, `0`, or `1`. Assigned against the scoring criteria of `primary_topic_id`. |
| `score_rationale` | `string` | LLM's justification for the score. Must reference the topic's scoring criteria. |
| `llm_confidence` | `number` | `0.0`–`1.0`. LLM's self-reported confidence. Clauses below `0.6` should be flagged in the UI. |
| `clause_text_start` | `number` | Character index of clause start within `full_text`. |
| `clause_text_end` | `number` | Character index of clause end within `full_text`. |


## Things to Improve:
- This seems like a lot to do for one single prompt. Maybe we can chain the prompts or build up this schema in multiple prompts/phases?
- I don't know yet about the "service-level" score and risk value... We don't have any objective metrics... so our LLM is just the one making it up.