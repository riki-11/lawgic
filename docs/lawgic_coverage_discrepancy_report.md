# Report: The Coverage Discrepancy in the Lawgic Fusion Pipeline

Language: ASD-STE100 Simplified Technical English.
Date: 31 July 2026.
Scope: the taxonomy, the fusion notebook, the built corpus, and the documents.

---

## 1. Summary

The documents report the wrong coverage numbers. The code also has a defect.
The defect is not a typing error. Two different rules build the labels and the mask.

There are four findings:

1. The mask uses the raw taxonomy file. The labels use an override table. The two do not agree.
2. The notebook prints the correct numbers. Nobody saved the printed output.
3. CLAUDETTE uses one rule. ToS;DR and 100 ToS use a different rule.
4. The ToS;DR override table is incomplete. Five labels still make too many topics.

---

## 2. The correct numbers

Count the topics that each source can supervise. There are two methods.

Method A counts a topic if the source list in `lawgic_topics.json` is not empty.
Method B counts a topic after you apply the override tables in the notebook.

| Source | Method A (raw file) | Method B (after overrides) |
| --- | --- | --- |
| CLAUDETTE | 21 | 10 |
| ToS;DR | 42 | 38 |
| 100 ToS | 30 | 30 |

The documents report 37 for ToS;DR and 26 for 100 ToS. No method gives these numbers.
The git history has only one version of `lawgic_topics.json`. That version gives 42 and 30.
Therefore the numbers 37 and 26 never matched the data. Somebody wrote them by hand.

---

## 3. Root cause 1: two rules build the corpus

The pipeline uses two different rules. This is the primary defect.

**Rule for the labels.** The function `map_tosdr_topic()` is in cell 10 of
`notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb`. The function reads
`TOSDR_TOPIC_OVERRIDES` first. The overrides make the label set narrow.

**Rule for the mask.** The function `build_source_coverage_mask()` is in cell 20 of the
same notebook. The function reads `source_mappings` from the taxonomy file. The function
does not read the overrides. The mask is therefore wide.

The mask is wider than the labels. Four topics are inside the ToS;DR mask. But ToS;DR can
never give a positive label to these four topics. Every ToS;DR row therefore gives a
negative answer for these four topics.

Measurements from `lawgic_multihead_wide.csv`:

| Topic | ToS;DR positives | Total positives | Supervised negatives | Ratio |
| --- | --- | --- | --- | --- |
| `price_changes` | 0 | 50 | 23,337 | 1 : 467 |
| `liability_cap` | 0 | 57 | 23,330 | 1 : 409 |
| `service_changes` | 0 | 115 | 23,272 | 1 : 202 |
| `limitation_of_liability` | 0 | 1,186 | 23,243 | 1 : 20 |

The reason for each result:

- `Guarantee` goes only to `warranty_disclaimer`. Therefore ToS;DR cannot reach
  `limitation_of_liability` or `liability_cap`.
- `Changes` goes only to `contract_changes`. Therefore ToS;DR cannot reach
  `price_changes` or `service_changes`.

These negative answers are false by construction. The design of section 9 must stop false
negative answers. The design does not stop them here.

### The `Waivers` override is dead code

The override `Waivers` goes to `indemnification`. But the ToS;DR data never uses the
label `Waivers`. There are 0 rows. The 100 ToS data never uses the code `indemn`.
There are 0 rows.

Two results follow:

- `indemnification` has 0 positives and 23,387 negatives. The model cannot learn it.
- The override removed `limitation_of_liability` and `liability_cap` from the ToS;DR
  label set. The override gained nothing, because `Waivers` does not occur.

---

## 4. Root cause 2: the notebook output was not saved

Cell 20 contains this line:

```python
print(f"  {src} taxonomy coverage: {covered} / {len(topic_id_to_index)} topics")
```

The cell prints the correct numbers 42, 30, and 21. The saved notebook has no output for
cell 20. The field `execution_count` is `None` for every code cell. Somebody cleared the
outputs before the save.

Therefore no document quotes a measured number. All documents quote an estimated number.

---

## 5. Root cause 3: the three sources use two different bases

The function `compute_source_aware_mask()` is in cell 21. The function does this:

- ToS;DR: use `source_coverage`. This is Method A. The width is 42.
- 100 ToS: use `source_coverage`. This is Method A. The width is 30.
- CLAUDETTE: use `CLAUDETTE_TOPIC_RULES`. This is Method B. The width is 1 to 4.

The code mixes Method A and Method B in one function. Your sentence "CLAUDETTE 10, ToS;DR
42, 100 ToS 30" mixes the same two methods. The sentence copies the defect in the code.

Measured mask width per clause in the built corpus:

| Sources for the clause | Rows | Mask width |
| --- | --- | --- |
| ToS;DR only | 21,876 | 42 |
| 100 ToS only | 1,402 | 30 |
| CLAUDETTE only | 3,092 | 1 to 4, mean 1.16 |
| CLAUDETTE and ToS;DR | 51 | 42 |
| 100 ToS and CLAUDETTE | 36 | 30 |
| 100 ToS and ToS;DR | 19 | 44 |
| All three | 3 | 44 |

Note the maximum CLAUDETTE width. It is 4, not 2. One clause can carry more than one
CLAUDETTE label. The value 1 to 2 is correct for one annotation. The value 1 to 4 is
correct for one clause.

---

## 6. Root cause 4: five ToS;DR labels still make too many topics

Section 3.2 of the dataset report says that the overrides stop the fan-out problem. The
override table has five entries. Five other labels still fan out.

| ToS;DR label | Points | Long rows | Topics made |
| --- | --- | --- | --- |
| `Governance` | 2,476 | 9,904 | 4 |
| `Suspension and Censorship` | 1,149 | 4,596 | 4 |
| `Transparency` | 1,381 | 4,143 | 3 |
| `User Choice` | 886 | 2,658 | 3 |
| `Ownership` | 411 | 1,233 | 3 |

ToS;DR has 22,404 points. The pipeline made 44,317 long rows from them. The fan-out factor
is 1.98. The pipeline made 21,913 extra rows. The five labels above made 16,231 of them.

This contradicts the rule in section 3. The rule says that a source must not assert a
distinction that the source does not annotate. The pipeline applies this rule to CLAUDETTE
`ltd`. The pipeline does not apply it to ToS;DR `Governance`.

---

## 7. A third near-duplicate group

The scan compared all 44 label columns. The scan found two identical pairs. The scan also
found one very close group.

**Identical pairs. Difference is 0 rows.**

- `transfer_of_contract` and `business_transfer`
- `recommender_transparency` and `transparency`

**Near-identical group. Difference is 27 rows out of 26,479.**

- `transparency` (1,368 positives)
- `recommender_transparency` (1,368 positives)
- `interpretation_clause` (1,341 positives)

All three columns agree on 99.90 percent of rows. The cause is the ToS;DR label
`Transparency`. It makes all three topics. Only 27 rows from 100 ToS separate them.

Your fix brief in `docs/lawgic_duplicate_topics_fix.md` removes
`recommender_transparency`. After that change, `interpretation_clause` stays. It is still
a near-copy of `transparency`. The fix brief does not cover this.

**Not a duplicate.** `contract_by_use` and `governance` have identical `source_mappings`.
But the CLAUDETTE rule sends `use` only to `contract_by_use`. The two columns differ on
366 rows. They are nested, not identical. State this correctly.

---

## 8. The files and what each one causes

| File | Part | Effect |
| --- | --- | --- |
| `generated_files/lawgic_taxonomy/lawgic_topics.json` | `source_mappings` | Gives the raw coverage 42, 30, 21. Three topic groups share the same lists. |
| `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb` cell 4 | `TOSDR_TOPIC_OVERRIDES`, `CLAUDETTE_TOPIC_RULES` | Defines the narrow label rule. |
| Same notebook cell 10 | `map_tosdr_topic()` | Applies the overrides to the labels only. |
| Same notebook cell 20 | `build_source_coverage_mask()` | Ignores the overrides. Makes the wide mask. This is the primary defect. |
| Same notebook cell 21 | `compute_source_aware_mask()` | Mixes the wide mask and the narrow CLAUDETTE rule. |
| `generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv` | `topic_mask` | Contains the false negative answers. |
| `docs/lawgic_dataset_report.md` lines 186 and 187 | Coverage table | Reports 37 and 26. Both are wrong. |
| `docs/lawgic_dual_head_architecture.md` lines 161 and 162 | Coverage table | Reports 37 and 26. Both are wrong. |
| `docs/lawgic_duplicate_topics_fix.md` | Defect section | Lists two pairs. Does not list the transparency group of three. |

---

## 9. What to do in the code

Do these steps in this order.

1. Select one basis for the mask. Use Method B. Build the coverage vector from the same
   override tables that build the labels. Do not read `source_mappings` directly.
2. Change `build_source_coverage_mask()`. Compute the coverage as the union of the topics
   that `map_tosdr_topic()` can return. Do the same for 100 ToS and CLAUDETTE.
3. Run the notebook. Save the printed output. Do not clear the outputs before you commit.
4. Add a check. The build must fail if the mask covers a topic that the source cannot
   label positively.
5. Add a check. The build must fail if two label columns are identical.
6. Decide about the five fan-out labels. Either add them to `TOSDR_TOPIC_OVERRIDES`, or
   write the reason why `Governance` may assert four topics but `ltd` may not assert five.
7. Remove or justify the `Waivers` override. The label does not occur in the data.
8. Rebuild the corpus. Retrain. Report the new numbers.

Step 6 changes the corpus a lot. Do step 1 to step 5 first. Then measure. Then decide
about step 6.

---

## 10. What to do in the manuscript

The manuscript is in the other repository. Make these changes.

1. Correct the coverage table. Write 42 and 30 if you keep the current build. Write the
   new numbers if you rebuild.
2. State the basis of each number in the same sentence. Use one sentence for the rule and
   one sentence for the number.
3. Add the mask width per clause. State that CLAUDETTE gives 1 to 2 cells per annotation
   and 1 to 4 cells per clause.
4. Add a limitation. Say that four topics receive negative answers from a source that
   cannot give them a positive answer. Give the counts from section 3 of this report.
5. Add a limitation. Say that five ToS;DR labels expand to three or four topics. Give the
   fan-out factor 1.98.
6. Correct the duplicate section. Report two identical pairs and one near-identical group
   of three. Report `contract_by_use` and `governance` as nested, not identical.
7. Keep the sentence about `indemnification`. Add the cause. Neither source emitted the
   label. The topic is unlearnable because of the data, not because of the model.

Do not remove the disclosure of the defects. The defects support your argument. The first
fusion attempt failed and you reported it. These findings are the same kind of evidence.

---

## 11. What to do in the slides

Slide 53 shows source-aware masking. The slide is correct for CLAUDETTE. The CLAUDETTE
example uses a `ch` label. The mask width is 1. This matches the code.

Make these changes.

1. Change the footer. Do not write "CLAUDETTE 1 to 2, ToS;DR 42, 100 ToS 30" as one list.
   The numbers use two different rules. Write two lines instead:
   - "CLAUDETTE: the mask covers only the topics of the label on the row. Width 1 to 2."
   - "ToS;DR and 100 ToS: the mask covers all topics that the source can label. Width 42
     and 30."
2. Do not say "contributed". Say "can answer" or "covers".
3. Add one sentence to your spoken notes. Say that the wide mask is a heuristic. Say that
   you found four topics where the heuristic gives false negative answers. Say that you
   report this in the limitations.

Do not add the fan-out data to slide 53. It is a different problem. Put it on the
limitations slide.

If a panel member asks about the difference between 42 and 38, answer with three
sentences:

- "The mask uses the raw taxonomy file. That gives 42."
- "The labels use an override table. That gives 38."
- "The two must agree. I found the difference, I measured the effect, and I report it."

---

## 12. Order of work before the defense

1. Correct the two documents that report 37 and 26. This takes minutes.
2. Correct the slide footer. This takes minutes.
3. Run the notebook and save the output. This gives you measured numbers.
4. Write the two new limitations into the manuscript.
5. Do the code fix and the rebuild only if you have time. If you do not have time, report
   the defect and give the plan.

A measured defect with a plan is acceptable in a proposal defense. An unmeasured number
is not.
