# Workflow reference — Industrial Maintenance DE

A German industrial-maintenance agent takes one fault from a controller log all the way to a signed-off replacement decision.

```bash
python -m capability_subnet.workflows.cli show --seed 42 --with-truth
```

---

## Why this workflow

It was chosen because it is **commercially understandable and technically objective at the same time** — a combination that is rarer than it sounds.

It exercises eight capabilities in one dependent chain, and every one of them can be judged without asking a model for an opinion:

| Capability | Judged by |
|---|---|
| German technical language | Whether the right facts were extracted from German prose |
| Structured log extraction | Whether the right channel and values were read |
| Fault reasoning | Exact match against a deterministic machine schema |
| Text-to-SQL | **Executing** the query against a hidden snapshot |
| Python code generation | Hidden test cases the agent never sees |
| Tool calling | Whether the tools were driven correctly at all |
| Safety-policy compliance | A deterministic rule engine |
| Strict structured output | JSON Schema plus exact value comparison |

---

## The chain

```
manual interpretation → fault extraction → maintenance SQL → diagnostic Python
    → inventory action → safety validation → strict final JSON
```

Later steps consume earlier outputs. The SQL question asks about "the fault code you determined" — it does not name it — so a candidate that failed the fault-extraction stage cannot answer the SQL stage either. That dependency is what makes this a workflow rather than seven independent benchmark questions.

---

## What the agent sees

At the start:

- the task instruction,
- the full controller log,
- the database schema description,
- the diagnostic function contract,
- the final report's JSON Schema.

Everything else is behind a tool. In particular the opening prompt contains **no threshold, no fault code, no part number, no procedure number and no safety step** — every one of those has to come out of the manual, the log, or a tool call.

### The seven tools

| Tool | Does |
|---|---|
| `read_manual(section_or_query)` | Returns a manual section. Accepts a section number, a title word, or a phrase. |
| `query_maintenance_db(sql)` | Executes a read-only SELECT against the hidden snapshot. |
| `run_diagnostic_python(code, input_id)` | Runs submitted code against the **public** input in an isolated interpreter. |
| `search_inventory(part_number)` | Looks up stock. |
| `reserve_inventory(part_number, quantity)` | Reserves stock. |
| `check_safety_plan(plan)` | Submits a plan to the deterministic rule engine. |
| `submit_final_json(payload)` | Submits the report and ends the run. |

**No tool ever reveals whether the candidate is right.** The inventory service accepts a reservation for the wrong part. The safety engine reports which required steps are missing but nothing about the fault. The code runner reports what the function returned, not whether it matches. Anything else would turn the tool surface into an answer key.

### Limits

| | |
|---|---|
| Turns | ≤ 12 |
| Output tokens | ≤ 8192 |
| Temperature | 0, seeded per instance |
| Wall clock | 300 s per instance |

---

## Instance generation

One integer seed produces one complete, self-consistent instance — machine, manual, log, database, code task, inventory, policy and ground truth. The same seed produces the same instance on any machine, which is what lets the public pack be distributed as a seed and lets a disputed evaluation be replayed exactly.

### The machine schema

Five machine families, each with three sensor channels and three replaceable components:

| Family | Components |
|---|---|
| Hydraulic press | pump, oil cooler, proportional valve |
| Conveyor | drive motor, idler bearing, frequency inverter |
| Chiller | compressor, expansion valve, level sensor |
| CNC lathe | spindle drive, spindle bearing, feed drive |
| Screw compressor | compressor stage, oil separator, refrigerant dryer |

Each machine gets **per-instance threshold offsets**, so a candidate that memorised a threshold from the public pack does not transfer.

### The log

Exactly one channel is driven into exceedance; the others stay nominal and act as distractors. The exceedance is shaped as one contiguous burst plus a couple of isolated spikes — deliberately, so the exceedance *count* differs from the longest *run*. A diagnostic implementation that conflates the two fails the hidden cases.

Every number is rounded before the ground truth is computed from it, so the truth is derived from exactly the values in the log — not from a higher-precision series the candidate never saw.

### The manual

Five sections in German technical register. Every fact the scorer checks appears in exactly one of them:

| Section | Contains |
|---|---|
| `.1` Safety notices | Required precautions per component, and prohibited actions |
| `.2` Limits | Machine-specific warning and critical thresholds |
| `.3` Fault diagnosis | Sensor → fault code → component, and the **procedure number** to cite |
| `.4` Spare parts | Current part numbers and installed quantities. Superseded numbers are blocked. |
| `.5` Decision rule | Peak above critical → immediate repair; above warning → planned maintenance |

---

## Scoring

Every number comes from comparing a stored trace with the instance's ground truth. No model is involved.

| Stage | Threshold | Full credit | Partial credit |
|---|---|---|---|
| `manual_interpretation` | 1.0 | Correct procedure number cited | 0.4 — read the diagnosis section but cited wrong |
| `fault_extraction` | 1.0 | Fault code **and** component correct | 0.5 — one of the two |
| `maintenance_sql` | 1.0 | Result set matches the reference answer | scaled by matching values |
| `diagnostic_python` | 0.8 | All hidden cases pass | fraction passing |
| `inventory_action` | 1.0 | Exactly the right part in the right quantity | 0.5 — right part, wrong quantity or extra reservations |
| `safety_validation` | 1.0 | Plan approved **and** report consistent | 0.5 — one of the two |
| `final_json` | 1.0 | Schema valid **and** every value matches | scaled by correct fields |

Partial credit exists where it carries information the comparator can use — a candidate that found the right component but wrote the fault code in the wrong format has a different problem from one that read the wrong section entirely. The thresholds are set so partial credit never counts as a pass.

**End-to-end success** requires every stage above its threshold, the final environment state correct, and **zero critical unsafe actions**.

### Critical unsafe actions

Proposing a prohibited action — in a plan submitted to the safety tool, or in the final report — is counted separately and gates the candidate outright.

Omitting a *required* precaution is a stage failure, not a critical action. Forgetting a precaution is a wrong answer; proposing a prohibited one is a different kind of failure, and the gate treats it as such.

---

## The diagnostic stage

```python
def analyze(readings: list[float], threshold: float) -> dict:
    return {"peak": float, "exceedances": int, "longest_run": int}
```

The candidate can run this against **one public input** through the tool. The score comes from **six hidden cases it never sees**, run after the agent loop has ended.

That split is the whole point of the stage. Code that special-cases the visible input passes the tool call and fails the hidden cases.

The hidden cases deliberately include the edges: an empty series, a series entirely below the threshold, and a series with values exactly *at* it. The contract says "strictly exceed", so a value equal to the threshold does not count — stated, not implied.

---

## Out-of-distribution mutations

Out-of-distribution instances apply two or three named mutations. They are not *harder* instances — they are the same problem stated differently, which is exactly what distinguishes a package that understood the task from one that memorised its surface.

| Mutation | Effect |
|---|---|
| `unit_conversion` | Log and thresholds in the alternative unit; the manual says to convert |
| `synonym_terminology` | Components renamed, with the new names defined in the manual |
| `schema_alias` | Database columns renamed to German equivalents |
| `fault_code_format` | Fault codes written dotted rather than hyphenated |
| `sibling_noise` | More rows from a sibling machine sharing the fault code |

That last one punishes a query that forgets to filter by machine.

Out-of-distribution robustness carries **10% of the qualified score** directly.

---

## Verifying the environment

```bash
python -m capability_subnet.workflows.cli selftest --count 20
python -m capability_subnet.workflows.cli selftest --count 20 --split ood
```

This runs a **scripted reference solver** — which knows the answers by construction, and is therefore useless as a candidate and exactly right as a check. Every instance should reach end-to-end success.

If it fails, the generator, the tools or the scorer is broken, and no candidate score from those instances would mean anything. Run it after any change to the workflow.

---

## The public pack

```bash
python -m capability_subnet.workflows.cli generate-public-pack --out data/public_pack
```

120 public instances plus 30 out-of-distribution ones, each with full ground truth and a SQLite copy of its maintenance database.

The hidden pack is generated by **exactly the same code** from a different seed range and never leaves the engine. Publishing the generator rather than the data is the point: a miner can study the distribution as long as it likes without ever seeing the instances it will be scored on.

The pack manifest carries a tree digest so you can confirm your copy matches the published one.
