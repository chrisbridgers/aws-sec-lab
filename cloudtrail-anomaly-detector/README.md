# CloudTrail AI Anomaly Detector

CLI tool that retrieves AWS CloudTrail logs and analyzes them for security anomalies using a 3-tier detection pipeline:

1. **Rule-based** — instant pattern matching for known-bad events
2. **ML (Isolation Forest)** — statistical outlier detection on featurized events
3. **LLM (local)** — natural-language reasoning on ML-flagged events only

All analysis runs locally. No cloud AI APIs are called — CloudTrail data never leaves your machine.

## Prerequisites

- Python 3.10+
- AWS credentials configured (`aws configure` or environment variables)
- [LM Studio](https://lmstudio.ai/) running locally on port 1234 (optional, for Tier 3)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `boto3`, `openai`, `scikit-learn`, `numpy`

## Usage

### Live CloudTrail API (last 90 days)

Queries the CloudTrail `lookup_events()` API directly:

```bash
python cloudtrail_analyzer.py --hours 24 --region us-east-1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--hours` | 24 | Lookback window in hours |
| `--region` | us-east-1 | AWS region to query |
| `--profile` | default | AWS CLI profile name |
| `--max-events` | 200 | Maximum events to retrieve |

### Historical S3 Logs

For logs older than 90 days stored in S3. Point `--s3-uri` at the year-level prefix:

```bash
python cloudtrail_analyzer.py \
  --s3-uri "s3://bucket/AWSLogs/org-id/account-id/CloudTrail/us-east-1/2025/"
```

S3 log structure: `.../CloudTrail/region/YYYY/MM/DD/*.json.gz`

When `--s3-uri` is used, the tool will interactively prompt for:

1. **Month** (1-12) — numbered list of months
2. **Day(s)** — single day, day range, or Enter for the entire month

To skip prompts (for scripting/CI), pass `--month` and `--days` on the command line:

```bash
# Single day
python cloudtrail_analyzer.py --s3-uri "s3://bucket/.../2025/" --month 3 --days 15

# Day range
python cloudtrail_analyzer.py --s3-uri "s3://bucket/.../2025/" --month 3 --days 10-20

# Entire month (prompted for day selection)
python cloudtrail_analyzer.py --s3-uri "s3://bucket/.../2025/" --month 3
```

### Analysis Options

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-ml` | off | Skip ML anomaly detection (Tier 2) |
| `--ml-contamination` | 0.05 | Expected anomaly fraction for Isolation Forest (0.01-0.10) |
| `--ml-top-n` | 20 | Number of top ML anomalies to display |
| `--skip-ai` | off | Skip LLM analysis (Tier 3) |
| `--batch-size` | 20 | Events per LLM analysis batch |
| `--output-json FILE` | none | Export full report as JSON |

### Examples

```bash
# Quick rule-based + ML scan, no LLM needed
python cloudtrail_analyzer.py --hours 12 --skip-ai

# Full 3-tier pipeline with JSON export
python cloudtrail_analyzer.py --hours 48 --output-json report.json

# Historical S3 logs, high sensitivity, all 3 tiers
python cloudtrail_analyzer.py \
  --s3-uri "s3://bucket/.../2025/" \
  --month 3 --days 10-20 \
  --max-events 2500 \
  --ml-contamination 0.10 \
  --output-json report.json

# Rules only (fastest, no dependencies beyond boto3)
python cloudtrail_analyzer.py --hours 24 --skip-ml --skip-ai
```

## Detection Pipeline

### Tier 1: Rule-Based (instant)
Pattern matching against known-suspicious CloudTrail events:
- Failed console logins (`ConsoleLogin` with error)
- `AccessDenied` / `UnauthorizedAccess` errors
- IAM privilege escalation (CreateUser, AttachUserPolicy, CreateAccessKey, etc.)
- STS AssumeRole denials
- Security group and network ACL changes
- CloudTrail/GuardDuty/Config tampering (StopLogging, DeleteTrail, DisableRule, etc.)

### Tier 2: ML — Isolation Forest (seconds)
Featurizes each CloudTrail event into an 11-dimension numeric vector and runs scikit-learn's Isolation Forest to find statistical outliers.

**Features extracted per event:**

| Feature | Description |
|---------|-------------|
| `event_name_freq` | How rare this API call is in the dataset |
| `event_source_freq` | How rare this AWS service is |
| `user_freq` | How rare this IAM principal is |
| `ip_hash` | Hashed source IP (captures diversity) |
| `ip_freq` | How rare this source IP is |
| `hour_of_day` | Hour (0-23), normalized |
| `day_of_week` | Day (Mon-Sun), normalized |
| `has_error` | Whether an error code is present |
| `is_write_event` | Write vs. read-only event |
| `region_freq` | How rare this AWS region is |
| `user_type_code` | User identity type (Root, IAMUser, AssumedRole, etc.) |

Events with rare API calls, unusual IPs, off-hours timing, or error codes score as more anomalous. The `--ml-contamination` flag controls sensitivity (higher = more flags).

### Tier 3: LLM Analysis (seconds, with ML pre-filter)
When ML runs first, only the top ML-flagged events are sent to the LLM — typically ~20 events instead of thousands. This reduces LLM inference by 90%+ while focusing analysis on the most suspicious activity.

The LLM provides:
- Natural-language explanation of why events are suspicious (or benign)
- Severity classification (CRITICAL / HIGH / MEDIUM / LOW / INFO)
- Cross-event pattern detection (privilege escalation chains, impossible travel, etc.)

Without ML (`--skip-ml`), all events are sent to the LLM in batches.

## Output

The terminal report includes:
- **Event Summary** — counts of total events and each anomaly category
- **Rule-Based Findings** — color-coded flags with event details
- **ML Anomaly Detection** — top outliers with anomaly scores and contributing features
- **AI Analysis Findings** — LLM severity-ranked findings with explanations

JSON export (`--output-json`) contains all findings across all 3 tiers.

## LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model — recommended options:

| Model | Size | Notes |
|-------|------|-------|
| Qwen2.5-7B-Instruct | 7B | Best reasoning-per-VRAM at small size |
| Mistral-7B-Instruct-v0.3 | 7B | Fast, good at structured output |
| DeepSeek-Coder-V2-Lite | 16B | Strong at structured analysis |
| Qwen QwQ-32B | 32B | Most capable, but GPU-intensive |

3. Start the local server (defaults to `http://localhost:1234`)
4. The tool connects automatically — no API key required
5. The model name in the script defaults to `qwen-qwq-32b` — LM Studio will route to whatever model is loaded regardless of the name
