# CloudTrail AI Anomaly Detector

CLI tool that retrieves AWS CloudTrail logs and analyzes them for security anomalies using a 4-tier detection pipeline with persistent historical context:

1. **Rule-based** — instant pattern matching for known-bad events
2. **Baseline comparison** — flags never-before-seen users, IPs, API calls, and services
3. **ML (Isolation Forest)** — statistical outlier detection using baseline-aware frequency features
4. **LLM (local)** — natural-language reasoning on ML-flagged events only

A persistent baseline file accumulates across runs, so the tool gets smarter over time. Activity that was normal yesterday won't be flagged today — but a brand-new IAM user or never-seen IP will be.

All analysis runs locally. No cloud AI APIs are called — CloudTrail data never leaves your machine.

## Prerequisites

- Python 3.10+
- AWS credentials configured (`aws configure` or environment variables)
- [LM Studio](https://lmstudio.ai/) running locally on port 1234 (optional, for Tier 4)

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

### Baseline Options

The baseline file (`~/.cloudtrail_baseline.json`) persists across runs, accumulating frequency data for users, IPs, API calls, services, regions, and per-user hour-of-day patterns.

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline-file FILE` | `~/.cloudtrail_baseline.json` | Path to baseline JSON file |
| `--no-baseline` | off | Disable baseline (single-run mode, no historical context) |
| `--reset-baseline` | off | Delete existing baseline and start fresh |
| `--yesterday` | off | Analyze yesterday's S3 logs (requires `--s3-uri`) |

### Analysis Options

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-ml` | off | Skip ML anomaly detection (Tier 3) |
| `--ml-contamination` | 0.05 | Expected anomaly fraction for Isolation Forest (0.01-0.10) |
| `--ml-top-n` | 20 | Number of top ML anomalies to display |
| `--skip-ai` | off | Skip LLM analysis (Tier 4) |
| `--batch-size` | 20 | Events per LLM analysis batch |
| `--output-json FILE` | none | Export full report as JSON |

### Examples

```bash
# First run: build baseline from a known-good day
python cloudtrail_analyzer.py \
  --s3-uri "s3://bucket/.../2026/" \
  --month 1 --days 20 \
  --max-events 5000 --skip-ai

# Second run: analyze a different day with baseline context
python cloudtrail_analyzer.py \
  --s3-uri "s3://bucket/.../2026/" \
  --month 1 --days 28 \
  --max-events 5000 --skip-ai

# Daily cron job: analyze yesterday, grow baseline, export JSON
python cloudtrail_analyzer.py \
  --s3-uri "s3://bucket/.../2026/" \
  --yesterday --max-events 5000 \
  --skip-ai --output-json /reports/$(date +%F).json

# Full pipeline with all 4 tiers
python cloudtrail_analyzer.py --hours 48 --output-json report.json

# Single-run mode (no baseline), rules + ML only
python cloudtrail_analyzer.py --hours 24 --no-baseline --skip-ai

# Reset baseline and start fresh
python cloudtrail_analyzer.py \
  --s3-uri "s3://bucket/.../2026/" \
  --month 1 --days 15-20 \
  --reset-baseline --max-events 10000 --skip-ai
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

### Tier 2: Baseline Comparison (instant)
Compares current events against the persistent baseline to detect never-before-seen activity:

| Finding Type | Severity | What It Means |
|-------------|----------|---------------|
| `NEW_USER` | HIGH | IAM user or role ARN never seen in any previous run |
| `NEW_IP` | MEDIUM | Source IP address never seen before |
| `NEW_EVENT` | MEDIUM | API call (e.g. `CreatePolicy`) never seen before |
| `NEW_SERVICE` | MEDIUM | AWS service (e.g. `q.amazonaws.com`) never seen before |
| `UNUSUAL_HOUR` | LOW | Known user active at an hour where they have <1% of historical activity |

The baseline tracks per-entity first-seen and last-seen dates, cumulative frequency counters, and per-user hour-of-day distributions.

### Tier 3: ML — Isolation Forest (seconds)
Featurizes each CloudTrail event into an 11-dimension numeric vector and runs scikit-learn's Isolation Forest to find statistical outliers.

When a baseline exists, frequency features are computed against the **combined historical + current** dataset. This means a user seen 10,000 times in the baseline won't be flagged as "rare" even if they only appear once today.

**Features extracted per event:**

| Feature | Description |
|---------|-------------|
| `event_name_freq` | How rare this API call is (against baseline + current) |
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

### Tier 4: LLM Analysis (seconds, with ML pre-filter)
When ML runs first, only the top ML-flagged events are sent to the LLM — typically ~20 events instead of thousands. This reduces LLM inference by 90%+ while focusing analysis on the most suspicious activity.

The LLM provides:
- Natural-language explanation of why events are suspicious (or benign)
- Severity classification (CRITICAL / HIGH / MEDIUM / LOW / INFO)
- Cross-event pattern detection (privilege escalation chains, impossible travel, etc.)

## Output

The terminal report includes:
- **Event Summary** — counts of total events and each anomaly category
- **Rule-Based Findings** — color-coded flags with event details
- **New Activity vs. Baseline** — never-before-seen users, IPs, API calls, services
- **ML Anomaly Detection** — top outliers with anomaly scores and contributing features
- **AI Analysis Findings** — LLM severity-ranked findings with explanations

JSON export (`--output-json`) contains all findings across all 4 tiers.

## Baseline File

The baseline is stored as a JSON file (default: `~/.cloudtrail_baseline.json`) with:

```
total_events:    cumulative event count across all runs
runs:            number of analysis runs incorporated
dates_processed: list of date ranges analyzed
event_names:     {API_call: count}
event_sources:   {service: count}
users:           {arn: {count, first_seen, last_seen}}
ips:             {ip: {count, first_seen, last_seen}}
regions:         {region: count}
hour_by_user:    {arn: {hour: count}}
```

The file grows with each run. Use `--reset-baseline` to start fresh, or `--no-baseline` for one-off analysis without historical context.

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
