#!/usr/bin/env python3
"""CloudTrail AI Anomaly Detector

Retrieves AWS CloudTrail logs and analyzes them for security anomalies
using rule-based pre-filtering and a local LM Studio model (Qwen QwQ-32B).
"""

import argparse
import functools
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Unbuffered print so background/piped runs show live progress
print = functools.partial(print, flush=True)

import boto3
from openai import OpenAI

# ── ANSI colors ──────────────────────────────────────────────────────────────

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SEVERITY_COLORS = {
    "CRITICAL": RED + BOLD,
    "HIGH": RED,
    "MEDIUM": YELLOW,
    "LOW": GREEN,
    "INFO": DIM,
}

# ── Suspicious event patterns ───────────────────────────────────────────────

# IAM / privilege escalation actions
IAM_PRIV_ACTIONS = {
    "CreateUser",
    "CreateRole",
    "AttachUserPolicy",
    "AttachRolePolicy",
    "PutUserPolicy",
    "PutRolePolicy",
    "CreateAccessKey",
    "CreateLoginProfile",
    "UpdateLoginProfile",
    "AddUserToGroup",
    "AttachGroupPolicy",
    "PutGroupPolicy",
}

# Security-service tampering actions
TAMPER_ACTIONS = {
    "StopLogging",
    "DeleteTrail",
    "UpdateTrail",
    "PutEventSelectors",
    "DeleteDetector",
    "DisableRule",
    "DeleteRule",
    "DeleteConfigRule",
    "StopConfigurationRecorder",
    "DeleteFlowLogs",
}

# Network / security-group mutations
NETWORK_ACTIONS = {
    "AuthorizeSecurityGroupIngress",
    "AuthorizeSecurityGroupEgress",
    "RevokeSecurityGroupIngress",
    "RevokeSecurityGroupEgress",
    "CreateSecurityGroup",
    "DeleteSecurityGroup",
    "ModifyInstanceAttribute",
    "CreateNetworkAclEntry",
    "ReplaceNetworkAclEntry",
}

# ── CloudTrail retrieval ─────────────────────────────────────────────────────


def fetch_cloudtrail_events(region, profile, hours, max_events):
    """Retrieve recent CloudTrail events via lookup_events()."""
    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(region_name=region, **session_kwargs)
    client = session.client("cloudtrail")

    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    end_time = datetime.now(timezone.utc)

    events = []
    kwargs = {
        "StartTime": start_time,
        "EndTime": end_time,
        "MaxResults": min(max_events, 50),  # API max per page
    }

    print(f"{CYAN}Fetching CloudTrail events ({hours}h window, region={region})...{RESET}")

    while True:
        resp = client.lookup_events(**kwargs)
        for event in resp.get("Events", []):
            raw = event.get("CloudTrailEvent")
            if raw:
                parsed = json.loads(raw)
            else:
                parsed = {
                    "eventName": event.get("EventName", "Unknown"),
                    "eventSource": event.get("EventSource", ""),
                    "eventTime": str(event.get("EventTime", "")),
                    "username": event.get("Username", ""),
                }
            events.append(parsed)
            if len(events) >= max_events:
                break

        if len(events) >= max_events:
            break
        token = resp.get("NextToken")
        if not token:
            break
        kwargs["NextToken"] = token

    print(f"{CYAN}Retrieved {len(events)} events.{RESET}\n")
    return events


def parse_s3_uri(uri):
    """Parse 's3://bucket/key/prefix' into (bucket, prefix)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with s3://): {uri}")
    path = uri[5:]
    bucket, _, prefix = path.partition("/")
    return bucket, prefix


def fetch_s3_events(s3_uri, profile, max_events, month=None,
                    start_day=None, end_day=None):
    """Download and parse CloudTrail log files from S3.

    CloudTrail stores logs as gzipped JSON files in the structure:
      .../CloudTrail/region/YYYY/MM/DD/*.json.gz

    Args:
        month:     Month number (1-12) to narrow the prefix.
        start_day: First day to include (1-31). Requires month.
        end_day:   Last day to include (1-31). Requires start_day.
                   If omitted, only start_day is fetched.
    """
    bucket, prefix = parse_s3_uri(s3_uri)

    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    # Ensure prefix ends with /
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # Build list of prefixes to scan based on month/day range
    prefixes = []
    if month is not None:
        month_prefix = prefix + f"{int(month):02d}/"
        if start_day is not None:
            ed = end_day if end_day is not None else start_day
            for d in range(int(start_day), int(ed) + 1):
                prefixes.append(month_prefix + f"{d:02d}/")
        else:
            # Whole month
            prefixes.append(month_prefix)
    else:
        prefixes.append(prefix)

    # Collect all .json.gz keys across all target prefixes
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for pfx in prefixes:
        print(f"{CYAN}Listing s3://{bucket}/{pfx}...{RESET}")
        for page in paginator.paginate(Bucket=bucket, Prefix=pfx):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json.gz"):
                    keys.append(key)

    if not keys:
        print(f"{YELLOW}No .json.gz files found for the specified date range.{RESET}")
        return []

    keys.sort()
    print(f"{CYAN}Found {len(keys)} log file(s). Downloading...{RESET}")

    events = []
    for key in keys:
        if len(events) >= max_events:
            break
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            compressed = resp["Body"].read()
            raw_json = gzip.decompress(compressed)
            data = json.loads(raw_json)
            records = data.get("Records", [])
            remaining = max_events - len(events)
            events.extend(records[:remaining])
        except Exception as e:
            print(f"{YELLOW}Skipping {key}: {e}{RESET}")

    print(f"{CYAN}Retrieved {len(events)} events from S3.{RESET}\n")
    return events


# ── Rule-based pre-filtering ─────────────────────────────────────────────────


def prefilter_events(events):
    """Flag known-suspicious patterns. Returns (flagged, summary_stats)."""
    flagged = []
    stats = {
        "total": len(events),
        "access_denied": 0,
        "failed_logins": 0,
        "iam_priv_actions": 0,
        "tamper_actions": 0,
        "network_changes": 0,
        "sts_denied": 0,
    }

    for ev in events:
        name = ev.get("eventName", "")
        error_code = ev.get("errorCode", "")
        error_msg = ev.get("errorMessage", "")
        source = ev.get("eventSource", "")
        reasons = []

        # Failed console logins
        if name == "ConsoleLogin":
            response = ev.get("responseElements", {})
            if isinstance(response, dict) and response.get("ConsoleLogin") == "Failure":
                reasons.append("Failed console login")
                stats["failed_logins"] += 1

        # AccessDenied / UnauthorizedAccess errors
        if error_code in ("AccessDenied", "Client.UnauthorizedAccess", "UnauthorizedAccess"):
            reasons.append(f"AccessDenied: {error_msg or error_code}")
            stats["access_denied"] += 1

        # STS denied
        if "sts" in source.lower() and name == "AssumeRole" and error_code:
            reasons.append(f"STS AssumeRole denied: {error_code}")
            stats["sts_denied"] += 1

        # IAM privilege escalation
        if name in IAM_PRIV_ACTIONS:
            severity = "HIGH" if error_code == "" else "MEDIUM"
            reasons.append(f"IAM privilege action: {name}")
            stats["iam_priv_actions"] += 1

        # Security-service tampering
        if name in TAMPER_ACTIONS:
            reasons.append(f"Security-service tampering: {name}")
            stats["tamper_actions"] += 1

        # Network / SG changes
        if name in NETWORK_ACTIONS:
            reasons.append(f"Network/SG change: {name}")
            stats["network_changes"] += 1

        if reasons:
            flagged.append({"event": ev, "reasons": reasons})

    return flagged, stats


# ── ML-based anomaly detection (Isolation Forest) ────────────────────────────


def _hash_encode(value, buckets=64):
    """Deterministic hash of a string into a bucket index (0 to buckets-1)."""
    if not value:
        return 0
    h = int(hashlib.md5(value.encode("utf-8", errors="replace")).hexdigest(), 16)
    return h % buckets


def _parse_event_time(ev):
    """Best-effort parse of eventTime into a datetime."""
    raw = ev.get("eventTime", "")
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except (ValueError, TypeError):
            continue
    return None


def featurize_events(events):
    """Convert CloudTrail events into numeric feature vectors.

    Features per event:
      0  event_name_freq      — how rare is this API call in the dataset
      1  event_source_freq    — how rare is this service
      2  user_freq            — how rare is this principal
      3  ip_hash              — hashed source IP (captures diversity)
      4  ip_freq              — how rare is this IP
      5  hour_of_day          — 0-23
      6  day_of_week          — 0-6
      7  has_error            — 1 if errorCode present
      8  is_write_event       — 1 if not readOnly
      9  region_freq          — how rare is this region
     10  user_type_code       — ordinal for userIdentity.type
    """
    # Pre-compute frequency tables from the full dataset
    name_counts = Counter(ev.get("eventName", "") for ev in events)
    source_counts = Counter(ev.get("eventSource", "") for ev in events)
    ip_counts = Counter(ev.get("sourceIPAddress", "") for ev in events)
    region_counts = Counter(ev.get("awsRegion", "") for ev in events)
    user_counts = Counter(
        ev.get("userIdentity", {}).get("arn", "") or
        ev.get("userIdentity", {}).get("userName", "")
        for ev in events
    )
    n = len(events)

    user_type_map = {
        "Root": 0, "IAMUser": 1, "AssumedRole": 2, "FederatedUser": 3,
        "AWSAccount": 4, "AWSService": 5,
    }

    features = []
    for ev in events:
        name = ev.get("eventName", "")
        source = ev.get("eventSource", "")
        ip = ev.get("sourceIPAddress", "")
        region = ev.get("awsRegion", "")
        user_arn = (
            ev.get("userIdentity", {}).get("arn", "") or
            ev.get("userIdentity", {}).get("userName", "")
        )
        user_type = ev.get("userIdentity", {}).get("type", "")
        error_code = ev.get("errorCode", "")
        read_only = ev.get("readOnly")

        dt = _parse_event_time(ev)
        hour = dt.hour if dt else 12
        dow = dt.weekday() if dt else 0

        features.append([
            1.0 - (name_counts[name] / n),       # rare event name → higher
            1.0 - (source_counts[source] / n),    # rare service → higher
            1.0 - (user_counts[user_arn] / n),    # rare user → higher
            _hash_encode(ip) / 64.0,              # IP diversity
            1.0 - (ip_counts[ip] / n),            # rare IP → higher
            hour / 23.0,                          # hour normalized
            dow / 6.0,                            # day of week normalized
            1.0 if error_code else 0.0,           # has error
            0.0 if read_only is True else 1.0,    # write event
            1.0 - (region_counts[region] / n),    # rare region → higher
            user_type_map.get(user_type, 6) / 6.0,  # user type ordinal
        ])

    return np.array(features)


def analyze_with_ml(events, contamination=0.05, top_n=20):
    """Run Isolation Forest on featurized CloudTrail events.

    Returns a list of anomaly dicts sorted by anomaly score (most anomalous first).
    """
    print(f"{CYAN}ML analysis: featurizing {len(events)} events...{RESET}")
    X = featurize_events(events)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"{CYAN}ML analysis: running Isolation Forest (contamination={contamination})...{RESET}")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # decision_function: lower = more anomalous (negative = outlier)
    scores = model.decision_function(X_scaled)
    labels = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

    # Collect anomalies
    anomalies = []
    feature_names = [
        "event_name_freq", "event_source_freq", "user_freq", "ip_hash",
        "ip_freq", "hour_of_day", "day_of_week", "has_error",
        "is_write_event", "region_freq", "user_type_code",
    ]
    for i, (label, score) in enumerate(zip(labels, scores)):
        if label == -1:
            ev = events[i]
            feat = X[i]
            # Identify which features contributed most (highest raw values)
            top_feats = sorted(
                zip(feature_names, feat), key=lambda x: x[1], reverse=True
            )[:3]
            reasons = [f"{fn}={fv:.2f}" for fn, fv in top_feats]

            anomalies.append({
                "event": ev,
                "score": float(score),
                "top_features": reasons,
            })

    anomalies.sort(key=lambda a: a["score"])
    anomalies = anomalies[:top_n]

    n_anomalies = int((labels == -1).sum())
    print(f"{CYAN}ML analysis: {n_anomalies} anomalies detected, showing top {len(anomalies)}.{RESET}\n")
    return anomalies


# ── AI analysis ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a cloud security analyst reviewing AWS CloudTrail logs.
Analyze the provided events for security anomalies and suspicious patterns.

For each finding, output a JSON object on its own line with these fields:
- "severity": one of CRITICAL, HIGH, MEDIUM, LOW, INFO
- "title": short summary (one sentence)
- "details": explanation of why this is suspicious and recommended actions
- "events": list of eventName values involved

Look for:
- Privilege escalation patterns (creating users/keys then attaching policies)
- Credential abuse (access from unusual IPs/regions, after-hours activity)
- Defense evasion (disabling logging, deleting trails, modifying configs)
- Lateral movement (assuming roles across accounts)
- Data exfiltration indicators (S3 bulk downloads, unusual API volume)
- Impossible travel (same user from geographically distant IPs in short time)

Output ONLY the JSON objects, one per line. If no anomalies are found, output:
{"severity": "INFO", "title": "No anomalies detected", "details": "Events appear normal.", "events": []}
"""


def build_event_summary(ev):
    """Compact single-event summary for the AI prompt."""
    return {
        "eventName": ev.get("eventName"),
        "eventSource": ev.get("eventSource"),
        "eventTime": ev.get("eventTime"),
        "awsRegion": ev.get("awsRegion"),
        "sourceIPAddress": ev.get("sourceIPAddress"),
        "userIdentity": {
            "type": ev.get("userIdentity", {}).get("type"),
            "arn": ev.get("userIdentity", {}).get("arn"),
            "userName": ev.get("userIdentity", {}).get("userName"),
        },
        "errorCode": ev.get("errorCode"),
        "errorMessage": ev.get("errorMessage"),
    }


def analyze_with_ai(events, batch_size=20):
    """Send event batches to local LM Studio model for anomaly analysis."""
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    findings = []

    batches = [events[i : i + batch_size] for i in range(0, len(events), batch_size)]
    total = len(batches)

    for idx, batch in enumerate(batches, 1):
        print(f"{CYAN}AI analysis: batch {idx}/{total} ({len(batch)} events)...{RESET}")
        summaries = [build_event_summary(ev) for ev in batch]

        try:
            response = client.chat.completions.create(
                model="qwen-qwq-32b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Analyze these CloudTrail events for security anomalies:\n\n"
                            + json.dumps(summaries, indent=2, default=str)
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            text = response.choices[0].message.content.strip()
            for line in text.splitlines():
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    finding = json.loads(line)
                    if "severity" in finding and "title" in finding:
                        findings.append(finding)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"{RED}AI analysis error: {e}{RESET}")
            print(f"{YELLOW}Ensure LM Studio is running on localhost:1234{RESET}")
            break

    return findings


# ── Report output ────────────────────────────────────────────────────────────


def severity_rank(sev):
    return {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}.get(sev, 5)


def print_report(flagged, stats, ml_anomalies, ai_findings):
    """Print color-coded terminal report."""
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  CloudTrail AI Anomaly Report{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    # Stats summary
    print(f"{BOLD}Event Summary:{RESET}")
    print(f"  Total events scanned:    {stats['total']}")
    print(f"  Access denied errors:    {stats['access_denied']}")
    print(f"  Failed console logins:   {stats['failed_logins']}")
    print(f"  IAM privilege actions:   {stats['iam_priv_actions']}")
    print(f"  Tampering actions:       {stats['tamper_actions']}")
    print(f"  Network/SG changes:      {stats['network_changes']}")
    print(f"  STS assume-role denied:  {stats['sts_denied']}")
    print()

    # Rule-based findings
    if flagged:
        print(f"{BOLD}── Rule-Based Findings ({len(flagged)}) ──{RESET}\n")
        for item in flagged:
            ev = item["event"]
            name = ev.get("eventName", "?")
            time = ev.get("eventTime", "?")
            user = (
                ev.get("userIdentity", {}).get("userName")
                or ev.get("userIdentity", {}).get("arn", "?")
            )
            ip = ev.get("sourceIPAddress", "?")
            for reason in item["reasons"]:
                color = YELLOW
                if "tampering" in reason.lower() or "privilege" in reason.lower():
                    color = RED
                print(f"  {color}[!] {reason}{RESET}")
                print(f"      Event: {name}  User: {user}  IP: {ip}  Time: {time}")
            print()
    else:
        print(f"{GREEN}No rule-based flags.{RESET}\n")

    # ML anomaly findings
    if ml_anomalies:
        print(f"{BOLD}── ML Anomaly Detection ({len(ml_anomalies)} top outliers) ──{RESET}\n")
        for a in ml_anomalies:
            ev = a["event"]
            score = a["score"]
            name = ev.get("eventName", "?")
            time = ev.get("eventTime", "?")
            user = (
                ev.get("userIdentity", {}).get("userName")
                or ev.get("userIdentity", {}).get("arn", "?")
            )
            ip = ev.get("sourceIPAddress", "?")
            source = ev.get("eventSource", "?")
            error = ev.get("errorCode", "")

            # Color by score severity
            if score < -0.3:
                color = RED
            elif score < -0.1:
                color = YELLOW
            else:
                color = CYAN

            print(f"  {color}[score: {score:.3f}] {name}{RESET}")
            print(f"      Service: {source}  User: {user}  IP: {ip}")
            print(f"      Time: {time}" + (f"  Error: {error}" if error else ""))
            print(f"      Top features: {', '.join(a['top_features'])}")
            print()
    elif ml_anomalies is not None:
        print(f"{GREEN}No ML anomalies detected.{RESET}\n")

    # AI findings
    if ai_findings:
        sorted_findings = sorted(ai_findings, key=lambda f: severity_rank(f.get("severity", "INFO")))
        print(f"{BOLD}── AI Analysis Findings ({len(ai_findings)}) ──{RESET}\n")
        for f in sorted_findings:
            sev = f.get("severity", "INFO")
            color = SEVERITY_COLORS.get(sev, DIM)
            print(f"  {color}[{sev}] {f.get('title', 'Unknown')}{RESET}")
            print(f"      {f.get('details', '')}")
            involved = f.get("events", [])
            if involved:
                print(f"      Events: {', '.join(str(e) for e in involved)}")
            print()
    else:
        print(f"{GREEN}No AI findings.{RESET}\n")

    print(f"{BOLD}{'=' * 70}{RESET}")


def export_json(flagged, stats, ml_anomalies, ai_findings, path):
    """Write machine-readable JSON report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "rule_based_findings": [
            {
                "eventName": item["event"].get("eventName"),
                "eventTime": str(item["event"].get("eventTime", "")),
                "sourceIPAddress": item["event"].get("sourceIPAddress"),
                "user": (
                    item["event"].get("userIdentity", {}).get("userName")
                    or item["event"].get("userIdentity", {}).get("arn")
                ),
                "reasons": item["reasons"],
            }
            for item in flagged
        ],
        "ml_anomalies": [
            {
                "eventName": a["event"].get("eventName"),
                "eventTime": str(a["event"].get("eventTime", "")),
                "eventSource": a["event"].get("eventSource"),
                "sourceIPAddress": a["event"].get("sourceIPAddress"),
                "user": (
                    a["event"].get("userIdentity", {}).get("userName")
                    or a["event"].get("userIdentity", {}).get("arn")
                ),
                "errorCode": a["event"].get("errorCode"),
                "anomaly_score": a["score"],
                "top_features": a["top_features"],
            }
            for a in (ml_anomalies or [])
        ],
        "ai_findings": ai_findings,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{GREEN}JSON report written to {path}{RESET}")


# ── Interactive S3 date prompts ──────────────────────────────────────────────

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def prompt_month():
    """Prompt user to select a month (1-12)."""
    print(f"\n{BOLD}Select month to retrieve:{RESET}")
    for i in range(1, 13):
        print(f"  {i:>2}) {MONTH_NAMES[i]}")
    print()
    while True:
        raw = input(f"{CYAN}Month (1-12): {RESET}").strip()
        try:
            m = int(raw)
            if 1 <= m <= 12:
                return m
        except ValueError:
            pass
        print(f"{YELLOW}Enter a number between 1 and 12.{RESET}")


def prompt_days(month):
    """Prompt user to select a day or day range within the chosen month."""
    print(f"\n{BOLD}Select day(s) within {MONTH_NAMES[month]}:{RESET}")
    print(f"  Enter a single day   (e.g. {CYAN}15{RESET})")
    print(f"  Enter a day range    (e.g. {CYAN}10-20{RESET})")
    print(f"  Press Enter for the  {CYAN}entire month{RESET}")
    print()
    while True:
        raw = input(f"{CYAN}Day(s) [Enter=all]: {RESET}").strip()
        if raw == "":
            return None, None  # whole month
        if "-" in raw:
            parts = raw.split("-", 1)
            try:
                s, e = int(parts[0]), int(parts[1])
                if 1 <= s <= 31 and 1 <= e <= 31 and s <= e:
                    return s, e
            except ValueError:
                pass
            print(f"{YELLOW}Enter a valid range like 10-20 (start <= end, 1-31).{RESET}")
        else:
            try:
                d = int(raw)
                if 1 <= d <= 31:
                    return d, d
            except ValueError:
                pass
            print(f"{YELLOW}Enter a day (1-31) or range (e.g. 10-20).{RESET}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="CloudTrail AI Anomaly Detector — analyze AWS CloudTrail logs for security anomalies"
    )
    parser.add_argument(
        "--s3-uri", default=None,
        help="S3 URI to CloudTrail year prefix (e.g. s3://bucket/AWSLogs/.../2025/). Overrides --hours/--region."
    )
    parser.add_argument(
        "--month", type=int, default=None, metavar="MM",
        help="Month to filter S3 logs (1-12). Use with --s3-uri. Prompted if omitted."
    )
    parser.add_argument(
        "--days", default=None, metavar="D or D-D",
        help="Day or day range for S3 logs (e.g. '15' or '10-20'). Prompted if omitted."
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="Lookback window in hours (default: 24)"
    )
    parser.add_argument(
        "--region", default="us-east-1", help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--profile", default=None, help="AWS CLI profile name"
    )
    parser.add_argument(
        "--max-events", type=int, default=200, help="Max events to retrieve (default: 200)"
    )
    parser.add_argument(
        "--output-json", metavar="FILE", default=None, help="Export report as JSON to FILE"
    )
    parser.add_argument(
        "--skip-ml", action="store_true", help="Skip ML anomaly detection"
    )
    parser.add_argument(
        "--ml-contamination", type=float, default=0.05, metavar="F",
        help="Expected anomaly fraction for Isolation Forest (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--ml-top-n", type=int, default=20, metavar="N",
        help="Number of top ML anomalies to display (default: 20)"
    )
    parser.add_argument(
        "--skip-ai", action="store_true", help="Skip LLM analysis"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Events per AI batch (default: 20)"
    )
    return parser.parse_args()


def parse_days_arg(days_str):
    """Parse --days value like '15' or '10-20' into (start, end)."""
    if "-" in days_str:
        parts = days_str.split("-", 1)
        return int(parts[0]), int(parts[1])
    d = int(days_str)
    return d, d


def main():
    args = parse_args()

    # Validate that --month/--days aren't used without --s3-uri
    if (args.month or args.days) and not args.s3_uri:
        print(f"{RED}--month/--days require --s3-uri{RESET}")
        sys.exit(1)

    month = args.month
    start_day, end_day = None, None

    if args.s3_uri:
        # Resolve month: from CLI or interactive prompt
        if month is None:
            month = prompt_month()

        # Resolve days: from CLI or interactive prompt
        if args.days is not None:
            start_day, end_day = parse_days_arg(args.days)
        else:
            start_day, end_day = prompt_days(month)

        label_parts = [MONTH_NAMES[month]]
        if start_day is not None:
            if start_day == end_day:
                label_parts.append(f"day {start_day}")
            else:
                label_parts.append(f"days {start_day}-{end_day}")
        print(f"\n{CYAN}Retrieving logs for: {', '.join(label_parts)}{RESET}")

    # 1. Fetch events
    try:
        if args.s3_uri:
            events = fetch_s3_events(args.s3_uri, args.profile, args.max_events,
                                     month=month, start_day=start_day,
                                     end_day=end_day)
        else:
            events = fetch_cloudtrail_events(args.region, args.profile, args.hours, args.max_events)
    except Exception as e:
        print(f"{RED}Failed to fetch CloudTrail events: {e}{RESET}")
        sys.exit(1)

    if not events:
        print(f"{YELLOW}No events found in the specified time window.{RESET}")
        sys.exit(0)

    # 2. Rule-based pre-filter
    flagged, stats = prefilter_events(events)

    # 3. ML anomaly detection (Isolation Forest)
    ml_anomalies = None
    if not args.skip_ml:
        ml_anomalies = analyze_with_ml(
            events, contamination=args.ml_contamination, top_n=args.ml_top_n
        )

    # 4. LLM analysis (optional — only on ML-flagged events if ML ran)
    ai_findings = []
    if not args.skip_ai:
        if ml_anomalies:
            # Send only anomalous events to LLM (saves GPU heat!)
            anomaly_events = [a["event"] for a in ml_anomalies]
            print(f"{CYAN}Sending {len(anomaly_events)} ML-flagged events to LLM (instead of all {len(events)})...{RESET}")
            ai_findings = analyze_with_ai(anomaly_events, batch_size=args.batch_size)
        else:
            ai_findings = analyze_with_ai(events, batch_size=args.batch_size)

    # 5. Report
    print_report(flagged, stats, ml_anomalies, ai_findings)

    if args.output_json:
        export_json(flagged, stats, ml_anomalies, ai_findings, args.output_json)


if __name__ == "__main__":
    main()
