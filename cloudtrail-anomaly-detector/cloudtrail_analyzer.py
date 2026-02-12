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
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def discover_trail_uri(region, profile, year=None):
    """Auto-discover CloudTrail S3 bucket and build the year-level S3 URI.

    Uses describe_trails() to find the bucket, sts to get the account ID,
    and S3 listing to detect org-trail path structure.

    Returns a fully-qualified S3 URI like:
      s3://bucket/AWSLogs/org-id/acct-id/CloudTrail/region/YYYY/
    """
    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(region_name=region, **session_kwargs)

    ct_client = session.client("cloudtrail")
    sts_client = session.client("sts")

    # Get account ID
    identity = sts_client.get_caller_identity()
    account_id = identity["Account"]
    print(f"{CYAN}Account: {account_id}{RESET}")

    # Get trails
    trails = ct_client.describe_trails().get("trailList", [])
    if not trails:
        print(f"{RED}No CloudTrail trails found in this account.{RESET}")
        return None

    # Pick the best trail: prefer multi-region or one homed in the requested region
    trail = None
    for t in trails:
        if t.get("IsMultiRegionTrail"):
            trail = t
            break
    if not trail:
        for t in trails:
            if t.get("HomeRegion") == region:
                trail = t
                break
    if not trail:
        trail = trails[0]

    bucket = trail["S3BucketName"]
    key_prefix = trail.get("S3KeyPrefix", "")
    trail_name = trail.get("Name", "?")
    is_org = trail.get("IsOrganizationTrail", False)

    print(f"{CYAN}Trail: {trail_name}  Bucket: {bucket}  Org trail: {is_org}{RESET}")

    # Build the base prefix: {key_prefix}/AWSLogs/
    base = (key_prefix + "/") if key_prefix else ""
    base += "AWSLogs/"

    # For org trails, detect the org ID by listing one level under AWSLogs/
    if is_org:
        s3 = session.client("s3")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=base, Delimiter="/", MaxKeys=10)
        org_id = None
        for cp in resp.get("CommonPrefixes", []):
            part = cp["Prefix"].replace(base, "").strip("/")
            if part.startswith("o-"):
                org_id = part
                break
        if org_id:
            base += f"{org_id}/{account_id}/CloudTrail/{region}/"
        else:
            print(f"{YELLOW}Org trail but couldn't detect org ID, trying non-org path.{RESET}")
            base += f"{account_id}/CloudTrail/{region}/"
    else:
        base += f"{account_id}/CloudTrail/{region}/"

    # Append year
    if year is None:
        year = datetime.now(timezone.utc).year
    base += f"{year}/"

    uri = f"s3://{bucket}/{base}"
    print(f"{CYAN}Discovered S3 URI: {uri}{RESET}")
    return uri


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


# ── Persistent baseline ──────────────────────────────────────────────────────

DEFAULT_BASELINE_PATH = os.path.join(Path.home(), ".cloudtrail_baseline.json")

EMPTY_BASELINE = {
    "version": 1,
    "total_events": 0,
    "runs": 0,
    "dates_processed": [],
    "last_updated": None,
    "event_names": {},       # name → count
    "event_sources": {},     # service → count
    "users": {},             # arn → {"count": N, "first_seen": date, "last_seen": date}
    "ips": {},               # ip → {"count": N, "first_seen": date, "last_seen": date}
    "regions": {},           # region → count
    "user_agents": {},       # truncated UA → count
    "hour_by_user": {},      # arn → {"0": N, "1": N, ...}
}


def load_baseline(path):
    """Load baseline from JSON file, or return empty baseline."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            bl = json.load(f)
        print(f"{CYAN}Loaded baseline: {bl['total_events']} events across "
              f"{bl['runs']} run(s), last updated {bl['last_updated']}{RESET}")
        return bl
    except (json.JSONDecodeError, KeyError) as e:
        print(f"{YELLOW}Baseline file corrupt ({e}), starting fresh.{RESET}")
        return None


def save_baseline(bl, path):
    """Write baseline to JSON file."""
    with open(path, "w") as f:
        json.dump(bl, f, indent=2, default=str)
    print(f"{GREEN}Baseline saved: {bl['total_events']} total events, "
          f"{bl['runs']} run(s) → {path}{RESET}")


def _event_user(ev):
    """Extract a consistent user identifier from a CloudTrail event.

    For assumed-role ARNs, strips the ephemeral session name so that the same
    role with different session IDs (e.g. SecurityHub UUIDs) is treated as one
    identity:
      arn:aws:sts::123:assumed-role/MyRole/session-id → arn:aws:sts::123:assumed-role/MyRole
    """
    uid = ev.get("userIdentity", {})
    arn = uid.get("arn", "")
    if arn and ":assumed-role/" in arn:
        # arn:aws:sts::ACCT:assumed-role/ROLE_NAME/SESSION_NAME → drop /SESSION_NAME
        parts = arn.split("/")
        if len(parts) >= 3:
            return "/".join(parts[:2])  # keep arn...assumed-role/ROLE_NAME
    return arn or uid.get("userName", "") or ""


def _event_date_str(ev):
    """Extract date string (YYYY-MM-DD) from an event."""
    dt = _parse_event_time(ev)
    if dt:
        return dt.strftime("%Y-%m-%d")
    return ""


def update_baseline(bl, events, run_label=""):
    """Merge current events into the baseline and return it.

    Tracks:
      - Cumulative frequency counters for event names, sources, regions
      - Per-user and per-IP: count, first_seen, last_seen dates
      - Per-user hour-of-day distribution
    """
    if bl is None:
        bl = json.loads(json.dumps(EMPTY_BASELINE))  # deep copy

    bl["runs"] += 1
    bl["last_updated"] = datetime.now(timezone.utc).isoformat()
    if run_label:
        bl["dates_processed"].append(run_label)

    for ev in events:
        bl["total_events"] += 1

        name = ev.get("eventName", "")
        source = ev.get("eventSource", "")
        ip = ev.get("sourceIPAddress", "")
        region = ev.get("awsRegion", "")
        user = _event_user(ev)
        date_str = _event_date_str(ev)

        # Simple counters
        bl["event_names"][name] = bl["event_names"].get(name, 0) + 1
        bl["event_sources"][source] = bl["event_sources"].get(source, 0) + 1
        bl["regions"][region] = bl["regions"].get(region, 0) + 1

        # User tracking with first/last seen
        if user:
            if user not in bl["users"]:
                bl["users"][user] = {"count": 0, "first_seen": date_str, "last_seen": date_str}
            bl["users"][user]["count"] += 1
            if date_str and date_str < bl["users"][user]["first_seen"]:
                bl["users"][user]["first_seen"] = date_str
            if date_str and date_str > bl["users"][user]["last_seen"]:
                bl["users"][user]["last_seen"] = date_str

        # IP tracking with first/last seen
        if ip:
            if ip not in bl["ips"]:
                bl["ips"][ip] = {"count": 0, "first_seen": date_str, "last_seen": date_str}
            bl["ips"][ip]["count"] += 1
            if date_str and date_str < bl["ips"][ip]["first_seen"]:
                bl["ips"][ip]["first_seen"] = date_str
            if date_str and date_str > bl["ips"][ip]["last_seen"]:
                bl["ips"][ip]["last_seen"] = date_str

        # Hour-of-day per user
        if user:
            dt = _parse_event_time(ev)
            if dt:
                h = str(dt.hour)
                if user not in bl["hour_by_user"]:
                    bl["hour_by_user"][user] = {}
                bl["hour_by_user"][user][h] = bl["hour_by_user"][user].get(h, 0) + 1

    return bl


def detect_new_activity(events, baseline):
    """Compare current events against baseline to find never-before-seen activity.

    Returns a list of findings with category, value, and the triggering event.
    """
    if baseline is None:
        return []

    findings = []
    # Track what we've already reported to avoid duplicates
    seen_new_users = set()
    seen_new_ips = set()
    seen_new_events = set()
    seen_new_sources = set()
    seen_unusual_hours = set()

    bl_users = baseline.get("users", {})
    bl_ips = baseline.get("ips", {})
    bl_names = baseline.get("event_names", {})
    bl_sources = baseline.get("event_sources", {})
    bl_hours = baseline.get("hour_by_user", {})

    for ev in events:
        user = _event_user(ev)
        ip = ev.get("sourceIPAddress", "")
        name = ev.get("eventName", "")
        source = ev.get("eventSource", "")

        # New user (never seen in baseline)
        if user and user not in bl_users and user not in seen_new_users:
            seen_new_users.add(user)
            findings.append({
                "category": "NEW_USER",
                "severity": "HIGH",
                "value": user,
                "detail": "User/role never seen in baseline history",
                "event": ev,
            })

        # New source IP (never seen in baseline)
        if ip and ip not in bl_ips and ip not in seen_new_ips:
            # Skip internal AWS service IPs
            if not ip.endswith(".amazonaws.com"):
                seen_new_ips.add(ip)
                findings.append({
                    "category": "NEW_IP",
                    "severity": "MEDIUM",
                    "value": ip,
                    "detail": "Source IP never seen in baseline history",
                    "event": ev,
                })

        # New API call (never seen in baseline)
        if name and name not in bl_names and name not in seen_new_events:
            seen_new_events.add(name)
            findings.append({
                "category": "NEW_EVENT",
                "severity": "MEDIUM",
                "value": name,
                "detail": "API call never seen in baseline history",
                "event": ev,
            })

        # New service (never seen in baseline)
        if source and source not in bl_sources and source not in seen_new_sources:
            seen_new_sources.add(source)
            findings.append({
                "category": "NEW_SERVICE",
                "severity": "MEDIUM",
                "value": source,
                "detail": "AWS service never seen in baseline history",
                "event": ev,
            })

        # Unusual hour for known user
        if user and user in bl_hours and user not in seen_unusual_hours:
            dt = _parse_event_time(ev)
            if dt:
                h = str(dt.hour)
                user_hour_dist = bl_hours[user]
                total_user_events = sum(user_hour_dist.values())
                hour_count = user_hour_dist.get(h, 0)
                # Flag if this hour has < 1% of the user's historical activity
                if total_user_events >= 50 and hour_count / total_user_events < 0.01:
                    seen_unusual_hours.add(user)
                    findings.append({
                        "category": "UNUSUAL_HOUR",
                        "severity": "LOW",
                        "value": f"{user} at hour {dt.hour}:00 UTC",
                        "detail": (f"User has {hour_count}/{total_user_events} historical "
                                   f"events at this hour ({hour_count/total_user_events*100:.1f}%)"),
                        "event": ev,
                    })

    return findings


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


def featurize_events(events, baseline=None):
    """Convert CloudTrail events into numeric feature vectors.

    When a baseline is provided, frequency features are computed against the
    historical totals (baseline + current batch) instead of just the current
    batch. This means a user seen 10,000 times in the baseline won't be
    flagged as "rare" even if they only appear once today.

    Features per event:
      0  event_name_freq      — how rare is this API call
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
    # Build frequency tables: baseline + current batch combined
    batch_name_counts = Counter(ev.get("eventName", "") for ev in events)
    batch_source_counts = Counter(ev.get("eventSource", "") for ev in events)
    batch_ip_counts = Counter(ev.get("sourceIPAddress", "") for ev in events)
    batch_region_counts = Counter(ev.get("awsRegion", "") for ev in events)
    batch_user_counts = Counter(_event_user(ev) for ev in events)

    if baseline:
        # Merge baseline counters with current batch
        bl_total = baseline.get("total_events", 0)
        name_counts = Counter(baseline.get("event_names", {}))
        name_counts.update(batch_name_counts)
        source_counts = Counter(baseline.get("event_sources", {}))
        source_counts.update(batch_source_counts)
        ip_counts = Counter({k: v["count"] for k, v in baseline.get("ips", {}).items()})
        ip_counts.update(batch_ip_counts)
        region_counts = Counter(baseline.get("regions", {}))
        region_counts.update(batch_region_counts)
        user_counts = Counter({k: v["count"] for k, v in baseline.get("users", {}).items()})
        user_counts.update(batch_user_counts)
        n = bl_total + len(events)
        print(f"{CYAN}ML featurization using baseline context: {bl_total} historical + {len(events)} current = {n} total{RESET}")
    else:
        name_counts = batch_name_counts
        source_counts = batch_source_counts
        ip_counts = batch_ip_counts
        region_counts = batch_region_counts
        user_counts = batch_user_counts
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
        user_arn = _event_user(ev)
        user_type = ev.get("userIdentity", {}).get("type", "")
        error_code = ev.get("errorCode", "")
        read_only = ev.get("readOnly")

        dt = _parse_event_time(ev)
        hour = dt.hour if dt else 12
        dow = dt.weekday() if dt else 0

        features.append([
            1.0 - (name_counts.get(name, 0) / n),     # rare event name → higher
            1.0 - (source_counts.get(source, 0) / n),  # rare service → higher
            1.0 - (user_counts.get(user_arn, 0) / n),  # rare user → higher
            _hash_encode(ip) / 64.0,                    # IP diversity
            1.0 - (ip_counts.get(ip, 0) / n),          # rare IP → higher
            hour / 23.0,                                # hour normalized
            dow / 6.0,                                  # day of week normalized
            1.0 if error_code else 0.0,                 # has error
            0.0 if read_only is True else 1.0,          # write event
            1.0 - (region_counts.get(region, 0) / n),  # rare region → higher
            user_type_map.get(user_type, 6) / 6.0,     # user type ordinal
        ])

    return np.array(features)


def analyze_with_ml(events, baseline=None, contamination=0.05, top_n=20):
    """Run Isolation Forest on featurized CloudTrail events.

    Returns a list of anomaly dicts sorted by anomaly score (most anomalous first).
    """
    print(f"{CYAN}ML analysis: featurizing {len(events)} events...{RESET}")
    X = featurize_events(events, baseline=baseline)

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


def print_report(flagged, stats, new_activity, ml_anomalies, ai_findings):
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

    # New activity (baseline comparison)
    if new_activity:
        cat_colors = {
            "NEW_USER": RED, "NEW_IP": YELLOW, "NEW_EVENT": YELLOW,
            "NEW_SERVICE": YELLOW, "UNUSUAL_HOUR": CYAN,
        }
        print(f"{BOLD}── New Activity vs. Baseline ({len(new_activity)}) ──{RESET}\n")
        for fa in new_activity:
            ev = fa["event"]
            cat = fa["category"]
            color = cat_colors.get(cat, YELLOW)
            time = ev.get("eventTime", "?")
            print(f"  {color}[{cat}] {fa['value']}{RESET}")
            print(f"      {fa['detail']}")
            print(f"      Event: {ev.get('eventName', '?')}  Time: {time}")
            print()
    elif new_activity is not None and len(new_activity) == 0:
        print(f"{GREEN}No new activity vs. baseline.{RESET}\n")

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


def export_json(flagged, stats, new_activity, ml_anomalies, ai_findings, path):
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
        "new_activity": [
            {
                "category": fa["category"],
                "severity": fa["severity"],
                "value": fa["value"],
                "detail": fa["detail"],
                "eventName": fa["event"].get("eventName"),
                "eventTime": str(fa["event"].get("eventTime", "")),
            }
            for fa in (new_activity or [])
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
        "--s3-logs", action="store_true",
        help="Auto-discover CloudTrail S3 bucket and analyze historical logs (no URI needed)"
    )
    parser.add_argument(
        "--s3-uri", default=None,
        help="S3 URI to CloudTrail year prefix (e.g. s3://bucket/AWSLogs/.../2025/). Overrides --hours/--region."
    )
    parser.add_argument(
        "--year", type=int, default=None, metavar="YYYY",
        help="Year for S3 logs (default: current year). Use with --s3-logs or --s3-uri."
    )
    parser.add_argument(
        "--month", type=int, default=None, metavar="MM",
        help="Month to filter S3 logs (1-12). Use with --s3-uri/--s3-logs. Prompted if omitted."
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
        "--baseline-file", default=DEFAULT_BASELINE_PATH, metavar="FILE",
        help=f"Path to baseline JSON file (default: {DEFAULT_BASELINE_PATH})"
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Disable baseline load/save (single-run mode, no historical context)"
    )
    parser.add_argument(
        "--reset-baseline", action="store_true",
        help="Delete existing baseline and start fresh from this run"
    )
    parser.add_argument(
        "--yesterday", action="store_true",
        help="Convenience: analyze yesterday's S3 logs (requires --s3-uri or --s3-logs)"
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

    # Auto-discover S3 URI if --s3-logs is set
    if args.s3_logs and not args.s3_uri:
        print(f"{CYAN}Auto-discovering CloudTrail S3 configuration...{RESET}")
        args.s3_uri = discover_trail_uri(args.region, args.profile, year=args.year)
        if not args.s3_uri:
            print(f"{RED}Could not auto-discover CloudTrail S3 bucket. Use --s3-uri instead.{RESET}")
            sys.exit(1)

    # If --year provided with existing --s3-uri, swap the year in the URI
    if args.year and args.s3_uri and not args.s3_logs:
        # Replace trailing YYYY/ with the requested year
        import re
        args.s3_uri = re.sub(r'/\d{4}/$', f'/{args.year}/', args.s3_uri)

    # Validate that --month/--days aren't used without --s3-uri/--s3-logs
    if (args.month or args.days) and not args.s3_uri:
        print(f"{RED}--month/--days require --s3-uri or --s3-logs{RESET}")
        sys.exit(1)
    if args.yesterday and not args.s3_uri:
        print(f"{RED}--yesterday requires --s3-uri or --s3-logs{RESET}")
        sys.exit(1)

    month = args.month
    start_day, end_day = None, None

    # --yesterday: auto-set month/day to yesterday's date
    if args.yesterday:
        yd = datetime.now(timezone.utc) - timedelta(days=1)
        month = yd.month
        start_day = end_day = yd.day
        print(f"{CYAN}--yesterday: targeting {yd.strftime('%Y-%m-%d')}{RESET}")

    if args.s3_uri and not args.yesterday:
        # Resolve month: from CLI or interactive prompt
        if month is None:
            month = prompt_month()

        # Resolve days: from CLI or interactive prompt
        if args.days is not None:
            start_day, end_day = parse_days_arg(args.days)
        else:
            start_day, end_day = prompt_days(month)

    if args.s3_uri:
        label_parts = [MONTH_NAMES[month]]
        if start_day is not None:
            if start_day == end_day:
                label_parts.append(f"day {start_day}")
            else:
                label_parts.append(f"days {start_day}-{end_day}")
        print(f"\n{CYAN}Retrieving logs for: {', '.join(label_parts)}{RESET}")

    # 0. Load baseline
    baseline = None
    if not args.no_baseline:
        if args.reset_baseline and os.path.exists(args.baseline_file):
            os.remove(args.baseline_file)
            print(f"{YELLOW}Baseline reset: {args.baseline_file}{RESET}")
        baseline = load_baseline(args.baseline_file)

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

    # 3. New-activity detection (compare against baseline)
    new_activity = []
    if baseline is not None:
        print(f"{CYAN}Comparing events against baseline...{RESET}")
        new_activity = detect_new_activity(events, baseline)
        if new_activity:
            print(f"{CYAN}Found {len(new_activity)} new-activity findings.{RESET}\n")
        else:
            print(f"{CYAN}No new activity vs. baseline.{RESET}\n")

    # 4. ML anomaly detection (Isolation Forest with baseline context)
    ml_anomalies = None
    if not args.skip_ml:
        ml_anomalies = analyze_with_ml(
            events, baseline=baseline,
            contamination=args.ml_contamination, top_n=args.ml_top_n
        )

    # 5. LLM analysis (optional — only on ML-flagged events if ML ran)
    ai_findings = []
    if not args.skip_ai:
        if ml_anomalies:
            anomaly_events = [a["event"] for a in ml_anomalies]
            print(f"{CYAN}Sending {len(anomaly_events)} ML-flagged events to LLM (instead of all {len(events)})...{RESET}")
            ai_findings = analyze_with_ai(anomaly_events, batch_size=args.batch_size)
        else:
            ai_findings = analyze_with_ai(events, batch_size=args.batch_size)

    # 6. Report
    print_report(flagged, stats, new_activity, ml_anomalies, ai_findings)

    if args.output_json:
        export_json(flagged, stats, new_activity, ml_anomalies, ai_findings, args.output_json)

    # 7. Update and save baseline
    if not args.no_baseline:
        run_label = ""
        if args.s3_uri and month:
            parts = [f"month={month}"]
            if start_day:
                parts.append(f"days={start_day}-{end_day}" if end_day != start_day else f"day={start_day}")
            run_label = ", ".join(parts)
        elif args.hours:
            run_label = f"last {args.hours}h"
        baseline = update_baseline(baseline, events, run_label=run_label)
        save_baseline(baseline, args.baseline_file)


if __name__ == "__main__":
    main()
