#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Applies a triage suggestion produced by the (untrusted-input) `analyze` job
# to a GitHub issue. This script is the ONLY component in the triage pipeline
# with `issues: write` permission, and it is plain, deterministic code — not
# an LLM — so a prompt-injection payload in the original issue body cannot
# reach it except as inert data inside a JSON field, which is validated
# against strict allowlists/limits below before any `gh` write command runs.
#
# Required env vars: GH_TOKEN, GITHUB_REPOSITORY, ISSUE_NUMBER, SUGGESTION

set -euo pipefail

: "${GITHUB_REPOSITORY:?}"
: "${ISSUE_NUMBER:?}"
SUGGESTION="${SUGGESTION:-}"

MAX_LABELS=3
MAX_COMMENT_CHARS=2000

if [[ -z "$SUGGESTION" ]]; then
  echo "No suggestion produced by analyze job; nothing to apply."
  exit 0
fi

if ! echo "$SUGGESTION" | jq empty >/dev/null 2>&1; then
  # Don't echo raw SUGGESTION into a workflow command: it's attacker-
  # influenced and can contain newlines / `::...::` sequences, enabling
  # workflow-command injection (fake annotations, stop-commands, etc.).
  echo "::warning::Triage suggestion is not valid JSON, discarding output from analyze job."
  exit 0
fi

# --- Validate labels against the repository's real label set -------------
VALID_LABELS=()
while IFS= read -r label; do
  [[ -n "$label" ]] && VALID_LABELS+=("$label")
done < <(gh label list --repo "$GITHUB_REPOSITORY" --limit 100 --json name --jq '.[].name')

SUGGESTED_LABELS=()
while IFS= read -r label; do
  [[ -n "$label" ]] && SUGGESTED_LABELS+=("$label")
done < <(echo "$SUGGESTION" | jq -r 'try (.labels[]) catch empty' | head -n "$MAX_LABELS")

APPLY_LABELS=()
for label in "${SUGGESTED_LABELS[@]+"${SUGGESTED_LABELS[@]}"}"; do
  for valid in "${VALID_LABELS[@]+"${VALID_LABELS[@]}"}"; do
    if [[ "$label" == "$valid" ]]; then
      APPLY_LABELS+=("$label")
      break
    fi
  done
done

if [[ ${#APPLY_LABELS[@]} -gt 0 ]]; then
  JOINED_LABELS=$(printf ',%s' "${APPLY_LABELS[@]}")
  JOINED_LABELS="${JOINED_LABELS#,}"
  echo "Applying labels: $JOINED_LABELS"
  gh issue edit "$ISSUE_NUMBER" --repo "$GITHUB_REPOSITORY" --add-label "$JOINED_LABELS"
else
  echo "No valid labels to apply."
fi

# --- Validate and post the comment, if any --------------------------------
NEEDS_MORE_INFO=$(echo "$SUGGESTION" | jq -r '.needs_more_info // false')
DUPLICATE_OF=$(echo "$SUGGESTION" | jq -r '.duplicate_of // empty')
TEMPLATE_MISMATCH=$(echo "$SUGGESTION" | jq -r '.template_mismatch // empty')
COMMENT=$(echo "$SUGGESTION" | jq -r '.comment // empty')

# Only post a comment when a concrete, expected reason is present. This
# prevents an injected "comment" field from being posted for arbitrary
# reasons unrelated to the actual triage signals.
ACTIONABLE=false
if [[ "$NEEDS_MORE_INFO" == "true" ]]; then
  ACTIONABLE=true
fi
if [[ -n "$DUPLICATE_OF" && "$DUPLICATE_OF" != "null" ]]; then
  # duplicate_of must be a plain integer issue number.
  if [[ "$DUPLICATE_OF" =~ ^[0-9]+$ ]]; then
    ACTIONABLE=true
  else
    # Same injection concern as above: don't echo the raw value.
    echo "::warning::Ignoring non-numeric duplicate_of value from suggestion."
  fi
fi
# template_mismatch is free-form text from the model and must NOT authorize
# a comment on its own — an injected suggestion could set any non-null
# string here to force ACTIONABLE=true and post an arbitrary comment. Only
# allow it to gate a comment when it names one of the known, fixed type
# labels the agent is allowed to classify issues with (see Step 3 of the
# agent prompt); anything else is ignored for gating purposes.
case "$TEMPLATE_MISMATCH" in
  "🐞bug" | "Feature Request" | "Enhancement" | "Question" | "Documentation" | "Refactor")
    ACTIONABLE=true
    ;;
esac

if [[ "$ACTIONABLE" == "true" && -n "$COMMENT" && "$COMMENT" != "null" ]]; then
  TRIMMED_COMMENT="${COMMENT:0:$MAX_COMMENT_CHARS}"
  COMMENT_FILE="$(mktemp)"
  printf '%s\n' "$TRIMMED_COMMENT" > "$COMMENT_FILE"
  echo "Posting comment on issue #$ISSUE_NUMBER."
  gh issue comment "$ISSUE_NUMBER" --repo "$GITHUB_REPOSITORY" --body-file "$COMMENT_FILE"
  rm -f "$COMMENT_FILE"
else
  echo "No actionable comment to post."
fi

# NOTE: this script intentionally never closes, locks, or assigns the issue.
