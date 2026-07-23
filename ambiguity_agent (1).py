"""
LangGraph Jira Ambiguity Scanner Agent
---------------------------------------
Watches Jira for stories labeled "ready-for-scan", runs a 5-dimension
ambiguity check via Claude, and posts structured findings back as a comment.

Setup:
  pip install langgraph langchain-anthropic jira python-dotenv

.env file:
  JIRA_URL=https://yourorg.atlassian.net
  JIRA_EMAIL=you@yourorg.com
  JIRA_API_TOKEN=your_token_here
  ANTHROPIC_API_KEY=your_key_here
  JIRA_PROJECT_KEY=ENG          # your project key
  POLL_INTERVAL_SECONDS=60
"""

import json
import os
import time
from typing import TypedDict, Optional

from dotenv import load_dotenv
from jira import JIRA
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

load_dotenv()

# ── Jira client ──────────────────────────────────────────────────────────────

jira = JIRA(
    server=os.environ["JIRA_URL"],
    basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"]),
)

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)

# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    issue_key: str
    story_text: str          # full story: summary + description + AC
    findings: Optional[dict] # parsed JSON from LLM
    status: str              # "pending" | "clean" | "flagged"


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior QA requirements analyst. Your job is to scan
user stories for ambiguity before sprint planning. Be precise and concise.
Always respond with valid JSON only — no preamble, no markdown fences."""

SCAN_PROMPT = """Analyze the following user story across exactly 5 dimensions.
For each dimension, return a list of findings. Each finding must have:
  - "issue": one sentence describing the specific problem
  - "severity": "high" | "medium" | "low"
  - "suggestion": one sentence on how to fix it

Return ONLY this JSON structure:
{{
  "unclear_nouns": [...],
  "missing_ac": [...],
  "open_business_rules": [...],
  "edge_cases": [...],
  "testability_gaps": [...]
}}

If a dimension has no issues, return an empty list for it.

USER STORY:
{story_text}"""


# ── Node 1: jira_watcher ──────────────────────────────────────────────────────

def jira_watcher(state: AgentState) -> AgentState:
    """
    Called by the polling loop (below) with a pre-fetched issue.
    Extracts the full story text from the Jira issue.
    """
    issue = jira.issue(state["issue_key"])

    summary     = issue.fields.summary or ""
    description = issue.fields.description or ""

    # Grab acceptance criteria from a custom field if present,
    # otherwise pull from description
    ac = ""
    if hasattr(issue.fields, "customfield_10016"):      # adjust field ID for your Jira
        ac = issue.fields.customfield_10016 or ""

    story_text = f"SUMMARY:\n{summary}\n\nDESCRIPTION:\n{description}"
    if ac:
        story_text += f"\n\nACCEPTANCE CRITERIA:\n{ac}"

    return {**state, "story_text": story_text, "status": "pending"}


# ── Node 2: ambiguity_scanner ─────────────────────────────────────────────────

def ambiguity_scanner(state: AgentState) -> AgentState:
    """Calls Claude with the story text, parses the 5-dimension JSON."""
    prompt = SCAN_PROMPT.format(story_text=state["story_text"])
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ])

    raw = response.content.strip()

    # Defensive: strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    findings = json.loads(raw)
    return {**state, "findings": findings}


# ── Routing function ───────────────────────────────────────────────────────────

def route_findings(state: AgentState) -> str:
    """Send to jira_writer if any HIGH finding exists, else END cleanly."""
    findings = state.get("findings", {})
    for dimension_findings in findings.values():
        for f in dimension_findings:
            if f.get("severity") == "high":
                return "jira_writer"
    return "end_clean"


# ── Node 3: jira_writer ───────────────────────────────────────────────────────

DIMENSION_LABELS = {
    "unclear_nouns":        "🔍 Unclear Nouns",
    "missing_ac":           "📋 Missing Acceptance Criteria",
    "open_business_rules":  "⚖️  Open Business Rules",
    "edge_cases":           "⚠️  Edge Cases",
    "testability_gaps":     "🧪 Testability Gaps",
}

SEVERITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def jira_writer(state: AgentState) -> AgentState:
    """Formats findings as a Jira comment and adds the 'ambiguity-flagged' label."""
    findings = state["findings"]
    lines = ["*AI Ambiguity Scan Results*\n----"]

    total = 0
    for dim_key, label in DIMENSION_LABELS.items():
        dim_findings = findings.get(dim_key, [])
        if not dim_findings:
            continue
        lines.append(f"\n*{label}*")
        for f in dim_findings:
            sev   = f.get("severity", "low")
            icon  = SEVERITY_EMOJI.get(sev, "⚪")
            issue = f.get("issue", "")
            suggestion = f.get("suggestion", "")
            lines.append(f"{icon} *{sev.upper()}* — {issue}")
            if suggestion:
                lines.append(f"   → _{suggestion}_")
            total += 1

    lines.append(f"\n_{total} finding(s) detected. Please resolve HIGH items before sprint entry._")
    comment_body = "\n".join(lines)

    # Post comment
    jira.add_comment(state["issue_key"], comment_body)

    # Add label so the team can filter
    issue = jira.issue(state["issue_key"])
    existing_labels = list(issue.fields.labels or [])
    if "ambiguity-flagged" not in existing_labels:
        existing_labels.append("ambiguity-flagged")
        issue.update(fields={"labels": existing_labels})

    # Remove the trigger label so it doesn't re-scan next poll
    if "ready-for-scan" in existing_labels:
        existing_labels.remove("ready-for-scan")
        issue.update(fields={"labels": existing_labels})

    print(f"[{state['issue_key']}] Flagged — {total} findings posted.")
    return {**state, "status": "flagged"}


def end_clean(state: AgentState) -> AgentState:
    """No high findings — mark the issue clean and remove the trigger label."""
    issue = jira.issue(state["issue_key"])
    existing_labels = list(issue.fields.labels or [])

    if "ready-for-scan" in existing_labels:
        existing_labels.remove("ready-for-scan")
    if "ambiguity-clean" not in existing_labels:
        existing_labels.append("ambiguity-clean")
    issue.update(fields={"labels": existing_labels})

    jira.add_comment(
        state["issue_key"],
        "_AI Ambiguity Scan: No high-severity issues found. Story cleared for sprint entry._"
    )
    print(f"[{state['issue_key']}] Clean — no high findings.")
    return {**state, "status": "clean"}


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("jira_watcher",      jira_watcher)
    graph.add_node("ambiguity_scanner", ambiguity_scanner)
    graph.add_node("jira_writer",       jira_writer)
    graph.add_node("end_clean",         end_clean)

    graph.set_entry_point("jira_watcher")

    graph.add_edge("jira_watcher",      "ambiguity_scanner")
    graph.add_conditional_edges(
        "ambiguity_scanner",
        route_findings,
        {
            "jira_writer": "jira_writer",
            "end_clean":   "end_clean",
        }
    )
    graph.add_edge("jira_writer", END)
    graph.add_edge("end_clean",   END)

    return graph.compile()


# ── Polling loop ───────────────────────────────────────────────────────────────

def fetch_pending_issues():
    """JQL: issues in your project labeled ready-for-scan, not yet scanned."""
    project = os.environ.get("JIRA_PROJECT_KEY", "ENG")
    jql = (
        f'project = "{project}" '
        f'AND labels = "ready-for-scan" '
        f'AND labels != "ambiguity-flagged" '
        f'AND labels != "ambiguity-clean" '
        f'ORDER BY created DESC'
    )
    return jira.search_issues(jql, maxResults=10)


def run_polling_loop():
    """Main entry point. Polls Jira every POLL_INTERVAL_SECONDS."""
    agent = build_graph()
    interval = int(os.environ.get("POLL_INTERVAL_SECONDS", 60))
    print(f"Agent started. Polling every {interval}s …")

    while True:
        try:
            issues = fetch_pending_issues()
            if not issues:
                print("No pending issues.")
            for issue in issues:
                print(f"Processing {issue.key} …")
                initial_state: AgentState = {
                    "issue_key":  issue.key,
                    "story_text": "",
                    "findings":   None,
                    "status":     "pending",
                }
                agent.invoke(initial_state)
        except Exception as e:
            print(f"Error in polling loop: {e}")

        time.sleep(interval)


# ── One-shot mode (for testing a single issue) ─────────────────────────────────

def scan_one(issue_key: str):
    """Quick test: scan a single issue by key."""
    agent = build_graph()
    result = agent.invoke({
        "issue_key":  issue_key,
        "story_text": "",
        "findings":   None,
        "status":     "pending",
    })
    print(json.dumps(result["findings"], indent=2))
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # python ambiguity_agent.py ENG-42
        scan_one(sys.argv[1])
    else:
        run_polling_loop()
....
