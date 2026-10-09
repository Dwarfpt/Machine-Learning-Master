"""Generates n8n/nrg_workflow.json (imported by n8n/setup_n8n.sh, or manually: Workflows -> Import from file).

Flow 1  (POST /webhook/nrg-action, sent by the AI agent for every created action)
    Webhook -> Log to Google Sheets -> Needs approval?
        yes -> E-mail manager with Approve / Reject links          (Human-in-the-loop)
        no  -> Mark executed in API -> E-mail team "task created"
Flow 2  (POST /webhook/nrg-decision, sent by the API after the manager clicked a link / button)
    Webhook -> Log decision to Google Sheets -> Approved?
        yes -> Mark executed in API -> E-mail team "approved task"
        no  -> E-mail team "rejected, nothing to do"

E-mail addresses come from n8n environment variables (set in .env, passed by docker-compose):
    NRG_MAIL_FROM, NRG_MANAGER_EMAIL, NRG_TEAM_EMAIL
Google Sheets logging is enabled when NRG_GSHEET_ID is set at build time (see n8n/setup_google.py); it uses a
Google service account credential and never blocks the flow (onError: continue).
"""
import json
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _dotenv(key: str) -> str:
    if os.getenv(key):
        return os.environ[key]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


SHEET_ID = _dotenv("NRG_GSHEET_ID")
API = "={{ $env.NRG_API_URL || 'http://host.docker.internal:8000' }}"
A = "$('Webhook: action created').item.json.body"
D = "$('Webhook: decision').item.json.body"
MAIL_FROM = "={{ $env.NRG_MAIL_FROM || 'nrg-bot@example.com' }}"
MANAGER = "={{ $env.NRG_MANAGER_EMAIL || 'manager@example.com' }}"
TEAM = "={{ $env.NRG_TEAM_EMAIL || 'dev-team@example.com' }}"


def node(name, type_, version, pos, params, **extra):
    n = {"parameters": params, "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, name)), "name": name, "type": type_,
         "typeVersion": version, "position": pos}
    n.update(extra)
    return n


def webhook(name, path, pos):
    return node(name, "n8n-nodes-base.webhook", 2, pos, {"httpMethod": "POST", "path": path, "options": {}},
                webhookId=str(uuid.uuid5(uuid.NAMESPACE_URL, path)))


def if_true(name, expr, pos):
    return node(name, "n8n-nodes-base.if", 2, pos, {
        "conditions": {"options": {"caseSensitive": True, "leftValue": "", "typeValidation": "loose"},
                       "conditions": [{"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, name + "c")), "leftValue": expr,
                                       "rightValue": "", "operator": {"type": "boolean", "operation": "true",
                                                                      "singleValue": True}}],
                       "combinator": "and"}, "options": {}})


def execute(name, src, pos):
    url = (API[:-3] + " }}/actions/{{ " + src + ".action_id }}/execute?token={{ " + src + ".token }}")
    return node(name, "n8n-nodes-base.httpRequest", 4.2, pos, {"method": "POST", "url": url, "options": {}})


def email(name, to, subject, html, pos):
    return node(name, "n8n-nodes-base.emailSend", 2.1, pos, {
        "fromEmail": MAIL_FROM, "toEmail": to, "subject": subject, "emailFormat": "html",
        "html": html, "options": {}}, credentials={"smtp": {"id": "nrgSmtpMailpit01", "name": "SMTP (Mailpit)"}})


SHEET_COLS = ["event", "action_id", "created_at", "title", "action_code", "cost_k", "expected_benefit_k", "status",
              "decided_by", "proba_negative", "risk_level"]


def sheets_log(name, src, event, pos):
    def expr(c):
        if c == "event":
            return event
        if c in ("action_id", "proba_negative", "risk_level"):
            return "={{ %s.%s ?? '' }}" % (src, c)
        return "={{ %s.action.%s ?? '' }}" % (src, c)

    extra = {"onError": "continueRegularOutput",
             "credentials": {"googleApi": {"id": "nrgGoogleSa00001", "name": "Google Service Account"}}}
    if not SHEET_ID:
        extra.update(disabled=True, notes="Run n8n/setup_google.py to enable (see README: Google Sheets)")
    return node(name, "n8n-nodes-base.googleSheets", 4.5, pos, {
        "authentication": "serviceAccount",
        "operation": "append",
        "documentId": {"__rl": True, "value": SHEET_ID or "PUT_YOUR_SPREADSHEET_ID_HERE", "mode": "id"},
        "sheetName": {"__rl": True, "value": "Actions", "mode": "name"},
        "columns": {"mappingMode": "defineBelow", "value": {c: expr(c) for c in SHEET_COLS}, "matchingColumns": [],
                    "schema": [{"id": c, "displayName": c, "required": False, "defaultMatch": False,
                                "display": True, "type": "string", "canBeUsedToMatch": True} for c in SHEET_COLS]},
        "options": {}}, **extra)


ROW = "<p>Game: <b>{{ %s.action.title }}</b><br>Measure: <b>{{ %s.action.action_code }}</b><br>" \
      "Cost: {{ %s.action.cost_k }} k$ · expected benefit: {{ Math.round(%s.action.expected_benefit_k) }} k$<br>" \
      "Rationale: <i>{{ %s.action.rationale }}</i></p>"

nodes = [
    webhook("Webhook: action created", "nrg-action", [200, 200]),
    sheets_log("Log to Google Sheets", A, "action_created", [440, 200]),
    if_true("Needs approval?", "={{ %s.requires_approval }}" % A, [680, 200]),
    email("E-mail manager (approve/reject)", MANAGER,
          "=[NRG] Approval needed: {{ %s.action.action_code }} for {{ %s.action.title }}" % (A, A),
          "=<h3>Negative Review Guard — approval needed</h3><p>Risk of a negative review: "
          "<b>{{ Math.round(%s.proba_negative * 100) }}%%</b> ({{ %s.risk_level }})</p>" % (A, A)
          + ROW % ((A,) * 5)
          + "<p><a href=\"{{ %s.approve_url }}\">✅ Approve</a> &nbsp; | &nbsp; "
            "<a href=\"{{ %s.reject_url }}\">❌ Reject</a></p>" % (A, A), [920, 80]),
    execute("Mark executed (auto)", A, [920, 320]),
    email("E-mail team (task created)", TEAM,
          "=[NRG] New task: {{ %s.action.action_code }} — {{ %s.action.title }}" % (A, A),
          "=<h3>New preventive task (auto-approved)</h3>" + ROW % ((A,) * 5), [1160, 320]),

    webhook("Webhook: decision", "nrg-decision", [200, 600]),
    sheets_log("Log decision to Google Sheets", D, "action_decided", [440, 600]),
    if_true("Approved?", "={{ %s.action.status === 'approved' }}" % D, [680, 600]),
    execute("Mark executed (approved)", D, [920, 500]),
    email("E-mail team (approved task)", TEAM,
          "=[NRG] Approved by manager: {{ %s.action.action_code }} — {{ %s.action.title }}" % (D, D),
          "=<h3>Manager approved a preventive measure</h3>" + ROW % ((D,) * 5)
          + "<p>Approved by: {{ %s.action.decided_by }}</p>" % D, [1160, 500]),
    email("E-mail team (rejected)", TEAM,
          "=[NRG] Rejected: {{ %s.action.action_code }} — {{ %s.action.title }}" % (D, D),
          "=<p>The manager rejected this measure; no task created.</p>" + ROW % ((D,) * 5), [920, 720]),
]


def link(*pairs):
    conns = {}
    for src, dst, out in pairs:
        conns.setdefault(src, {"main": []})
        main = conns[src]["main"]
        while len(main) <= out:
            main.append([])
        main[out].append({"node": dst, "type": "main", "index": 0})
    return conns


connections = link(
    ("Webhook: action created", "Log to Google Sheets", 0),
    ("Log to Google Sheets", "Needs approval?", 0),
    ("Needs approval?", "E-mail manager (approve/reject)", 0),
    ("Needs approval?", "Mark executed (auto)", 1),
    ("Mark executed (auto)", "E-mail team (task created)", 0),
    ("Webhook: decision", "Log decision to Google Sheets", 0),
    ("Log decision to Google Sheets", "Approved?", 0),
    ("Approved?", "Mark executed (approved)", 0),
    ("Approved?", "E-mail team (rejected)", 1),
    ("Mark executed (approved)", "E-mail team (approved task)", 0),
)

workflow = {"id": "nrgActionRouter1", "name": "Negative Review Guard — action router (HITL)", "nodes": nodes,
            "connections": connections, "active": False, "settings": {"executionOrder": "v1"}, "pinData": {}}

out = Path(__file__).with_name("nrg_workflow.json")
out.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Written {out} (Google Sheets logging: {'ON, sheet ' + SHEET_ID if SHEET_ID else 'off'})")
