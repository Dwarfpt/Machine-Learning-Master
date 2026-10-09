"""Connects Google Sheets logging to the n8n workflow (service-account auth, no OAuth screens).

Prerequisites (see README -> Google Sheets):
  1. n8n/google_service_account.json  - JSON key of a Google Cloud service account (Sheets API enabled)
  2. NRG_GSHEET_ID=<spreadsheet id> in .env, the sheet has a tab "Actions" and is shared with the service account

Run:  python n8n/setup_google.py
Then: docker compose exec n8n sh /workflows/setup_n8n.sh  &&  docker compose restart n8n
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = HERE / "google_service_account.json"
OUT = HERE / "credentials_google.json"  # git-ignored: contains the private key
HEADER = ["event", "action_id", "created_at", "title", "action_code", "cost_k", "expected_benefit_k", "status",
          "decided_by", "proba_negative", "risk_level"]

if not KEY.exists():
    sys.exit(f"Put the service-account JSON key at {KEY}")
key = json.loads(KEY.read_text(encoding="utf-8"))
OUT.write_text(json.dumps([{
    "id": "nrgGoogleSa00001", "name": "Google Service Account", "type": "googleApi",
    "data": {"email": key["client_email"], "privateKey": key["private_key"]},
}], indent=2), encoding="utf-8")
print(f"n8n credential written -> {OUT.name}")

subprocess.run([sys.executable, str(HERE / "build_workflow.py")], check=True)
print("\nNow make sure that:")
print(f"  * the spreadsheet is shared (Editor) with: {key['client_email']}")
print("  * it has a tab named 'Actions' whose first row is (copy & paste into A1):")
print("    " + "\t".join(HEADER))
