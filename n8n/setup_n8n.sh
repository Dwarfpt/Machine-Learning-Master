#!/bin/sh
# Imports credentials (Mailpit SMTP, optional Google service account) and the workflow into n8n, then publishes it.
# Run inside the n8n container:  docker compose exec n8n sh /workflows/setup_n8n.sh   then: docker compose restart n8n
set -e
# SMTP: import the Mailpit credential only once, so a real SMTP account set in the n8n UI is never overwritten
if n8n export:credentials --id=nrgSmtpMailpit01 --output=/tmp/smtp.json >/dev/null 2>&1; then
  echo "SMTP credential already exists - kept as is"
else
  n8n import:credentials --input=/workflows/credentials_mailpit.json
fi
if [ -f /workflows/credentials_google.json ]; then
  n8n import:credentials --input=/workflows/credentials_google.json
fi
n8n import:workflow --input=/workflows/nrg_workflow.json
# n8n 2.x uses publish:workflow, 1.x uses update:workflow --active=true
n8n publish:workflow --id=nrgActionRouter1 2>/dev/null || n8n update:workflow --id=nrgActionRouter1 --active=true
echo "Done. Restart n8n so the webhooks are registered: docker compose restart n8n"
