# Labus Sales RAG — Deployment Workflows
# ========================================
# Usage: make <target>
#
# WORKFLOW 1: make deploy        — push code to VPS (git push + pull + rebuild)
# WORKFLOW 2: make sync-data     — upload models/data/indexes to VPS
# WORKFLOW 3: make fetch-feedback — download chat DB from VPS for RLHF
# WORKFLOW 4: make retrain       — full cycle: fetch feedback + rebuild + sync
#
# Configuration: edit VPS_HOST and VPS_DIR below

# === Configuration ===
VPS_HOST    ?= user@your-vps-ip
VPS_DIR     ?= /opt/rag
COMPOSE     = docker compose -f docker-compose.prod.yml

# === WORKFLOW 1: Deploy Code (local -> VPS) ===
# Run after code changes. Pushes to GitHub, pulls on VPS, rebuilds container.
.PHONY: deploy
deploy:
	git push origin main
	ssh $(VPS_HOST) "cd $(VPS_DIR) && git pull && $(COMPOSE) up -d --build api"

# === WORKFLOW 2: Update Data (local -> VPS) ===
# Run after re-ingestion or index rebuild on local machine.
.PHONY: sync-data
sync-data:
	rsync -avz --progress ./models/ $(VPS_HOST):$(VPS_DIR)/models/
	rsync -avz --progress ./data/ $(VPS_HOST):$(VPS_DIR)/data/
	rsync -avz --progress ./indexes/ $(VPS_HOST):$(VPS_DIR)/indexes/
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) restart api"

# Sync only indexes (faster, after rebuild_index)
.PHONY: sync-indexes
sync-indexes:
	rsync -avz --progress ./indexes/ $(VPS_HOST):$(VPS_DIR)/indexes/
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) restart api"

# === WORKFLOW 3: Fetch Feedback (VPS -> local) ===
# Download chat history + feedback for local RLHF retraining.
.PHONY: fetch-feedback
fetch-feedback:
	scp $(VPS_HOST):$(VPS_DIR)/app.db ./feedback_from_vps.db
	@echo "Feedback DB saved to ./feedback_from_vps.db"

# === WORKFLOW 4: Full Retrain Cycle ===
# 1. Fetch feedback from VPS
# 2. Run local ingestion + index rebuild
# 3. Push updated data to VPS
.PHONY: retrain
retrain:
	$(MAKE) fetch-feedback
	python scripts/ingest.py
	python scripts/build_index.py --recreate
	$(MAKE) sync-data

# === Initial Setup ===
# First-time data upload (models are ~6GB, will take a while)
.PHONY: initial-upload
initial-upload:
	rsync -avz --progress ./models/ $(VPS_HOST):$(VPS_DIR)/models/
	rsync -avz --progress ./data/ $(VPS_HOST):$(VPS_DIR)/data/
	rsync -avz --progress ./indexes/ $(VPS_HOST):$(VPS_DIR)/indexes/

# Start services on VPS (after initial-upload)
.PHONY: up
up:
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) up -d"

# === Helpers ===
.PHONY: health
health:
	@ssh $(VPS_HOST) "curl -s http://localhost:8000/health | python3 -m json.tool"

.PHONY: logs
logs:
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) logs --tail=100 api"

.PHONY: logs-qdrant
logs-qdrant:
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) logs --tail=50 qdrant"

.PHONY: restart
restart:
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) restart"

.PHONY: stop
stop:
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) down"

.PHONY: status
status:
	ssh $(VPS_HOST) "cd $(VPS_DIR) && $(COMPOSE) ps"

# Show available commands
.PHONY: help
help:
	@echo "=== Labus Sales RAG — Deployment ==="
	@echo ""
	@echo "  deploy          Push code to VPS (git + docker rebuild)"
	@echo "  sync-data       Upload models/data/indexes to VPS"
	@echo "  sync-indexes    Upload only indexes (faster)"
	@echo "  fetch-feedback  Download chat DB from VPS"
	@echo "  retrain         Full cycle: feedback + rebuild + sync"
	@echo ""
	@echo "  initial-upload  First-time: upload all data to VPS"
	@echo "  up              Start containers on VPS"
	@echo "  health          Check API health"
	@echo "  logs            View API logs"
	@echo "  restart         Restart all services"
	@echo "  stop            Stop all services"
	@echo "  status          Show container status"
