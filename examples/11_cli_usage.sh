#!/usr/bin/env bash
# Example 11: CLI usage — all fastaiagent commands
#
# Usage:
#   chmod +x examples/11_cli_usage.sh
#   ./examples/11_cli_usage.sh

# Note: no set -e since some commands show usage examples only

echo "=== FastAIAgent CLI Examples ==="
echo ""

# --- Version ---
echo "1. Version"
fastaiagent version
echo ""

# --- Traces ---
echo "2. List recent traces (last 24 hours)"
fastaiagent traces list --last-hours 24
echo ""

echo "3. Export a trace (replace TRACE_ID with a real one)"
echo "   fastaiagent traces export <trace_id>"
echo "   fastaiagent traces export abc123def456 --format json"
echo ""

# --- Replay ---
echo "4. Replay — show steps for a trace"
echo "   fastaiagent replay show <trace_id>"
echo ""

echo "5. Replay — inspect a specific step"
echo "   fastaiagent replay inspect <trace_id> 3"
echo ""

# --- Eval ---
echo "6. Run evaluation"
echo "   fastaiagent eval run --dataset test_cases.jsonl --agent myapp:agent --scorers exact_match,contains"
echo ""

echo "7. Compare two evaluation runs"
echo "   fastaiagent eval compare results_v1.json results_v2.json"
echo ""

# --- Prompts ---
echo "8. List registered prompts"
fastaiagent prompts list
echo ""

echo "9. Diff two prompt versions"
echo "   fastaiagent prompts diff support-prompt 1 2"
echo ""

# --- Knowledge Base ---
echo "10. KB status"
fastaiagent kb status --name default
echo ""

echo "11. Add a file to KB"
echo "    fastaiagent kb add ./docs/readme.md --name product-docs"
echo ""

# --- Push to platform ---
echo "12. Push to platform (requires API key)"
echo "    export FASTAIAGENT_API_KEY=fa_k_..."
echo "    export FASTAIAGENT_TARGET=https://app.fastaiagent.net"
echo "    fastaiagent push --agent myapp:support_agent"
echo "    fastaiagent push --chain myapp:pipeline"
echo ""

echo "=== Done ==="
