#!/bin/bash
# =============================================================================
# AI Agent Swarm Demo Script - Chainlink Convergence 2026
# 5-Minute Video Recording Script
#
# Usage:
#   ./run_demo.sh              # Full demo (live LLM + simulation fallback)
#   ./run_demo.sh --sim        # Simulation only (no API keys needed)
#   ./run_demo.sh --fast       # Skip pauses (for testing)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRE_DIR="$SCRIPT_DIR/chainlink_cre"
CONTRACTS_DIR="$SCRIPT_DIR/contracts"
FAST=false
SIM_ONLY=false

for arg in "$@"; do
    case $arg in
        --fast) FAST=true ;;
        --sim) SIM_ONLY=true ;;
    esac
done

pause() {
    if [ "$FAST" = false ]; then
        sleep "${1:-2}"
    fi
}

header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Load env
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    set -a; source "$SCRIPT_DIR/../../.env" 2>/dev/null; set +a
fi

# =============================================================================
# INTRO (0:00 - 0:30)
# =============================================================================
clear
echo ""
echo "    ╔══════════════════════════════════════════════════════╗"
echo "    ║                                                      ║"
echo "    ║   🤖 AI AGENT SWARM                                  ║"
echo "    ║   with Autonomous Payments                           ║"
echo "    ║                                                      ║"
echo "    ║   Chainlink Convergence 2026 Hackathon               ║"
echo "    ║                                                      ║"
echo "    ║   What if AI agents could hire each other?           ║"
echo "    ║                                                      ║"
echo "    ╚══════════════════════════════════════════════════════╝"
echo ""
echo "    17 specialized agents | Byzantine consensus | x402 payments"
echo "    CRE workflows | CCIP cross-chain | Real LLM deliberation"
echo ""
pause 4

# =============================================================================
# SECTION 1: Architecture Overview (0:30 - 1:00)
# =============================================================================
header "SECTION 1: Architecture - CRE Workflow Pipeline"

echo "    External Trigger          CRE Workflow               On-Chain"
echo "         |                        |                         |"
echo "         |---WEBHOOK------------>|                         |"
echo "         |---EVM_LOG------------>|                         |"
echo "         |---CRON--------------->|                         |"
echo "         |                        |                         |"
echo "         |                 [1] receive_task()               |"
echo "         |                        |                         |"
echo "         |                 [2] route_to_agents()            |"
echo "         |                    /   |   \\                     |"
echo "         |                Agent Agent Agent (3-7)           |"
echo "         |                    \\   |   /                     |"
echo "         |                 [3] aggregate_consensus()        |"
echo "         |                    (67% Byzantine)               |"
echo "         |                        |                         |"
echo "         |                 [4] execute_action()             |"
echo "         |                     oracle/alert/defense         |"
echo "         |                        |                         |"
echo "         |                 [5] settle_payment()             |"
echo "         |                    x402 micropayments            |"
echo "         |                    90% agents / 10% protocol     |"
echo ""
pause 4

# =============================================================================
# SECTION 2: CRE Workflow YAML (1:00 - 1:30)
# =============================================================================
header "SECTION 2: CRE Workflow Specification"

echo "  Workflow YAML: cre-workflows/agent_swarm_orchestrator_v1.yaml"
echo ""
echo "  Triggers:"
echo "    - WEBHOOK: agent_task_request"
echo "    - EVM_LOG: TaskSubmitted event"
echo "    - CRON: 0 * * * * (hourly health check)"
echo ""
echo "  Key Actions:"
echo "    1. receive_task       -> Parse incoming requests"
echo "    2. route_to_agents    -> Select 3-7 agents per task"
echo "    3. aggregate_consensus -> Byzantine voting (67%)"
echo "    4. execute_action     -> Oracle update / alert / defense"
echo "    5. settle_payment     -> x402 micropayments in LINK"
echo "    6. ccip_settlement    -> Cross-chain via CCIP"
echo ""
pause 3

# =============================================================================
# SECTION 3: Smart Contracts (1:30 - 2:00)
# =============================================================================
header "SECTION 3: On-Chain Contracts (Compiled with Foundry)"

echo "  Compiling contracts..."
cd "$CONTRACTS_DIR"
forge build --sizes 2>&1 | grep -E "Contract|OracleSubscription|UnifiedPrediction" | head -10
echo ""
echo "  Contracts:"
echo "    OracleSubscription.sol         - Subscription management + CRE integration"
echo "      -> 4 tiers: Free, Basic (\$46), Pro (\$139), Enterprise (\$462)"
echo "      -> On-chain API key validation"
echo "      -> Auto-renewal via CRE CRON"
echo ""
echo "    UnifiedPredictionSubscription.sol - Prediction markets + IReceiver"
echo "      -> CRE workflow settlement via onReport()"
echo "      -> 24h dispute period with staking"
echo "      -> Multi-AI consensus (67% threshold)"
echo ""
echo "    IReceiver.sol                  - Chainlink CRE interface"
echo "      -> Standardized report receiving"
echo "      -> Metadata: workflow owner + name"
echo ""
pause 4

# =============================================================================
# SECTION 4: Validation (2:00 - 2:15)
# =============================================================================
header "SECTION 4: Asset Validation (27/27 checks)"

cd "$CRE_DIR"
python3 validate_hackathon_assets.py 2>&1 | tail -15
echo ""
pause 3

# =============================================================================
# SECTION 5: Live Agent Swarm Demo (2:15 - 4:00) -- THE MONEY SHOT
# =============================================================================
header "SECTION 5: LIVE Agent Swarm Demo"

echo "  Agent Pools:"
echo "    Prediction (5): market trends, sentiment, technical, macro, on-chain"
echo "    Analysis   (5): quantitative, research, risk, contracts, DeFi"
echo "    Monitoring (3): health, liquidity, anomaly"
echo "    Risk       (4): systemic, exposure, mitigation, regulatory"
echo ""
pause 2

if [ "$SIM_ONLY" = true ]; then
    echo "  Mode: SIMULATION (--sim flag set)"
    echo ""
    python3 agent_swarm_orchestrator.py --demo 2>&1
else
    echo "  Mode: LIVE LLM (Groq/Llama-3.3-70B)"
    echo ""
    python3 agent_swarm_orchestrator.py --demo --live 2>&1
fi

pause 3

# =============================================================================
# SECTION 6: Single Task Deep Dive (4:00 - 4:30)
# =============================================================================
header "SECTION 6: Single Task - ETH Price Prediction"

echo "  Submitting: 'Will ETH reach \$5,000 by March 2026?'"
echo "  Task Type:  prediction"
echo "  Agents:     5 prediction specialists"
echo ""

if [ "$SIM_ONLY" = true ]; then
    python3 agent_swarm_orchestrator.py \
        --task "Will ETH reach 5000 by March 2026?" \
        --task-type prediction 2>&1
else
    python3 agent_swarm_orchestrator.py \
        --live \
        --task "Will ETH reach 5000 by March 2026?" \
        --task-type prediction 2>&1
fi

pause 3

# =============================================================================
# SECTION 7: Payment & Results (4:30 - 5:00)
# =============================================================================
header "SECTION 7: Payment Settlement & Results"

echo "  x402 Payment Flow:"
echo "    -> Task requester pays 0.5 LINK"
echo "    -> 90% distributed to 5 contributing agents"
echo "    -> 10% protocol fee (0.05 LINK)"
echo ""
echo "  Agent Revenue Split (per task):"
echo "    pred-01: ~0.089 LINK"
echo "    pred-02: ~0.089 LINK"
echo "    pred-03: ~0.089 LINK"
echo "    pred-04: ~0.089 LINK"
echo "    pred-05: ~0.092 LINK (bonus for highest confidence)"
echo ""
echo "  Cross-Chain Settlement via CCIP:"
echo "    Supported: Ethereum, Polygon, Base, Arbitrum,"
echo "               Optimism, Avalanche, BSC, Solana"
echo ""

# Show latest demo results
LATEST=$(ls -t "$CRE_DIR/storage/demo_results_"*.json 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "  Latest Results File: $(basename "$LATEST")"
    echo ""
fi

pause 2

# =============================================================================
# CLOSING
# =============================================================================
echo ""
echo "    ╔══════════════════════════════════════════════════════╗"
echo "    ║                                                      ║"
echo "    ║   AI Agent Swarm with Autonomous Payments            ║"
echo "    ║                                                      ║"
echo "    ║   17 agents | 4 task types | Byzantine consensus     ║"
echo "    ║   x402 micropayments | CCIP cross-chain | CRE YAML   ║"
echo "    ║   Real LLM deliberation via Groq/OpenAI/XAI          ║"
echo "    ║                                                      ║"
echo "    ║   github.com/craigmbrown                             ║"
echo "    ║                                                      ║"
echo "    ╚══════════════════════════════════════════════════════╝"
echo ""
echo "    Track Submissions:"
echo "      CRE & AI Track (\$33.5K)"
echo "      Prediction Markets Track (\$32K)"
echo "      Risk & Compliance Track (\$32K)"
echo ""
