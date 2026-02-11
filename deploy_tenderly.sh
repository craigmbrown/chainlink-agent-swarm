#!/bin/bash
# =============================================================================
# Tenderly Virtual TestNet Deployment Script
# Chainlink Convergence 2026 Hackathon
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTRACTS_DIR="$SCRIPT_DIR/contracts"

echo "============================================"
echo "  AI Agent Swarm - Tenderly Deployment"
echo "  Chainlink Convergence 2026"
echo "============================================"
echo ""

# ─── Step 1: Check prerequisites ─────────────────────────────────────────────
echo "[1/5] Checking prerequisites..."

if ! command -v forge &> /dev/null; then
    echo "ERROR: Foundry (forge) not found. Install: curl -L https://foundry.paradigm.xyz | bash"
    exit 1
fi
echo "  forge $(forge --version 2>&1 | head -1 | awk '{print $2}')"

# ─── Step 2: Get Tenderly RPC URL ────────────────────────────────────────────
echo ""
echo "[2/5] Tenderly Virtual TestNet Configuration"
echo ""

if [ -z "$TENDERLY_RPC_URL" ]; then
    echo "  You need a Tenderly Virtual TestNet RPC URL."
    echo ""
    echo "  To create one:"
    echo "    1. Go to https://dashboard.tenderly.co/"
    echo "    2. Click 'Virtual TestNets' in the left sidebar"
    echo "    3. Click 'Create Virtual TestNet'"
    echo "    4. Select 'Ethereum Mainnet' as base network"
    echo "    5. Copy the RPC URL"
    echo ""
    read -p "  Paste your Tenderly RPC URL: " TENDERLY_RPC_URL
    export TENDERLY_RPC_URL
fi

echo "  RPC: ${TENDERLY_RPC_URL:0:50}..."

# ─── Step 3: Generate or use private key ─────────────────────────────────────
echo ""
echo "[3/5] Wallet Configuration"

if [ -z "$PRIVATE_KEY" ]; then
    # Generate a random private key for the virtual testnet
    PRIVATE_KEY=$(cast wallet new 2>/dev/null | grep "Private key" | awk '{print $3}')
    if [ -z "$PRIVATE_KEY" ]; then
        # Fallback: use a well-known test private key
        PRIVATE_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    fi
    export PRIVATE_KEY
    echo "  Generated test wallet for Virtual TestNet"
fi

DEPLOYER=$(cast wallet address "$PRIVATE_KEY" 2>/dev/null || echo "unknown")
echo "  Deployer: $DEPLOYER"

# ─── Step 4: Fund deployer on Virtual TestNet ────────────────────────────────
echo ""
echo "[4/5] Funding deployer on Virtual TestNet..."

# Tenderly Virtual TestNets allow funding via their API
# Use cast to set balance (Tenderly supports tenderly_setBalance)
cast rpc tenderly_setBalance "$DEPLOYER" "0xDE0B6B3A7640000" \
    --rpc-url "$TENDERLY_RPC_URL" 2>/dev/null && echo "  Funded with 1 ETH" || echo "  Note: Fund manually via Tenderly dashboard if needed"

# Check balance
BALANCE=$(cast balance "$DEPLOYER" --rpc-url "$TENDERLY_RPC_URL" 2>/dev/null || echo "0")
echo "  Balance: $BALANCE wei"

# ─── Step 5: Deploy contracts ────────────────────────────────────────────────
echo ""
echo "[5/5] Deploying contracts..."
echo ""

cd "$CONTRACTS_DIR"

forge script script/Deploy.s.sol:DeployAll \
    --rpc-url "$TENDERLY_RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --broadcast \
    -vvv 2>&1 | tee /tmp/tenderly_deploy.log

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "Check your Tenderly dashboard for contract details."
echo "The deployment log is saved to /tmp/tenderly_deploy.log"
echo ""

# Extract contract addresses from log
echo "Deployed Contract Addresses:"
grep -E "OracleSubscription:|UnifiedPredictionSubscription:" /tmp/tenderly_deploy.log 2>/dev/null || echo "(check log above)"
echo ""
echo "Next: Run the demo script"
echo "  ./run_demo.sh"
