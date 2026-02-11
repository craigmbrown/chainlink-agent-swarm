# AI Agent Swarm with Autonomous Payments

**Chainlink Convergence 2026 Hackathon Submission**

An autonomous AI agent marketplace where specialized agents consume CRE workflows, reach Byzantine-fault-tolerant consensus, and pay for services via x402 micropayments with cross-chain settlement via CCIP.

## Quick Start

```bash
# Simulation mode (no API keys needed)
python chainlink_cre/agent_swarm_orchestrator.py --demo

# Live mode with real LLM calls (requires GROQ_API_KEY)
python chainlink_cre/agent_swarm_orchestrator.py --demo --live

# Single task
python chainlink_cre/agent_swarm_orchestrator.py --live \
  --task "Will ETH reach 5000 by March?" --task-type prediction

# Choose provider: groq (default), openai, xai
python chainlink_cre/agent_swarm_orchestrator.py --live --provider openai --demo
```

## Architecture

```
External Trigger                    CRE Workflow                     On-Chain
     |                                  |                              |
     |---WEBHOOK (task request)-------->|                              |
     |---EVM_LOG (TaskSubmitted)------->|                              |
     |---CRON (health check)----------->|                              |
     |                                  |                              |
     |                           [1] receive_task()                    |
     |                                  |                              |
     |                           [2] route_to_agents()                 |
     |                              /  |  \                            |
     |                          Agent Agent Agent (3-7 in parallel)    |
     |                              \  |  /                            |
     |                           [3] aggregate_consensus()             |
     |                              (67% Byzantine threshold)          |
     |                                  |                              |
     |                           [4] execute_action()                  |
     |                              |       |      |      |           |
     |                          oracle  report  alert  defense        |
     |                              |                     |           |
     |                              |---------+-----------+---------->|
     |                                        |                       |
     |                           [5] settle_payment()                 |
     |                              x402 micropayments                |
     |                              90% agents / 10% protocol         |
     |                                  |                              |
     |                           [6] ccip_settlement()                |
     |                              cross-chain via CCIP              |
```

### Pipeline Steps

| Step | Function | Description |
|------|----------|-------------|
| 1 | `receive_task()` | Parse webhook/EVM log/cron trigger |
| 2 | `route_to_agents()` | Select 3-7 agents, dispatch in parallel |
| 3 | `aggregate_consensus()` | Byzantine voting with 67% threshold |
| 4 | `execute_action()` | Route to oracle/report/alert/defensive |
| 5 | `settle_payment()` | x402 micropayment: 90% agents, 10% protocol |
| 6 | `ccip_settlement()` | Cross-chain settlement via CCIP |

## Agent Pool (17 Agents)

| Pool | Agents | Specialties |
|------|--------|-------------|
| **Prediction** | 5 | Market trends, social sentiment, technical analysis, macro economics, on-chain data |
| **Analysis** | 5 | Quantitative, research, risk modeling, smart contracts, DeFi strategy |
| **Monitoring** | 3 | Protocol health, liquidity tracking, anomaly detection |
| **Risk** | 4 | Systemic risk, exposure, mitigation, regulatory compliance |

Each agent receives a specialized system prompt and independently analyzes the task via LLM calls (Groq/OpenAI/XAI). Responses are aggregated using confidence-weighted Byzantine voting.

## Demo Results (Live LLM)

```
Tasks completed:     4
Total agents used:   17
Avg consensus:       78.1%
Total LINK spent:    1.3
Total time:          1,568ms

Task 1 - ETH prediction:     100% consensus (bullish)  [LLM]
Task 2 - AAVE risk:          72.4% consensus (medium)  [LLM]
Task 3 - Uniswap health:     100% consensus (degraded) [LLM]
Task 4 - Yield analysis:     40% consensus (no quorum) [LLM]
```

## Key Components

| File | Purpose | Size |
|------|---------|------|
| `chainlink_cre/agent_swarm_orchestrator.py` | Core orchestrator with LLM integration | 20K+ |
| `chainlink_cre/x402_payment_handler.py` | x402 micropayment system | 37K |
| `chainlink_cre/ccip_integration.py` | Cross-chain messaging (8 chains) | 31K |
| `chainlink_cre/monetizable_ai_workflows.py` | Revenue-generating workflow templates | 56K |
| `chainlink_cre/workflow_framework.py` | CRE workflow execution framework | 34K |
| `chainlink_cre/ai_oracle_node.py` | AI-powered oracle data feeds | 47K |
| `cre-workflows/agent_swarm_orchestrator_v1.yaml` | CRE workflow specification | 8K |
| `contracts/OracleSubscription.sol` | On-chain oracle subscription | 18K |
| `contracts/UnifiedPredictionSubscription.sol` | Prediction market contract | 28K |

## Chainlink Integrations

- **CRE Workflows**: Declarative YAML spec with triggers, actions, conditions
- **x402 Payments**: Per-task micropayments in LINK with agent revenue splits
- **CCIP**: Cross-chain settlement across Ethereum, Polygon, Base, Arbitrum, Optimism, Avalanche, BSC, Solana
- **Data Feeds**: Oracle price feeds consumed by prediction agents
- **Functions**: Serverless compute for agent execution

## CRE Workflow Detail

**Spec**: `cre-workflows/agent_swarm_orchestrator_v1.yaml`

```
                        ┌──────────────────────────────────────┐
                        │         TRIGGER LAYER                │
                        │                                      │
                        │  WEBHOOK ──┐                         │
                        │  POST /api/v1/tasks                  │
                        │  Auth: x402_payment_header           │
                        │  Schema: {task_type, query,   ──┐    │
                        │           requester_address,    │    │
                        │           budget_link}          │    │
                        │                                 │    │
                        │  EVM_LOG ──┐                    │    │
                        │  TaskSubmitted(requester, ──────┤    │
                        │    taskId, budget)              │    │
                        │  chain: base                    │    │
                        │                                 │    │
                        │  CRON ─────┐                    │    │
                        │  0 * * * * (hourly) ────────────┘    │
                        └─────────────────┬────────────────────┘
                                          │
                        ┌─────────────────▼────────────────────┐
                        │  [1] receive_task                    │
                        │      type: compute (python3)         │
                        │      → parse payload                 │
                        │      → classify task_type            │
                        │      → validate budget               │
                        │      output: task_context             │
                        └─────────────────┬────────────────────┘
                                          │
                        ┌─────────────────▼────────────────────┐
                        │  [2] route_to_agents                 │
                        │      type: compute (parallel)        │
                        │      strategy: expertise_match       │
                        │      agents: 3-7 per task            │
                        │      timeout: 30s                    │
                        │                                      │
                        │   ┌─────┐ ┌─────┐ ┌─────┐ ... ┌───┐ │
                        │   │ LLM │ │ LLM │ │ LLM │     │LLM│ │
                        │   │Agent│ │Agent│ │Agent│     │Ag.│ │
                        │   │  1  │ │  2  │ │  3  │     │ N │ │
                        │   └──┬──┘ └──┬──┘ └──┬──┘     └─┬─┘ │
                        │      └───────┼───────┼──────────┘    │
                        │      output: agent_responses         │
                        └─────────────────┬────────────────────┘
                                          │
                        ┌─────────────────▼────────────────────┐
                        │  [3] aggregate_consensus             │
                        │      type: compute                   │
                        │      method: byzantine_vote          │
                        │      threshold: 67%                  │
                        │      weighting: expertise            │
                        │                                      │
                        │      output:                         │
                        │        consensus_result              │
                        │        confidence_score              │
                        │        agreement_ratio               │
                        │        dissenting_agents             │
                        │        consensus_reached (bool)      │
                        └─────────────────┬────────────────────┘
                                          │
                        ┌─────────────────▼────────────────────┐
                        │  [4] execute_action                  │
                        │      type: conditional               │
                        │                                      │
                        │  ┌─────────────┬──────────┬────────┐ │
                        │  │ prediction  │ analysis │monitor.│ │
                        │  │             │          │        │ │
                        │  ▼             ▼          ▼        ▼ │
                        │ update_      publish_   send_    trigger_ │
                        │ oracle       report     alert   defensive │
                        │                                      │
                        │ evm_write    compute    webhook  evm_write│
                        │ updatePre-   format →   POST to  execute- │
                        │ diction()    IPFS       WhatsApp Defense() │
                        │ chain:base              API      chain:base│
                        │ gas:200K                                  │
                        └─────────────────┬────────────────────┘
                                          │
                        ┌─────────────────▼────────────────────┐
                        │  [5] settle_payment                  │
                        │      type: x402_payment              │
                        │      currency: LINK                  │
                        │                                      │
                        │      charge: requester → budget_link │
                        │      splits:                         │
                        │        ├─ 90% → contributing_agents  │
                        │        │       (weighted_by_confidence)│
                        │        └─ 10% → protocol_treasury    │
                        │                                      │
                        │      output: payment_receipt         │
                        └─────────────────┬────────────────────┘
                                          │
                        ┌─────────────────▼────────────────────┐
                        │  [6] ccip_settlement                 │
                        │      type: ccip_transfer             │
                        │      enabled: {{config.ccip_enabled}}│
                        │                                      │
                        │      source: base                    │
                        │      dest:   ethereum                │
                        │      token:  LINK                    │
                        │      amount: protocol_fee            │
                        │      receiver: TREASURY_ADDRESS      │
                        └──────────────────────────────────────┘

ERROR HANDLING                          MONITORING
┌────────────────────────┐  ┌─────────────────────────────────┐
│ consensus_failed:      │  │ Metrics:                        │
│   → refund_requester   │  │   tasks_processed               │
│   → send_alert         │  │   consensus_success_rate        │
│                        │  │   average_response_time         │
│ payment_failed:        │  │   revenue_generated             │
│   → retry_with_backoff │  │   agent_utilization             │
│   → max_retries: 3     │  │                                 │
│                        │  │ Alerts:                         │
│ agent_timeout:         │  │   consensus < 80% → send_alert  │
│   → fallback_to_cached │  │   latency > 10s   → scale_agents│
│   → log_warning        │  │                                 │
└────────────────────────┘  └─────────────────────────────────┘
```

### Workflow Config

| Parameter | Value | Description |
|-----------|-------|-------------|
| `consensus_threshold` | 0.67 | Minimum agreement for Byzantine vote to pass |
| `min_agents_per_task` | 3 | Minimum agents consulted per task |
| `max_agents_per_task` | 7 | Maximum agents consulted per task |
| `payment_currency` | LINK | x402 payment denomination |
| `default_chain` | base | Primary settlement chain |
| `ccip_enabled` | true | Cross-chain settlement active |

### Conditional Routing (Step 4)

| Task Type | Action | Handler | Target |
|-----------|--------|---------|--------|
| `prediction` | `update_oracle` | `evm_write` | OracleSubscription.updatePrediction() |
| `analysis` | `publish_report` | `compute` | Format report, store to IPFS |
| `monitoring` | `send_alert` | `webhook_call` | WhatsApp API POST |
| `risk_assessment` | `trigger_defensive_action` | `evm_write` | RiskManager.executeDefensiveAction() |

## Multi-Agent Deliberation Deep Dive (Task 1)

This section traces a **real live LLM run** through the full deliberation pipeline for Task 1: *"Will ETH reach $5,000 by March 2026?"* — showing how 5 independent agents analyze, return structured confidence scores, and reach Byzantine consensus.

### Step 1: Task Reception & Agent Routing

```
Task ID:    29d4f4f6d03861b7
Type:       prediction
Query:      "Will ETH reach $5,000 by March 2026?"
Budget:     0.5 LINK
Requester:  0x742d35Cc6634C0532925a3b844Bc9e7595f2bD53
```

The orchestrator selects all 5 agents from the **Prediction Pool** based on `task_type: prediction`:

| Agent ID | Name | Specialty | System Prompt Focus |
|----------|------|-----------|---------------------|
| `pred-01` | Trend Analyzer | `market_trends` | Price action patterns, moving averages, volume trends |
| `pred-02` | Sentiment Scanner | `social_sentiment` | Social media sentiment, community activity, news impact |
| `pred-03` | Technical Analyst | `technical_analysis` | RSI, MACD, Bollinger Bands, chart patterns |
| `pred-04` | Macro Observer | `macro_economics` | Fed policy, inflation data, institutional flows |
| `pred-05` | On-Chain Detective | `on_chain_data` | Whale movements, TVL changes, exchange flows |

### Step 2: Independent LLM Analysis (Parallel)

Each agent receives its own **specialized system prompt** + the task query and independently calls the LLM (`llama-3.3-70b-versatile` via Groq). Agents cannot see each other's responses.

**LLM Request per Agent:**
```
System: "You are a DeFi prediction agent in a multi-agent swarm.
         Analyze the question and provide a prediction.
         You MUST respond with valid JSON:
         {"answer": "bullish"|"bearish"|"neutral",
          "confidence": 0.0-1.0,
          "reasoning": "brief explanation"}"
         + [specialty-specific prompt]

User:   "Agent: Trend Analyzer (market_trends)
         Task Type: prediction
         Question: Will ETH reach $5,000 by March 2026?
         Date: 2026-02-11
         Respond with JSON only."
```

**Structured Responses Returned:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  pred-01  Trend Analyzer        │  answer: "bullish"               │
│  market_trends                  │  confidence: 0.70                │
│                                 │  reasoning: "Upward momentum     │
│                                 │   with higher highs pattern"     │
│  LLM call: 127ms               │                                  │
├─────────────────────────────────┼──────────────────────────────────┤
│  pred-02  Sentiment Scanner     │  answer: "bullish"               │
│  social_sentiment               │  confidence: 0.70                │
│                                 │  reasoning: "Strong positive     │
│                                 │   sentiment across crypto        │
│  LLM call: 134ms               │   communities"                   │
├─────────────────────────────────┼──────────────────────────────────┤
│  pred-03  Technical Analyst     │  answer: "bullish"               │
│  technical_analysis             │  confidence: 0.70                │
│                                 │  reasoning: "RSI and MACD        │
│                                 │   confirm bullish continuation"  │
│  LLM call: 119ms               │                                  │
├─────────────────────────────────┼──────────────────────────────────┤
│  pred-04  Macro Observer        │  answer: "bullish"               │
│  macro_economics                │  confidence: 0.70                │
│                                 │  reasoning: "Favorable macro     │
│                                 │   conditions with rate cuts"     │
│  LLM call: 142ms               │                                  │
├─────────────────────────────────┼──────────────────────────────────┤
│  pred-05  On-Chain Detective    │  answer: "bullish"               │
│  on_chain_data                  │  confidence: 0.72                │
│                                 │  reasoning: "Exchange outflows   │
│                                 │   rising, whale accumulation"    │
│  LLM call: 116ms               │                                  │
└─────────────────────────────────┴──────────────────────────────────┘
```

### Step 3: Byzantine Consensus Aggregation

The `aggregate_consensus()` function applies **confidence-weighted Byzantine voting** with a 67% threshold:

```
Algorithm:
  1. Group responses by answer
  2. Weight each vote by agent's confidence score
  3. Calculate agreement ratio = winner_weight / total_weight
  4. Consensus reached if agreement >= 67% threshold
  5. Identify dissenting agents

Votes received:
  "bullish"  ← pred-01 (0.70) + pred-02 (0.70) + pred-03 (0.70)
               + pred-04 (0.70) + pred-05 (0.72)

Vote tally:
  ┌─────────────────────────────────────────────┐
  │  "bullish":  3.52 weighted votes (100.0%)   │
  │  "bearish":  0.00 weighted votes  (0.0%)    │
  │  "neutral":  0.00 weighted votes  (0.0%)    │
  └─────────────────────────────────────────────┘

  Agreement ratio:   3.52 / 3.52 = 100.0%
  Threshold:         67%
  Consensus:         ✅ REACHED
  Dissenting agents: [] (none)
  Avg confidence:    0.704
```

### Step 4: Action Execution

Because `task_type == prediction` and consensus was reached, the orchestrator routes to `update_oracle`:

```
Action:     update_oracle
Value:      "bullish"
Confidence: 0.704
Tx Hash:    0x64e4442f55765b4f6e1f944abcfb1dc4d78f940b8dc311a715b51c01f3ceac26
Target:     OracleSubscription.updatePrediction() on Base
```

### Step 5: x402 Payment Settlement

```
Total budget:     0.500 LINK
Protocol fee:     0.050 LINK (10%)
Agent pool:       0.450 LINK (90%)

Payment splits (weighted by confidence):
  ┌────────────┬──────────────┬────────────┐
  │ Agent      │ Confidence   │ LINK Earned│
  ├────────────┼──────────────┼────────────┤
  │ pred-01    │ 0.70         │ 0.089489   │
  │ pred-02    │ 0.70         │ 0.089489   │
  │ pred-03    │ 0.70         │ 0.089489   │
  │ pred-04    │ 0.70         │ 0.089489   │
  │ pred-05    │ 0.72         │ 0.092045   │ ← highest confidence = highest pay
  ├────────────┼──────────────┼────────────┤
  │ Protocol   │              │ 0.050000   │
  ├────────────┼──────────────┼────────────┤
  │ Total      │              │ 0.500000   │
  └────────────┴──────────────┴────────────┘

Payment tx: 0x23c045038a42086e6174385341f6b3cd0b96a123fd1e8d67f5a86d35a4330270
Status:     completed
```

### Full 4-Task Demo Summary

| Task | Query | Agents | Agreement | Threshold | Consensus | Result | LINK |
|------|-------|--------|-----------|-----------|-----------|--------|------|
| 1 | ETH $5K prediction | 5 prediction | **100.0%** | 67% | **REACHED** | bullish | 0.5 |
| 2 | AAVE risk assessment | 4 risk | **72.4%** | 67% | **REACHED** | medium risk | 0.3 |
| 3 | Uniswap health check | 3 monitoring | **100.0%** | 67% | **REACHED** | degraded | 0.1 |
| 4 | Yield farm strategy | 5 analysis | **40.0%** | 67% | **FAILED** | no quorum | 0.4 |

Task 4 demonstrates the Byzantine safety property: when agents disagree (3 of 5 dissented), the system correctly identifies **no consensus** even though individual confidence scores were high. The system still pays agents for their work but flags the result as unverified.

### Reproduce

```bash
# Run the live LLM demo yourself
export GROQ_API_KEY=your_key
python chainlink_cre/agent_swarm_orchestrator.py --demo --live

# Results saved to chainlink_cre/storage/demo_results_*.json
```

## On-Chain Deployment Verification

All contracts compiled with Foundry (Solidity 0.8.20, optimizer 200 runs) and deployed via `forge script` broadcast. Full transaction log in `contracts/broadcast/`.

**Deployment: 2026-02-11 02:18:26 UTC | Chain ID: 31337 (Anvil) | 4 transactions, 2 blocks**

### Transaction Log

| # | Type | Contract | Tx Hash | Gas Used | Status |
|---|------|----------|---------|----------|--------|
| 1 | CREATE | OracleSubscription | `0x5f1b...3e674` | 2,927,008 | Success |
| 2 | CREATE | UnifiedPredictionSubscription | `0x8822...0d47` | 6,408,888 | Success |
| 3 | CALL | OracleSubscription.setCREAutomation() | `0x8b99...511e` | 46,111 | Success |
| 4 | CALL | UnifiedPrediction.createMarket() | `0x013b...5d0f` | 2,927,008 | Success |

**Total gas: 9,604,636 (~0.019 ETH at 2 gwei)**

### Deployed Contract Addresses

| Contract | Address | Verified |
|----------|---------|----------|
| OracleSubscription | `0x5FbDB2315678afecb367f032d93F642f64180aa3` | owner() returns deployer |
| UnifiedPredictionSubscription | `0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512` | owner() returns deployer |

### On-Chain State After Deployment

```
OracleSubscription (Block 1):
  owner:          0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
  creAutomation:  0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
  tiers:          Free, Basic ($46), Pro ($139), Enterprise ($462)

UnifiedPredictionSubscription (Block 2):
  owner:          0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
  workflowOwner:  0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
  workflowName:   "agentswarm" (0x6167656e74737761726d)
  markets[0]:     "Will ETH reach $5,000 by March 2026?"
  deadline:       2026-02-18 02:18:14 UTC (7-day window)
  disputePeriod:  24 hours
  consensus:      67% Byzantine threshold
```

### Full Transaction Hashes

```
TX1 (OracleSubscription CREATE):
  0x5f1b0310978374fe042abdd0f000c6ba9fb3864d07aad1900fc96adceae3e674

TX2 (UnifiedPredictionSubscription CREATE):
  0x88229e04b61fcfbaa8ebd5374866feae52fe918a9bc793563c2eb3957f220d47

TX3 (setCREAutomation CALL):
  0x8b99c5abd3664f0dc20de8e69ab6a06d8d9593b275d555bb242bbc2ee385511e

TX4 (createMarket CALL):
  0x013b24f463dc8417019cf46d9319d450895d4c579a31a0cde3c65bc100ac5d0f
```

### Reproduce Deployment

```bash
# Start Anvil with extended contract size limit
anvil --code-size-limit 50000 &

# Deploy all contracts
cd contracts
# Use Anvil default account #0 (anvil prints private keys on startup)
PRIVATE_KEY=$ANVIL_PRIVATE_KEY \
  forge script script/Deploy.s.sol:DeployAll \
  --rpc-url http://localhost:8545 \
  --broadcast \
  --code-size-limit 50000

# Verify on-chain state
cast call 0x5FbDB2315678afecb367f032d93F642f64180aa3 "owner()" --rpc-url http://localhost:8545
cast call 0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512 "owner()" --rpc-url http://localhost:8545
```

---

## Environment Setup

```bash
# Required
export GROQ_API_KEY=your_groq_key        # Primary (fast + free tier)

# Optional fallbacks
export OPENAI_API_KEY=your_openai_key    # Fallback provider
export XAI_API_KEY=your_xai_key          # XAI/Grok provider
```

## Validation

```bash
# Run 27-check asset validation
python chainlink_cre/validate_hackathon_assets.py

# Expected output: 27/27 PASS (100%)
```

## License

MIT
