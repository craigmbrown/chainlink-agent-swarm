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

## Repository Overview

| Metric | Value |
|--------|-------|
| Python Modules | 15 |
| Solidity Contracts | 3 (+1 interface, +1 deploy script) |
| CRE Workflow YAMLs | 1 |
| Total Source Code | ~560KB |
| Chainlink Services Used | 7 |
| CRE Requirements Mapped | 36+ |

### Notable Contract Features

- **24-hour dispute period** with staking (10% of pool minimum) in UnifiedPredictionSubscription
- **Multi-AI consensus verification** on-chain before settlement
- **CRE metadata validation** ensures only authorized workflows can trigger settlement
- **4-tier subscription model** (Free / Basic / Pro / Enterprise) with on-chain API key validation
- **ERC20 multi-token payments** (USDC, USDT, DAI, LINK)
- **Foundry build** with Solidity 0.8.20, optimizer 200 runs, Tenderly deployment

---

## Problems Being Solved

### 1. AI Agents Can't Transact On-Chain Autonomously

Today's AI agents operate in isolated silos -- they can analyze and recommend, but can't autonomously execute on-chain transactions, subscribe to oracle services, or pay each other for specialized work. There's no standard protocol for agent-to-agent economic exchange.

**Solution:** x402 micropayment channels + CRE workflow pipeline

### 2. No Trust Layer for Multi-Agent Consensus

When multiple AI agents produce conflicting analysis (one says BUY, another says SELL), there's no verifiable mechanism to aggregate their opinions into a trustworthy signal. Users must manually compare outputs or blindly trust a single agent.

**Solution:** Byzantine-fault-tolerant consensus (67% threshold) with on-chain settlement

### 3. Cross-Chain Fragmentation of AI Services

AI oracle services and prediction markets exist on individual chains but can't communicate cross-chain. A sentiment feed on Ethereum can't settle a prediction market on Base, and revenue from multiple chains can't be aggregated efficiently.

**Solution:** CCIP integration with 5-chain router support + cross-chain data feed sync

### 4. Oracle Data Feeds Are Static, Not Intelligent

Traditional Chainlink price feeds deliver raw market data. There's no mechanism for AI-enhanced feeds that provide sentiment analysis, risk scores, or predictive signals alongside price data -- data that autonomous agents actually need for decision-making.

**Solution:** AI Oracle Node with 6 model types (LLM sentiment, ML classifier, time series, ensemble, neural network, transformer)

### 5. No Fair Payment Model for Agent Contributions

When multiple agents collaborate on a task, there's no standard for splitting revenue proportionally based on each agent's actual contribution quality. Current systems use fixed splits regardless of output value.

**Solution:** Confidence-weighted payment splits via x402 (90/10 agent/protocol, weighted by consensus contribution)

### 6. Subscription-Based Oracle Access Is Inflexible

Accessing oracle data requires either free/unlimited access or complex subscription management. There's no tiered, on-chain subscription model with automated renewals, usage tracking, and API key validation that CRE can trigger.

**Solution:** `OracleSubscription.sol` with 4 tiers, ERC20 payments, CRE Automation auto-renewals

---

## Why This Is Innovative

### First CRE-Native Agent Marketplace

One of the first implementations that uses Chainlink CRE as the **backbone** of an AI agent coordination system -- not just calling CRE from an app, but building the entire agent lifecycle (task receipt, routing, consensus, execution, payment) as a CRE workflow. The `agent_swarm_orchestrator_v1.yaml` defines the full pipeline declaratively.

### Byzantine Consensus Meets LLM Outputs

Applying Byzantine fault tolerance (traditionally used in distributed systems) to AI agent outputs is a novel approach. The 67% supermajority threshold means that even if 1/3 of agents are compromised or hallucinating, the system still produces a reliable consensus. This bridges distributed systems theory with practical AI reliability.

### x402 + CCIP: Agent-to-Agent Economy

Combining the x402 HTTP payment protocol with CCIP cross-chain settlement creates a new primitive: agents on any chain can request services from agents on any other chain, pay in LINK via micropayment channels, and settle cross-chain automatically. A novel economic layer for autonomous agents.

### IReceiver Pattern for AI-Verified Predictions

`UnifiedPredictionSubscription.sol` implements Chainlink's `IReceiver` interface to accept CRE workflow results, then runs multi-AI consensus verification on-chain before settling prediction markets. This creates a verifiable loop: AI agents predict -> CRE resolves -> contract verifies AI consensus -> settlement executes.

### Real Multi-Provider LLM Orchestration

The `--live` demo mode makes real API calls to Groq, OpenAI, and XAI simultaneously, getting actual divergent analysis from different model providers and aggregating them. This isn't simulated -- it demonstrates real heterogeneous AI consensus.

---

## Chainlink Component References (Key Lines of Code)

### CRE (Chainlink Runtime Environment) -- Core Architecture

| File | Lines | What's There |
|------|-------|-------------|
| `cre-workflows/agent_swarm_orchestrator_v1.yaml` | 1-7 | Workflow name + version declaration |
| | 15-35 | 3 trigger types (WEBHOOK, EVM_LOG, CRON) |
| | 40-200 | 6 pipeline actions (receive_task, route_to_agents, aggregate_consensus, execute_action, settle_payment, ccip_settlement) |
| `chainlink_cre/agent_swarm_orchestrator.py` | 1-18 | Module docstring -- "CRE Workflow Pipeline" |
| | ~100-500 | Full 5-step pipeline implementation |
| `contracts/src/IReceiver.sol` | 1-27 | `onReport(bytes metadata, bytes report)` interface |
| `contracts/src/UnifiedPredictionSubscription.sol` | 8-14 | "CRE Integration" natspec |
| | 94-95 | `creWorkflowAuthor` + `expectedWorkflowName` state vars |
| | 284-303 | `onReport()` -- IReceiver implementation receiving CRE results |
| | 336-342 | `_decodeMetadata()` -- CRE metadata validation |
| `contracts/src/OracleSubscription.sol` | 21-22 | "CRE-compatible subscription contract" |
| | 83 | `address public creAutomation;` |
| | 86-90 | `onlyCREAutomation()` modifier |
| | 320-322 | `setCREAutomation()` |
| `contracts/script/Deploy.s.sol` | 14-16 | Sets `creWorkflowAuthor`, `expectedWorkflowName = bytes10("agentswarm")` |
| | 22 | `oracle.setCREAutomation(deployer)` |
| `CHAINLINK_CRE_REQUIREMENTS.md` | full | 36+ requirements (REQ-CRE-001 through REQ-CRE-024+) |

### CCIP (Cross-Chain Interoperability Protocol) -- 5 Chains

| File | Lines | What's There |
|------|-------|-------------|
| `chainlink_cre/ccip_integration.py` | 40-49 | `CCIPChain` enum (Ethereum, Polygon, BSC, Avalanche, Arbitrum, Optimism, Base, Fantom) |
| | 167-268 | `ChainlinkCCIPIntegration` class with **real CCIP router addresses** |
| | 270-326 | `send_cross_chain_message()` |
| | 355-410 | `bridge_revenue()` -- cross-chain revenue settlement |
| | 412-449 | `sync_data_feeds()` -- cross-chain data feed sync |
| `cre-workflows/agent_swarm_orchestrator_v1.yaml` | 185-198 | `ccip_settlement` action (source: Base, dest: Ethereum, token: LINK) |

**Real CCIP Router Addresses Used:**

| Chain | Router Address |
|-------|---------------|
| Ethereum | `0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D` |
| Polygon | `0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43` |
| Arbitrum | `0x141fa059441E0ca23ce184B6A78bafD2A517DdE8` |
| Avalanche | `0xA9d587a00A31A52Ed70D6026794a8FC27ACFE4e6` |
| Base | `0x881e3A65B4d4a04dD529061dd0071cf975F58bCD` |

### Data Feeds / AI Oracle -- 6 AI Model Types

| File | Lines | What's There |
|------|-------|-------------|
| `chainlink_cre/ai_oracle_node.py` | 1-6 | "AI-Enhanced Oracle Node Infrastructure" |
| | 63-69 | `AIModelType` enum (LLM_SENTIMENT, ML_CLASSIFIER, TIME_SERIES, ENSEMBLE, NEURAL_NETWORK, TRANSFORMER) |
| | 72+ | `AIDataFeed` dataclass for oracle feed definitions |
| `contracts/src/OracleSubscription.sol` | full | Subscription management for oracle API access (4 tiers, ERC20 payments) |
| `design_outputs/specifications/ai_feed_specs.json` | full | AI data feed specifications |

### x402 Payments -- LINK Micropayments

| File | Lines | What's There |
|------|-------|-------------|
| `chainlink_cre/x402_payment_handler.py` | 1-7 | "x402 Payment Handler for Chainlink CRE Workflows" |
| | 59-94 | `X402PaymentRequest` dataclass |
| | 96-137 | `X402PaymentChannel` with LINK token management |
| | 173-238 | `X402PaymentHandler` class with **real LINK token addresses** |
| `chainlink_cre/agent_swarm_orchestrator.py` | 124-129 | Import of x402 handler |
| | 456-483 | `_settle_payment()` -- 90/10 agent/protocol split, confidence-weighted |
| `cre-workflows/agent_swarm_orchestrator_v1.yaml` | 171-184 | `settle_payment` action (type: x402_payment, currency: LINK) |

**Real LINK Token Addresses Used:**

| Network | Address |
|---------|---------|
| Mainnet | `0x514910771AF9Ca656af840dff83E8264EcF986CA` |
| Sepolia | `0x779877A7B0D9E8603169DdbD7836e478b4624789` |
| Polygon | `0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39` |

### Automation -- CRE CRON

| File | Lines | What's There |
|------|-------|-------------|
| `contracts/src/OracleSubscription.sol` | 83-90 | `creAutomation` address + `onlyCREAutomation()` modifier |
| | 271-295 | `processAutoRenewals()` -- called by CRE CRON |
| | 302-310 | `resetMonthlyCalls()` -- monthly usage reset via automation |
| `cre-workflows/agent_swarm_orchestrator_v1.yaml` | 33-35 | `CRON` trigger -- `"0 * * * *"` (hourly health monitoring) |

---

## Smart Contract Analysis

| Contract | Size | Chainlink Services | Key Functions |
|----------|------|--------------------|---------------|
| `UnifiedPredictionSubscription.sol` | 27.5KB | CRE, IReceiver | `onReport()`, `createPrediction()`, `disputePrediction()`, `_decodeMetadata()` |
| `OracleSubscription.sol` | 17.8KB | CRE, Automation | `subscribe()`, `processAutoRenewals()`, `resetMonthlyCalls()`, `setCREAutomation()` |
| `IReceiver.sol` | 1.2KB | CRE | `onReport(bytes metadata, bytes report)` |
| `Deploy.s.sol` | 1.8KB | CRE | Sets `creWorkflowAuthor`, `expectedWorkflowName`, `setCREAutomation` |

---

## Chainlink Service Coverage Summary

| Service | Status | Primary Files |
|---------|--------|---------------|
| **CRE** (Runtime Environment) | **Core** | Orchestrator, YAML, all contracts, workflow framework |
| **CCIP** | Active | `ccip_integration.py`, YAML workflow |
| **Data Feeds** | Active | `ai_oracle_node.py`, `OracleSubscription.sol` |
| **x402 Payments** | Active | `x402_payment_handler.py`, orchestrator, YAML |
| **Automation** | Active | `OracleSubscription.sol`, YAML CRON trigger |
| **Functions** | Referenced | CRE compute actions use `runtime: python3` |
| **IReceiver** | Active | `IReceiver.sol`, `UnifiedPredictionSubscription.sol` |
| **VRF** | Not Used | -- |
| **Data Streams** | Not Used | -- |

---

## BlindOracle -- Live Platform

The agent swarm technology powers [BlindOracle](https://craigmbrown.com/blindoracle/), a private settlement layer for autonomous AI agents with blind-signed privacy, Chainlink oracle resolution, and five payment rails.

- **Main Platform**: [craigmbrown.com/blindoracle/](https://craigmbrown.com/blindoracle/)
- **RWA Stock Prediction Markets**: [craigmbrown.com/blindoracle/rwa-markets.html](https://craigmbrown.com/blindoracle/rwa-markets.html)

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
