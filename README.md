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
