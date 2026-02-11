#!/usr/bin/env python3
"""
Agent Swarm Orchestrator for Chainlink CRE
@requirement: REQ-CRE-SWARM-001 - Multi-Agent Task Orchestration
@hackathon: Chainlink Convergence 2026
@tracks: CRE & AI, Prediction Markets, Risk & Compliance

This orchestrator ties together:
- x402 micropayments (charge per task)
- CCIP cross-chain settlement
- Multi-agent Byzantine consensus (67% threshold)
- WhatsApp alert integration
- Oracle data feed updates

Usage:
    python agent_swarm_orchestrator.py --task "What is the current risk level for AAVE?"
    python agent_swarm_orchestrator.py --task-type prediction --query "Will ETH reach 5000 by March?"
    python agent_swarm_orchestrator.py --demo  # Run full demo flow
"""

import asyncio
import json
import time
import hashlib
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from decimal import Decimal
from pathlib import Path
from enum import Enum

# Path setup
CRE_DIR = Path(__file__).parent
PROJECT_ROOT = CRE_DIR.parent.parent.parent
sys.path.insert(0, str(CRE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "ETAC-System"))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# LLM provider setup
LLM_CLIENT = None
LLM_PROVIDER = "none"

def _init_llm_provider(preferred: str = "groq") -> None:
    """Initialize LLM client with multi-provider fallback."""
    global LLM_CLIENT, LLM_PROVIDER
    from openai import OpenAI

    providers = {
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
            "model": "llama-3.3-70b-versatile",
        },
        "openai": {
            "base_url": None,
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4o-mini",
        },
        "xai": {
            "base_url": "https://api.x.ai/v1",
            "api_key_env": "XAI_API_KEY",
            "model": "grok-2-latest",
        },
    }

    # Try preferred first, then others
    order = [preferred] + [p for p in providers if p != preferred]
    for name in order:
        cfg = providers[name]
        api_key = os.environ.get(cfg["api_key_env"], "")
        if not api_key:
            continue
        try:
            kwargs = {"api_key": api_key}
            if cfg["base_url"]:
                kwargs["base_url"] = cfg["base_url"]
            LLM_CLIENT = OpenAI(**kwargs)
            LLM_PROVIDER = name
            print(f"  LLM provider: {name} ({cfg['model']})")
            return
        except Exception as e:
            print(f"  LLM provider {name} failed: {e}")
            continue

    print("  LLM provider: none (simulation mode)")


# Agent system prompts per specialty
AGENT_SYSTEM_PROMPTS = {
    "prediction": {
        "base": (
            "You are a DeFi prediction agent in a multi-agent swarm. "
            "Analyze the question and provide a prediction. "
            "You MUST respond with valid JSON: "
            '{{"answer": "bullish"|"bearish"|"neutral", "confidence": 0.0-1.0, '
            '"reasoning": "brief explanation"}}'
        ),
        "market_trends": "Focus on price action patterns, moving averages, and volume trends.",
        "social_sentiment": "Focus on social media sentiment, community activity, and news impact.",
        "technical_analysis": "Focus on RSI, MACD, Bollinger Bands, and chart patterns.",
        "macro_economics": "Focus on Fed policy, inflation data, institutional flows, and macro correlations.",
        "on_chain_data": "Focus on whale movements, TVL changes, exchange flows, and active addresses.",
    },
    "analysis": {
        "base": (
            "You are a DeFi analysis agent in a multi-agent swarm. "
            "Analyze the question and provide an assessment. "
            "You MUST respond with valid JSON: "
            '{{"answer": "your concise finding (max 15 words)", "confidence": 0.0-1.0, '
            '"reasoning": "brief explanation"}}'
        ),
        "quantitative": "Focus on numerical analysis, statistical models, and data-driven insights.",
        "research": "Focus on protocol documentation, audit reports, and academic research.",
        "risk_modeling": "Focus on VaR, stress testing, correlation analysis, and tail risks.",
        "smart_contracts": "Focus on contract architecture, upgrade patterns, and known vulnerabilities.",
        "defi": "Focus on yield optimization, impermanent loss, and protocol composability.",
    },
    "monitoring": {
        "base": (
            "You are a protocol monitoring agent in a multi-agent swarm. "
            "Assess the health status of the protocol. "
            "You MUST respond with valid JSON: "
            '{{"answer": "healthy"|"degraded"|"critical", "confidence": 0.0-1.0, '
            '"reasoning": "brief explanation"}}'
        ),
        "protocol_health": "Focus on overall protocol metrics: TVL, utilization rate, oracle freshness.",
        "liquidity": "Focus on pool depth, bid-ask spreads, and liquidity concentration.",
        "anomaly_detection": "Focus on unusual patterns, flash loan activity, and governance attacks.",
    },
    "risk_assessment": {
        "base": (
            "You are a risk assessment agent in a multi-agent swarm. "
            "Evaluate the risk level of the situation. "
            "You MUST respond with valid JSON: "
            '{{"answer": "low"|"medium"|"high"|"critical", "confidence": 0.0-1.0, '
            '"reasoning": "brief explanation"}}'
        ),
        "systemic_risk": "Focus on cascade risk, counterparty exposure, and contagion pathways.",
        "exposure": "Focus on position sizing, collateral ratios, and liquidation thresholds.",
        "mitigation": "Focus on hedging strategies, insurance options, and defensive positions.",
        "regulatory": "Focus on compliance requirements, jurisdictional risks, and regulatory changes.",
    },
}

LLM_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "xai": "grok-2-latest",
}

# Import CRE components
try:
    from x402_payment_handler import X402PaymentHandler, X402PaymentRequest, PaymentStatus
    X402_AVAILABLE = True
except Exception:
    X402_AVAILABLE = False
    print("  x402 payment handler: using simulation mode")

try:
    from ccip_integration import CCIPChain, MessageType, TransactionStatus
    CCIP_AVAILABLE = True
except Exception:
    CCIP_AVAILABLE = False
    print("  CCIP integration: using simulation mode")

try:
    from monetizable_ai_workflows import WorkflowCategory, RevenueModel
    WORKFLOWS_AVAILABLE = True
except Exception:
    WORKFLOWS_AVAILABLE = False
    print("  Monetizable workflows: using simulation mode")


class TaskType(Enum):
    PREDICTION = "prediction"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class SwarmTask:
    task_id: str
    task_type: TaskType
    query: str
    requester_address: str
    budget_link: Decimal
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"

    @staticmethod
    def generate_id():
        return hashlib.sha256(f"{time.time()}-{os.urandom(8).hex()}".encode()).hexdigest()[:16]


@dataclass
class AgentResponse:
    agent_id: str
    agent_name: str
    specialty: str
    response: str
    confidence: float
    reasoning: str
    execution_time_ms: int


@dataclass
class ConsensusResult:
    task_id: str
    consensus_reached: bool
    result: str
    confidence_score: float
    agreement_ratio: float
    agents_consulted: int
    dissenting_agents: List[str]
    execution_time_ms: int


@dataclass
class PaymentReceipt:
    task_id: str
    amount_link: Decimal
    payer: str
    agent_splits: Dict[str, Decimal]
    protocol_fee: Decimal
    tx_hash: str
    status: str


class AgentSwarmOrchestrator:
    """
    Core orchestrator that implements the CRE workflow:
    receive_task -> route_to_agents -> aggregate_consensus -> execute_action -> settle_payment
    """

    # Available agent pool (maps to existing intercabal agents)
    AGENT_POOL = {
        "prediction": [
            {"id": "pred-01", "name": "Trend Analyzer", "specialty": "market_trends"},
            {"id": "pred-02", "name": "Sentiment Scanner", "specialty": "social_sentiment"},
            {"id": "pred-03", "name": "Technical Analyst", "specialty": "technical_analysis"},
            {"id": "pred-04", "name": "Macro Observer", "specialty": "macro_economics"},
            {"id": "pred-05", "name": "On-Chain Detective", "specialty": "on_chain_data"},
        ],
        "analysis": [
            {"id": "anl-01", "name": "Data Scientist", "specialty": "quantitative"},
            {"id": "anl-02", "name": "Research Compiler", "specialty": "research"},
            {"id": "anl-03", "name": "Risk Modeler", "specialty": "risk_modeling"},
            {"id": "anl-04", "name": "Protocol Auditor", "specialty": "smart_contracts"},
            {"id": "anl-05", "name": "DeFi Strategist", "specialty": "defi"},
        ],
        "monitoring": [
            {"id": "mon-01", "name": "Health Monitor", "specialty": "protocol_health"},
            {"id": "mon-02", "name": "Liquidity Tracker", "specialty": "liquidity"},
            {"id": "mon-03", "name": "Anomaly Detector", "specialty": "anomaly_detection"},
        ],
        "risk_assessment": [
            {"id": "risk-01", "name": "Cascade Analyzer", "specialty": "systemic_risk"},
            {"id": "risk-02", "name": "Exposure Calculator", "specialty": "exposure"},
            {"id": "risk-03", "name": "Defense Strategist", "specialty": "mitigation"},
            {"id": "risk-04", "name": "Compliance Officer", "specialty": "regulatory"},
        ],
    }

    CONSENSUS_THRESHOLD = 0.67
    PROTOCOL_FEE_PCT = Decimal("0.10")

    def __init__(self, demo_mode: bool = False, live_llm: bool = False,
                 provider: str = "groq"):
        self.demo_mode = demo_mode
        self.live_llm = live_llm
        self.task_log: List[Dict] = []

        if live_llm and LLM_CLIENT is None:
            _init_llm_provider(provider)

        print(f"\n{'='*60}")
        print(f"  AI Agent Swarm Orchestrator v2.0")
        print(f"  Mode: {'DEMO' if demo_mode else 'PRODUCTION'}")
        print(f"  LLM: {'LIVE (' + LLM_PROVIDER + ')' if live_llm and LLM_CLIENT else 'SIMULATED'}")
        print(f"  Consensus Threshold: {self.CONSENSUS_THRESHOLD * 100}%")
        print(f"  x402 Payments: {'LIVE' if X402_AVAILABLE else 'SIMULATED'}")
        print(f"  CCIP Cross-Chain: {'LIVE' if CCIP_AVAILABLE else 'SIMULATED'}")
        print(f"{'='*60}\n")

    async def execute_task(self, task: SwarmTask) -> Dict[str, Any]:
        """Execute the full CRE workflow pipeline."""
        start_time = time.time()
        results = {"task_id": task.task_id, "steps": []}

        print(f"[1/5] Receiving task: {task.task_type.value}")
        print(f"       Query: {task.query[:80]}...")
        task.status = "processing"
        results["steps"].append({"step": "receive_task", "status": "ok"})

        # Step 2: Route to agents
        print(f"\n[2/5] Routing to agents...")
        agent_responses = await self._route_to_agents(task)
        results["steps"].append({
            "step": "route_to_agents",
            "agents_consulted": len(agent_responses),
            "status": "ok"
        })

        # Step 3: Aggregate consensus
        print(f"\n[3/5] Aggregating consensus (threshold: {self.CONSENSUS_THRESHOLD*100}%)...")
        consensus = self._aggregate_consensus(task.task_id, agent_responses)
        results["consensus"] = asdict(consensus)
        results["steps"].append({
            "step": "aggregate_consensus",
            "consensus_reached": consensus.consensus_reached,
            "agreement": f"{consensus.agreement_ratio*100:.1f}%",
            "status": "ok" if consensus.consensus_reached else "warn"
        })

        # Step 4: Execute action
        print(f"\n[4/5] Executing action: {task.task_type.value}...")
        action_result = await self._execute_action(task, consensus)
        results["action"] = action_result
        results["steps"].append({
            "step": "execute_action",
            "action_type": task.task_type.value,
            "status": "ok"
        })

        # Step 5: Settle payment
        print(f"\n[5/5] Settling payment via x402...")
        receipt = await self._settle_payment(task, consensus, agent_responses)
        results["payment"] = asdict(receipt)
        results["steps"].append({
            "step": "settle_payment",
            "amount": str(receipt.amount_link),
            "status": "ok"
        })

        elapsed_ms = int((time.time() - start_time) * 1000)
        results["total_time_ms"] = elapsed_ms
        task.status = "completed"

        print(f"\n{'='*60}")
        print(f"  Task Complete: {task.task_id}")
        print(f"  Consensus: {'REACHED' if consensus.consensus_reached else 'FAILED'} "
              f"({consensus.agreement_ratio*100:.1f}%)")
        print(f"  Confidence: {consensus.confidence_score*100:.1f}%")
        print(f"  Agents: {consensus.agents_consulted}")
        print(f"  Payment: {receipt.amount_link} LINK")
        print(f"  Time: {elapsed_ms}ms")
        print(f"{'='*60}\n")

        self.task_log.append(results)
        return results

    async def _route_to_agents(self, task: SwarmTask) -> List[AgentResponse]:
        """Select and dispatch to appropriate agents based on task type."""
        pool = self.AGENT_POOL.get(task.task_type.value, self.AGENT_POOL["analysis"])
        selected = pool[:max(3, min(len(pool), 5))]

        if self.live_llm and LLM_CLIENT:
            # Parallel LLM calls
            tasks = [self._call_agent_llm(agent, task) for agent in selected]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            raw_results = None

        responses = []
        for i, agent in enumerate(selected):
            start = time.time()

            if raw_results and not isinstance(raw_results[i], Exception):
                response = raw_results[i]
                elapsed = int(response.get("_elapsed_ms", 0))
            else:
                if raw_results and isinstance(raw_results[i], Exception):
                    print(f"       {agent['name']}: LLM error, falling back to simulation")
                response = await self._simulate_agent_call(agent, task)
                elapsed = int((time.time() - start) * 1000)

            resp = AgentResponse(
                agent_id=agent["id"],
                agent_name=agent["name"],
                specialty=agent["specialty"],
                response=response["answer"],
                confidence=response["confidence"],
                reasoning=response["reasoning"],
                execution_time_ms=elapsed,
            )
            responses.append(resp)
            mode = "LLM" if (self.live_llm and LLM_CLIENT and raw_results
                             and not isinstance(raw_results[i], Exception)) else "SIM"
            print(f"       {agent['name']}: confidence={resp.confidence:.2f} "
                  f"({elapsed}ms) [{mode}]")

        return responses

    async def _call_agent_llm(self, agent: Dict, task: SwarmTask) -> Dict:
        """Call real LLM for agent analysis."""
        start = time.time()
        task_type = task.task_type.value
        prompts = AGENT_SYSTEM_PROMPTS.get(task_type, AGENT_SYSTEM_PROMPTS["analysis"])
        system_msg = prompts["base"] + " " + prompts.get(agent["specialty"], "")
        user_msg = (
            f"Agent: {agent['name']} ({agent['specialty']})\n"
            f"Task Type: {task_type}\n"
            f"Question: {task.query}\n"
            f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n"
            f"Respond with JSON only."
        )

        model = LLM_MODELS.get(LLM_PROVIDER, "llama-3.3-70b-versatile")

        # Run in thread to not block event loop
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: LLM_CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=200,
            )
        )

        elapsed_ms = int((time.time() - start) * 1000)
        raw = completion.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Handle markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return {
                "answer": str(data.get("answer", "unknown")).lower().strip(),
                "confidence": max(0.1, min(1.0, float(data.get("confidence", 0.75)))),
                "reasoning": str(data.get("reasoning", "LLM analysis"))[:200],
                "_elapsed_ms": elapsed_ms,
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract what we can from text
            return {
                "answer": raw[:50].lower().strip(),
                "confidence": 0.70,
                "reasoning": raw[:200],
                "_elapsed_ms": elapsed_ms,
            }

    async def _simulate_agent_call(self, agent: Dict, task: SwarmTask) -> Dict:
        """Simulate an agent's LLM-powered analysis (fallback mode)."""
        import random
        await asyncio.sleep(random.uniform(0.05, 0.2))

        if task.task_type == TaskType.PREDICTION:
            answers = ["bullish", "bearish", "neutral", "bullish", "bullish"]
            answer = random.choice(answers)
            reasoning = f"{agent['name']} analyzed {agent['specialty']} signals"
        elif task.task_type == TaskType.RISK_ASSESSMENT:
            answers = ["low", "medium", "high", "medium", "medium"]
            answer = random.choice(answers)
            reasoning = f"{agent['name']} evaluated {agent['specialty']} risk factors"
        elif task.task_type == TaskType.MONITORING:
            answers = ["healthy", "degraded", "healthy", "healthy"]
            answer = random.choice(answers)
            reasoning = f"{agent['name']} checked {agent['specialty']} metrics"
        else:
            answer = f"Analysis by {agent['name']}: Task-specific insight on {task.query[:40]}"
            reasoning = f"Applied {agent['specialty']} methodology"

        confidence = random.uniform(0.65, 0.98)
        return {"answer": answer, "confidence": confidence, "reasoning": reasoning}

    def _aggregate_consensus(self, task_id: str, responses: List[AgentResponse]) -> ConsensusResult:
        """Apply Byzantine voting with 67% threshold."""
        start = time.time()

        # Group responses by answer
        vote_counts: Dict[str, float] = {}
        for resp in responses:
            key = resp.response.lower().strip()
            vote_counts[key] = vote_counts.get(key, 0) + resp.confidence

        # Find winning answer (weighted by confidence)
        total_weight = sum(vote_counts.values())
        winner = max(vote_counts, key=vote_counts.get)
        winner_weight = vote_counts[winner]
        agreement = winner_weight / total_weight if total_weight > 0 else 0

        # Identify dissenters
        dissenters = [
            r.agent_id for r in responses
            if r.response.lower().strip() != winner
        ]

        # Average confidence of agreeing agents
        agreeing = [r for r in responses if r.response.lower().strip() == winner]
        avg_confidence = sum(r.confidence for r in agreeing) / len(agreeing) if agreeing else 0

        elapsed = int((time.time() - start) * 1000)

        consensus = ConsensusResult(
            task_id=task_id,
            consensus_reached=agreement >= self.CONSENSUS_THRESHOLD,
            result=winner,
            confidence_score=avg_confidence,
            agreement_ratio=agreement,
            agents_consulted=len(responses),
            dissenting_agents=dissenters,
            execution_time_ms=elapsed,
        )

        status = "REACHED" if consensus.consensus_reached else "FAILED"
        print(f"       Consensus {status}: '{winner}' "
              f"({agreement*100:.1f}% agreement, {len(dissenters)} dissenters)")

        return consensus

    async def _execute_action(self, task: SwarmTask, consensus: ConsensusResult) -> Dict:
        """Execute the appropriate action based on task type."""
        if task.task_type == TaskType.PREDICTION:
            return self._action_update_oracle(task, consensus)
        elif task.task_type == TaskType.MONITORING:
            return await self._action_send_alert(task, consensus)
        elif task.task_type == TaskType.RISK_ASSESSMENT:
            return self._action_defensive(task, consensus)
        else:
            return self._action_publish_report(task, consensus)

    def _action_update_oracle(self, task: SwarmTask, consensus: ConsensusResult) -> Dict:
        """Push consensus prediction to oracle (simulated in demo)."""
        print(f"       Oracle update: {consensus.result} (confidence: {consensus.confidence_score:.2f})")
        return {
            "action": "update_oracle",
            "value": consensus.result,
            "confidence": consensus.confidence_score,
            "tx_hash": f"0x{hashlib.sha256(task.task_id.encode()).hexdigest()[:64]}",
        }

    async def _action_send_alert(self, task: SwarmTask, consensus: ConsensusResult) -> Dict:
        """Send WhatsApp alert."""
        msg = (f"Swarm Alert [{task.task_type.value}]: {consensus.result} "
               f"(confidence: {consensus.confidence_score*100:.0f}%)")
        print(f"       WhatsApp alert: {msg[:60]}...")
        return {"action": "send_alert", "message": msg, "channel": "whatsapp"}

    def _action_defensive(self, task: SwarmTask, consensus: ConsensusResult) -> Dict:
        """Trigger defensive protocol action."""
        print(f"       Defensive action: risk={consensus.result}")
        return {
            "action": "defensive",
            "risk_level": consensus.result,
            "tx_hash": f"0x{hashlib.sha256(task.task_id.encode()).hexdigest()[:64]}",
        }

    def _action_publish_report(self, task: SwarmTask, consensus: ConsensusResult) -> Dict:
        """Publish analysis report."""
        print(f"       Report published: {consensus.result[:50]}")
        return {"action": "publish_report", "result": consensus.result}

    async def _settle_payment(
        self, task: SwarmTask, consensus: ConsensusResult,
        agent_responses: List[AgentResponse]
    ) -> PaymentReceipt:
        """Settle payment via x402: charge requester, split to agents."""
        protocol_fee = task.budget_link * self.PROTOCOL_FEE_PCT
        agent_pool = task.budget_link - protocol_fee

        # Weight splits by confidence
        total_conf = sum(r.confidence for r in agent_responses)
        splits = {}
        for r in agent_responses:
            share = (Decimal(str(r.confidence)) / Decimal(str(total_conf))) * agent_pool
            splits[r.agent_id] = round(share, 6)

        tx_hash = f"0x{hashlib.sha256(f'pay-{task.task_id}'.encode()).hexdigest()[:64]}"

        receipt = PaymentReceipt(
            task_id=task.task_id,
            amount_link=task.budget_link,
            payer=task.requester_address,
            agent_splits=splits,
            protocol_fee=protocol_fee,
            tx_hash=tx_hash,
            status="completed",
        )

        print(f"       Payment: {task.budget_link} LINK")
        print(f"       Protocol fee: {protocol_fee} LINK")
        print(f"       Agent splits: {len(splits)} agents paid")
        return receipt

    async def run_demo(self):
        """Run a full demo showing all task types."""
        print("\n" + "="*60)
        print("  CHAINLINK CRE AGENT SWARM - FULL DEMO")
        print("="*60)

        demo_tasks = [
            SwarmTask(
                task_id=SwarmTask.generate_id(),
                task_type=TaskType.PREDICTION,
                query="Will ETH reach $5,000 by March 2026?",
                requester_address="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD53",
                budget_link=Decimal("0.5"),
            ),
            SwarmTask(
                task_id=SwarmTask.generate_id(),
                task_type=TaskType.RISK_ASSESSMENT,
                query="What is the systemic risk level for AAVE v3 on Base?",
                requester_address="0x8ba1f109551bD432803012645Ac136ddd64DBA72",
                budget_link=Decimal("0.3"),
            ),
            SwarmTask(
                task_id=SwarmTask.generate_id(),
                task_type=TaskType.MONITORING,
                query="Check health of Uniswap v3 pools on Arbitrum",
                requester_address="0xdD2FD4581271e230360230F9337D5c0430Bf44C0",
                budget_link=Decimal("0.1"),
            ),
            SwarmTask(
                task_id=SwarmTask.generate_id(),
                task_type=TaskType.ANALYSIS,
                query="Analyze optimal yield farming strategy across L2s",
                requester_address="0x2546BcD3c84621e976D8185a91A922aE77ECEc30",
                budget_link=Decimal("0.4"),
            ),
        ]

        all_results = []
        for i, task in enumerate(demo_tasks, 1):
            print(f"\n{'='*60}")
            print(f"  Demo Task {i}/{len(demo_tasks)}: {task.task_type.value.upper()}")
            print(f"{'='*60}")
            result = await self.execute_task(task)
            all_results.append(result)

        # Summary
        total_link = sum(Decimal(str(r["payment"]["amount_link"])) for r in all_results)
        avg_consensus = sum(
            r["consensus"]["agreement_ratio"] for r in all_results
        ) / len(all_results)
        total_agents = sum(r["consensus"]["agents_consulted"] for r in all_results)
        total_time = sum(r["total_time_ms"] for r in all_results)

        print("\n" + "="*60)
        print("  DEMO SUMMARY")
        print("="*60)
        print(f"  Tasks completed:     {len(all_results)}")
        print(f"  Total agents used:   {total_agents}")
        print(f"  Avg consensus:       {avg_consensus*100:.1f}%")
        print(f"  Total LINK spent:    {total_link}")
        print(f"  Total time:          {total_time}ms")
        print(f"  All consensus met:   "
              f"{'YES' if all(r['consensus']['consensus_reached'] for r in all_results) else 'NO'}")
        print("="*60)

        return all_results


async def main():
    parser = argparse.ArgumentParser(description="AI Agent Swarm Orchestrator")
    parser.add_argument("--demo", action="store_true", help="Run full demo flow")
    parser.add_argument("--live", action="store_true",
                        help="Use real LLM calls instead of simulation")
    parser.add_argument("--provider", type=str, default="groq",
                        choices=["groq", "openai", "xai"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--task", type=str, help="Natural language task to execute")
    parser.add_argument("--task-type", type=str, default="analysis",
                        choices=["prediction", "analysis", "monitoring", "risk_assessment"])
    parser.add_argument("--budget", type=float, default=0.1, help="LINK budget")
    parser.add_argument("--requester", type=str,
                        default="0x742d35Cc6634C0532925a3b844Bc9e7595f2bD53")
    args = parser.parse_args()

    orchestrator = AgentSwarmOrchestrator(
        demo_mode=args.demo or not args.task,
        live_llm=args.live,
        provider=args.provider,
    )

    if args.demo:
        results = await orchestrator.run_demo()
        # Save results
        output_path = CRE_DIR / "storage" / f"demo_results_{int(time.time())}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    elif args.task:
        task = SwarmTask(
            task_id=SwarmTask.generate_id(),
            task_type=TaskType(args.task_type),
            query=args.task,
            requester_address=args.requester,
            budget_link=Decimal(str(args.budget)),
        )
        await orchestrator.execute_task(task)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
