#!/usr/bin/env python3
"""
x402 Payment Handler for Chainlink CRE Workflows
@requirement: REQ-CRE-013 - x402 Payment Standard Integration
@component: Automated Micropayment Processing for CRE Workflows
@integration: MetaMask SDK, LINK token, Chainlink CRE Framework
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os
from decimal import Decimal, getcontext
from enum import Enum

# Set decimal precision for financial calculations
getcontext().prec = 18

# Add ETAC system to path
sys.path.insert(0, '/home/craigmbrown/Project/ETAC-System')

# Import ETAC core modules
try:
    from strategic_platform.core.security_infrastructure_logger import SecurityInfrastructureLogger
    from strategic_platform.core.base_level_properties import BaseLevelPropertyManager, PropertyCategory
    from strategic_platform.core.agent_core_directives import (
        UniversalAgentDirective, AgentProperty, create_etac_agent, improve_agent_for_requirement
    )
    from strategic_platform.utils.enhanced_exception_handler import (
        enhanced_exception_handler, ETACException, ETACAPIError, ETACSecurityError
    )
    print("✅ Successfully imported ETAC core modules")
except ImportError as e:
    print(f"⚠️ Could not import ETAC modules: {e}")
    # Fallback implementations (same as workflow framework)

class PaymentStatus(Enum):
    """Payment processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    EXPIRED = "expired"

class ChannelStatus(Enum):
    """Payment channel status enumeration"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    DEPLETED = "depleted"

@dataclass
class X402PaymentRequest:
    """
    @requirement: REQ-CRE-014 - x402 Payment Request Structure
    Standardized payment request for CRE workflow execution
    """
    request_id: str
    workflow_id: str
    payer_address: str
    receiver_address: str
    amount_link: Decimal
    currency: str = "LINK"
    payment_type: str = "per_call"  # per_call, subscription, performance_based
    metadata: Dict[str, Any] = None
    expiry_timestamp: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expiry_timestamp is None:
            # Default 24-hour expiry
            self.expiry_timestamp = self.created_at + timedelta(hours=24)
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_expired(self) -> bool:
        """Check if payment request has expired"""
        return datetime.now() > self.expiry_timestamp
    
    @property
    def amount_wei(self) -> int:
        """Convert LINK amount to wei (18 decimal places)"""
        return int(self.amount_link * Decimal(10**18))

@dataclass
class X402PaymentChannel:
    """
    @requirement: REQ-CRE-015 - x402 Payment Channel Management
    Manages state channel for repeated micropayments
    """
    channel_id: str
    workflow_id: str
    payer_address: str
    receiver_address: str
    token_contract: str  # LINK token contract address
    initial_balance: Decimal
    current_balance: Decimal
    reserved_balance: Decimal = Decimal(0)
    payment_count: int = 0
    total_paid: Decimal = Decimal(0)
    last_payment_timestamp: Optional[datetime] = None
    channel_nonce: int = 0
    status: ChannelStatus = ChannelStatus.ACTIVE
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    @property
    def available_balance(self) -> Decimal:
        """Calculate available balance for payments"""
        return self.current_balance - self.reserved_balance
    
    @property
    def utilization_rate(self) -> float:
        """Calculate channel utilization rate"""
        if self.initial_balance <= 0:
            return 0.0
        return float(self.total_paid / self.initial_balance)
    
    def can_pay(self, amount: Decimal) -> bool:
        """Check if channel has sufficient balance for payment"""
        return self.available_balance >= amount and self.status == ChannelStatus.ACTIVE

@dataclass
class X402PaymentTransaction:
    """
    @requirement: REQ-CRE-016 - x402 Payment Transaction Record
    Individual payment transaction within a channel
    """
    tx_id: str
    channel_id: str
    request_id: str
    amount_link: Decimal
    gas_fee: Decimal = Decimal(0)
    status: PaymentStatus = PaymentStatus.PENDING
    blockchain_tx_hash: Optional[str] = None
    confirmation_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = None
    confirmed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost including gas"""
        return self.amount_link + self.gas_fee
    
    @property
    def is_confirmed(self) -> bool:
        """Check if transaction is confirmed on blockchain"""
        return self.confirmation_count >= 3  # 3 confirmations required

class X402PaymentHandler:
    """
    @requirement: REQ-CRE-017 - x402 Payment Handler Implementation
    Core payment processing system for Chainlink CRE workflows
    """
    
    def __init__(self, project_root: str = "/home/craigmbrown/Project/chainlink-prediction-markets-mcp"):
        self.project_root = Path(project_root)
        
        # Initialize ETAC infrastructure
        self.logger = SecurityInfrastructureLogger()
        self.blp_manager = BaseLevelPropertyManager()
        self.agent = create_etac_agent("X402PaymentHandler", "payment_processor")
        
        # Payment processing state
        self.active_channels: Dict[str, X402PaymentChannel] = {}
        self.payment_requests: Dict[str, X402PaymentRequest] = {}
        self.transactions: Dict[str, X402PaymentTransaction] = {}
        
        # Configuration
        self.handler_id = hashlib.md5(f"x402_handler_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        self.initialized_at = datetime.now()
        
        # Blockchain configuration
        self.network_config = {
            "ethereum_mainnet": {
                "chain_id": 1,
                "link_token": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
                "gas_price_gwei": 20,
                "confirmation_blocks": 3
            },
            "ethereum_sepolia": {
                "chain_id": 11155111,
                "link_token": "0x779877A7B0D9E8603169DdbD7836e478b4624789",
                "gas_price_gwei": 10,
                "confirmation_blocks": 3
            },
            "polygon": {
                "chain_id": 137,
                "link_token": "0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39",
                "gas_price_gwei": 50,
                "confirmation_blocks": 5
            }
        }
        
        self.current_network = "ethereum_sepolia"  # Default to testnet
        
        # Storage setup
        self.storage_dir = self.project_root / "chainlink_cre" / "payments"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_payment_data()
        
        self.logger.log_security_event(
            "x402_payment_handler",
            "INITIALIZED",
            {
                "handler_id": self.handler_id,
                "network": self.current_network,
                "active_channels": len(self.active_channels),
                "agent_id": self.agent.agent_id
            }
        )
        
        print(f"✅ x402 Payment Handler initialized - {len(self.active_channels)} channels loaded")

    @enhanced_exception_handler(retry_attempts=2, component_name="X402PaymentHandler")
    async def create_payment_channel(
        self,
        workflow_id: str,
        payer_address: str,
        receiver_address: str,
        initial_balance: Decimal,
        metadata: Dict[str, Any] = None
    ) -> X402PaymentChannel:
        """
        @requirement: REQ-CRE-018 - Create Payment Channel
        Create a new x402 payment channel for recurring micropayments
        """
        
        print(f"💳 Creating payment channel for workflow: {workflow_id}")
        
        # Validate addresses
        if not self._validate_ethereum_address(payer_address):
            raise ETACSecurityError(f"Invalid payer address: {payer_address}")
        
        if not self._validate_ethereum_address(receiver_address):
            raise ETACSecurityError(f"Invalid receiver address: {receiver_address}")
        
        # Generate channel ID
        channel_data = f"{workflow_id}_{payer_address}_{receiver_address}_{datetime.now().isoformat()}"
        channel_id = hashlib.sha256(channel_data.encode()).hexdigest()[:32]
        
        # Create channel
        channel = X402PaymentChannel(
            channel_id=channel_id,
            workflow_id=workflow_id,
            payer_address=payer_address,
            receiver_address=receiver_address,
            token_contract=self.network_config[self.current_network]["link_token"],
            initial_balance=initial_balance,
            current_balance=initial_balance
        )
        
        # Store channel
        self.active_channels[channel_id] = channel
        
        # Update agent properties
        self.agent.update_property(AgentProperty.SELF_ORGANIZATION, 0.001, f"channel_creation_{workflow_id}")
        
        # Log channel creation
        self.logger.log_security_event(
            "payment_channel",
            "CREATED",
            {
                "channel_id": channel_id,
                "workflow_id": workflow_id,
                "initial_balance": str(initial_balance),
                "network": self.current_network
            }
        )
        
        # Save data
        self._save_payment_data()
        
        print(f"✅ Payment channel created: {channel_id} ({initial_balance} LINK)")
        return channel

    @enhanced_exception_handler(retry_attempts=3, component_name="X402PaymentHandler")
    async def process_payment(
        self,
        payment_request: X402PaymentRequest,
        channel_id: Optional[str] = None
    ) -> X402PaymentTransaction:
        """
        @requirement: REQ-CRE-019 - Process x402 Payment
        Process a payment request through the x402 standard
        """
        
        print(f"💰 Processing payment: {payment_request.amount_link} LINK")
        
        # Validate payment request
        if payment_request.is_expired:
            raise ETACAPIError(f"Payment request expired: {payment_request.request_id}")
        
        # Store payment request
        self.payment_requests[payment_request.request_id] = payment_request
        
        # Generate transaction ID
        tx_data = f"{payment_request.request_id}_{datetime.now().isoformat()}"
        tx_id = hashlib.sha256(tx_data.encode()).hexdigest()[:32]
        
        try:
            if channel_id and channel_id in self.active_channels:
                # Process through existing channel
                transaction = await self._process_channel_payment(payment_request, channel_id, tx_id)
            else:
                # Process as standalone transaction
                transaction = await self._process_standalone_payment(payment_request, tx_id)
            
            # Update agent properties for successful payment
            self.agent.update_property(AgentProperty.AUTONOMY, 0.001, f"payment_processing_{payment_request.payment_type}")
            self.agent.record_operation(True, "payment_processing")
            
            return transaction
            
        except Exception as e:
            # Create failed transaction record
            failed_tx = X402PaymentTransaction(
                tx_id=tx_id,
                channel_id=channel_id or "standalone",
                request_id=payment_request.request_id,
                amount_link=payment_request.amount_link,
                status=PaymentStatus.FAILED,
                error_message=str(e)
            )
            
            self.transactions[tx_id] = failed_tx
            self.agent.record_operation(False, "payment_processing")
            
            self.logger.log_security_event(
                "payment_processing",
                "FAILED",
                {
                    "tx_id": tx_id,
                    "request_id": payment_request.request_id,
                    "error": str(e)
                }
            )
            
            self._save_payment_data()
            raise

    async def _process_channel_payment(
        self,
        payment_request: X402PaymentRequest,
        channel_id: str,
        tx_id: str
    ) -> X402PaymentTransaction:
        """Process payment through existing channel"""
        
        channel = self.active_channels[channel_id]
        
        # Validate channel can process payment
        if not channel.can_pay(payment_request.amount_link):
            if channel.status != ChannelStatus.ACTIVE:
                raise ETACAPIError(f"Channel {channel_id} is not active: {channel.status.value}")
            else:
                raise ETACAPIError(f"Insufficient channel balance: {channel.available_balance} < {payment_request.amount_link}")
        
        # Reserve balance
        channel.reserved_balance += payment_request.amount_link
        channel.channel_nonce += 1
        
        try:
            # Create transaction
            transaction = X402PaymentTransaction(
                tx_id=tx_id,
                channel_id=channel_id,
                request_id=payment_request.request_id,
                amount_link=payment_request.amount_link,
                status=PaymentStatus.PROCESSING
            )
            
            # Simulate blockchain transaction
            blockchain_result = await self._simulate_blockchain_transaction(transaction)
            
            if blockchain_result["success"]:
                # Update channel state
                channel.current_balance -= payment_request.amount_link
                channel.reserved_balance -= payment_request.amount_link
                channel.total_paid += payment_request.amount_link
                channel.payment_count += 1
                channel.last_payment_timestamp = datetime.now()
                channel.updated_at = datetime.now()
                
                # Update transaction
                transaction.status = PaymentStatus.COMPLETED
                transaction.blockchain_tx_hash = blockchain_result["tx_hash"]
                transaction.gas_fee = Decimal(blockchain_result["gas_fee"])
                transaction.confirmed_at = datetime.now()
                transaction.confirmation_count = 3  # Simulate confirmations
                
                # Check if channel is depleted
                if channel.current_balance < Decimal("0.01"):  # Less than 0.01 LINK
                    channel.status = ChannelStatus.DEPLETED
                
                self.logger.log_security_event(
                    "channel_payment",
                    "COMPLETED",
                    {
                        "tx_id": tx_id,
                        "channel_id": channel_id,
                        "amount": str(payment_request.amount_link),
                        "remaining_balance": str(channel.current_balance)
                    }
                )
                
                print(f"✅ Channel payment completed: {tx_id} ({payment_request.amount_link} LINK)")
                
            else:
                # Release reserved balance on failure
                channel.reserved_balance -= payment_request.amount_link
                transaction.status = PaymentStatus.FAILED
                transaction.error_message = blockchain_result["error"]
                
                raise ETACAPIError(f"Blockchain transaction failed: {blockchain_result['error']}")
            
            # Store transaction
            self.transactions[tx_id] = transaction
            
            # Update channel in storage
            self.active_channels[channel_id] = channel
            
            # Save all data
            self._save_payment_data()
            
            return transaction
            
        except Exception as e:
            # Release reserved balance on any failure
            channel.reserved_balance = max(Decimal(0), channel.reserved_balance - payment_request.amount_link)
            raise

    async def _process_standalone_payment(
        self,
        payment_request: X402PaymentRequest,
        tx_id: str
    ) -> X402PaymentTransaction:
        """Process standalone payment without channel"""
        
        # Create transaction
        transaction = X402PaymentTransaction(
            tx_id=tx_id,
            channel_id="standalone",
            request_id=payment_request.request_id,
            amount_link=payment_request.amount_link,
            status=PaymentStatus.PROCESSING
        )
        
        # Simulate blockchain transaction
        blockchain_result = await self._simulate_blockchain_transaction(transaction)
        
        if blockchain_result["success"]:
            transaction.status = PaymentStatus.COMPLETED
            transaction.blockchain_tx_hash = blockchain_result["tx_hash"]
            transaction.gas_fee = Decimal(blockchain_result["gas_fee"])
            transaction.confirmed_at = datetime.now()
            transaction.confirmation_count = 3
            
            self.logger.log_security_event(
                "standalone_payment",
                "COMPLETED",
                {
                    "tx_id": tx_id,
                    "amount": str(payment_request.amount_link),
                    "tx_hash": blockchain_result["tx_hash"]
                }
            )
            
            print(f"✅ Standalone payment completed: {tx_id} ({payment_request.amount_link} LINK)")
            
        else:
            transaction.status = PaymentStatus.FAILED
            transaction.error_message = blockchain_result["error"]
            
            raise ETACAPIError(f"Standalone payment failed: {blockchain_result['error']}")
        
        # Store transaction
        self.transactions[tx_id] = transaction
        self._save_payment_data()
        
        return transaction

    async def _simulate_blockchain_transaction(self, transaction: X402PaymentTransaction) -> Dict[str, Any]:
        """
        @requirement: REQ-CRE-020 - Simulate Blockchain Interaction
        Simulate blockchain transaction processing (mock for development)
        """
        
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # Mock transaction success (95% success rate)
        import random
        if random.random() > 0.05:
            # Calculate gas fee
            network_config = self.network_config[self.current_network]
            gas_limit = 65000  # Typical LINK transfer gas limit
            gas_fee = Decimal(network_config["gas_price_gwei"]) * Decimal(gas_limit) / Decimal(10**9)
            
            # Generate mock transaction hash
            tx_data = f"{transaction.tx_id}_{datetime.now().isoformat()}"
            tx_hash = "0x" + hashlib.sha256(tx_data.encode()).hexdigest()
            
            return {
                "success": True,
                "tx_hash": tx_hash,
                "gas_fee": float(gas_fee),
                "block_number": random.randint(18000000, 19000000),
                "gas_used": gas_limit
            }
        else:
            error_messages = [
                "Insufficient gas",
                "Network congestion",
                "Invalid nonce",
                "Token transfer failed",
                "RPC timeout"
            ]
            return {
                "success": False,
                "error": random.choice(error_messages)
            }

    def get_channel_status(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        @requirement: REQ-CRE-021 - Get Channel Status
        Retrieve detailed status of a payment channel
        """
        
        if channel_id not in self.active_channels:
            return None
        
        channel = self.active_channels[channel_id]
        
        # Get channel transactions
        channel_transactions = [
            tx for tx in self.transactions.values() 
            if tx.channel_id == channel_id
        ]
        
        # Calculate statistics
        successful_payments = len([tx for tx in channel_transactions if tx.status == PaymentStatus.COMPLETED])
        failed_payments = len([tx for tx in channel_transactions if tx.status == PaymentStatus.FAILED])
        
        status = {
            "channel_info": {
                "channel_id": channel.channel_id,
                "workflow_id": channel.workflow_id,
                "payer_address": channel.payer_address,
                "receiver_address": channel.receiver_address,
                "status": channel.status.value,
                "created_at": channel.created_at.isoformat(),
                "updated_at": channel.updated_at.isoformat()
            },
            "balances": {
                "initial_balance": str(channel.initial_balance),
                "current_balance": str(channel.current_balance),
                "reserved_balance": str(channel.reserved_balance),
                "available_balance": str(channel.available_balance),
                "total_paid": str(channel.total_paid),
                "utilization_rate": channel.utilization_rate
            },
            "statistics": {
                "payment_count": channel.payment_count,
                "successful_payments": successful_payments,
                "failed_payments": failed_payments,
                "success_rate": successful_payments / max(1, channel.payment_count),
                "last_payment": channel.last_payment_timestamp.isoformat() if channel.last_payment_timestamp else None
            },
            "network_info": {
                "network": self.current_network,
                "token_contract": channel.token_contract,
                "channel_nonce": channel.channel_nonce
            }
        }
        
        return status

    def get_payment_analytics(self) -> Dict[str, Any]:
        """
        @requirement: REQ-CRE-022 - Payment Analytics Dashboard
        Generate comprehensive analytics for payment processing
        """
        
        analytics = {
            "handler_info": {
                "handler_id": self.handler_id,
                "initialized_at": self.initialized_at.isoformat(),
                "current_network": self.current_network,
                "total_channels": len(self.active_channels),
                "total_transactions": len(self.transactions)
            },
            "channel_metrics": {
                "active_channels": len([ch for ch in self.active_channels.values() if ch.status == ChannelStatus.ACTIVE]),
                "depleted_channels": len([ch for ch in self.active_channels.values() if ch.status == ChannelStatus.DEPLETED]),
                "suspended_channels": len([ch for ch in self.active_channels.values() if ch.status == ChannelStatus.SUSPENDED]),
                "total_channel_balance": sum(ch.current_balance for ch in self.active_channels.values()),
                "total_channel_volume": sum(ch.total_paid for ch in self.active_channels.values()),
                "average_utilization": sum(ch.utilization_rate for ch in self.active_channels.values()) / max(1, len(self.active_channels))
            },
            "transaction_metrics": {
                "completed_transactions": len([tx for tx in self.transactions.values() if tx.status == PaymentStatus.COMPLETED]),
                "failed_transactions": len([tx for tx in self.transactions.values() if tx.status == PaymentStatus.FAILED]),
                "pending_transactions": len([tx for tx in self.transactions.values() if tx.status == PaymentStatus.PENDING]),
                "total_volume": sum(tx.amount_link for tx in self.transactions.values() if tx.status == PaymentStatus.COMPLETED),
                "total_gas_fees": sum(tx.gas_fee for tx in self.transactions.values() if tx.status == PaymentStatus.COMPLETED),
                "success_rate": 0.0
            },
            "agent_performance": self.agent.get_agent_summary(),
            "generated_at": datetime.now().isoformat()
        }
        
        # Calculate success rate
        total_final_transactions = len([tx for tx in self.transactions.values() if tx.status in [PaymentStatus.COMPLETED, PaymentStatus.FAILED]])
        if total_final_transactions > 0:
            analytics["transaction_metrics"]["success_rate"] = analytics["transaction_metrics"]["completed_transactions"] / total_final_transactions
        
        return analytics

    def _validate_ethereum_address(self, address: str) -> bool:
        """Validate Ethereum address format"""
        if not address.startswith("0x"):
            return False
        if len(address) != 42:
            return False
        try:
            int(address[2:], 16)
            return True
        except ValueError:
            return False

    def _save_payment_data(self):
        """Save payment handler data to persistent storage"""
        
        payment_data = {
            "handler_info": {
                "handler_id": self.handler_id,
                "initialized_at": self.initialized_at.isoformat(),
                "current_network": self.current_network,
                "agent_id": self.agent.agent_id
            },
            "channels": {
                channel_id: {
                    **asdict(channel),
                    "created_at": channel.created_at.isoformat(),
                    "updated_at": channel.updated_at.isoformat(),
                    "last_payment_timestamp": channel.last_payment_timestamp.isoformat() if channel.last_payment_timestamp else None,
                    "status": channel.status.value,
                    "initial_balance": str(channel.initial_balance),
                    "current_balance": str(channel.current_balance),
                    "reserved_balance": str(channel.reserved_balance),
                    "total_paid": str(channel.total_paid)
                }
                for channel_id, channel in self.active_channels.items()
            },
            "transactions": {
                tx_id: {
                    **asdict(transaction),
                    "created_at": transaction.created_at.isoformat(),
                    "confirmed_at": transaction.confirmed_at.isoformat() if transaction.confirmed_at else None,
                    "status": transaction.status.value,
                    "amount_link": str(transaction.amount_link),
                    "gas_fee": str(transaction.gas_fee)
                }
                for tx_id, transaction in list(self.transactions.items())[-500:]  # Last 500 transactions
            },
            "requests": {
                req_id: {
                    **asdict(request),
                    "created_at": request.created_at.isoformat(),
                    "expiry_timestamp": request.expiry_timestamp.isoformat(),
                    "amount_link": str(request.amount_link)
                }
                for req_id, request in list(self.payment_requests.items())[-200:]  # Last 200 requests
            },
            "saved_at": datetime.now().isoformat()
        }
        
        storage_file = self.storage_dir / f"x402_payments_{self.handler_id}.json"
        
        try:
            with open(storage_file, 'w') as f:
                json.dump(payment_data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save payment data: {str(e)}")

    def _load_payment_data(self):
        """Load payment handler data from persistent storage"""
        
        # Find the most recent payment data file
        payment_files = list(self.storage_dir.glob("x402_payments_*.json"))
        if not payment_files:
            return
        
        latest_file = max(payment_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Load channels
            for channel_id, channel_data in data.get("channels", {}).items():
                # Convert string dates back to datetime
                channel_data["created_at"] = datetime.fromisoformat(channel_data["created_at"])
                channel_data["updated_at"] = datetime.fromisoformat(channel_data["updated_at"])
                if channel_data["last_payment_timestamp"]:
                    channel_data["last_payment_timestamp"] = datetime.fromisoformat(channel_data["last_payment_timestamp"])
                else:
                    channel_data["last_payment_timestamp"] = None
                
                # Convert string amounts back to Decimal
                channel_data["initial_balance"] = Decimal(channel_data["initial_balance"])
                channel_data["current_balance"] = Decimal(channel_data["current_balance"])
                channel_data["reserved_balance"] = Decimal(channel_data["reserved_balance"])
                channel_data["total_paid"] = Decimal(channel_data["total_paid"])
                
                # Convert status string back to enum
                channel_data["status"] = ChannelStatus(channel_data["status"])
                
                channel = X402PaymentChannel(**channel_data)
                self.active_channels[channel_id] = channel
            
            # Load transactions
            for tx_id, tx_data in data.get("transactions", {}).items():
                # Convert string dates back to datetime
                tx_data["created_at"] = datetime.fromisoformat(tx_data["created_at"])
                if tx_data["confirmed_at"]:
                    tx_data["confirmed_at"] = datetime.fromisoformat(tx_data["confirmed_at"])
                else:
                    tx_data["confirmed_at"] = None
                
                # Convert string amounts back to Decimal
                tx_data["amount_link"] = Decimal(tx_data["amount_link"])
                tx_data["gas_fee"] = Decimal(tx_data["gas_fee"])
                
                # Convert status string back to enum
                tx_data["status"] = PaymentStatus(tx_data["status"])
                
                transaction = X402PaymentTransaction(**tx_data)
                self.transactions[tx_id] = transaction
            
            # Load payment requests
            for req_id, req_data in data.get("requests", {}).items():
                # Convert string dates back to datetime
                req_data["created_at"] = datetime.fromisoformat(req_data["created_at"])
                req_data["expiry_timestamp"] = datetime.fromisoformat(req_data["expiry_timestamp"])
                
                # Convert string amount back to Decimal
                req_data["amount_link"] = Decimal(req_data["amount_link"])
                
                request = X402PaymentRequest(**req_data)
                self.payment_requests[req_id] = request
            
            print(f"✅ Loaded payment data from {latest_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to load payment data: {str(e)}")


async def main():
    """
    @requirement: REQ-CRE-023 - Main x402 Payment Handler Testing
    Test the x402 Payment Handler functionality
    """
    print("\n💳 Testing x402 Payment Handler")
    print("=" * 80)
    
    try:
        # Initialize payment handler
        handler = X402PaymentHandler()
        
        # Test payment channel creation
        print(f"\n🧪 Testing Payment Channel Creation")
        
        channel = await handler.create_payment_channel(
            workflow_id="test_workflow_001",
            payer_address="0x742D35Cc6678d5c6e7b3c3F4F6e52738a8c1D5d3",
            receiver_address="0x1234567890123456789012345678901234567890",
            initial_balance=Decimal("10.0"),  # 10 LINK
            metadata={"purpose": "testing", "created_by": "test_suite"}
        )
        
        print(f"   ✅ Channel created: {channel.channel_id}")
        print(f"   💰 Initial balance: {channel.initial_balance} LINK")
        print(f"   📊 Available: {channel.available_balance} LINK")
        
        # Test payment processing
        print(f"\n💰 Testing Payment Processing")
        
        # Create payment request
        payment_request = X402PaymentRequest(
            request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            workflow_id="test_workflow_001",
            payer_address="0x742D35Cc6678d5c6e7b3c3F4F6e52738a8c1D5d3",
            receiver_address="0x1234567890123456789012345678901234567890",
            amount_link=Decimal("0.5"),  # 0.5 LINK
            payment_type="per_call",
            metadata={"service": "AI prediction", "complexity": "basic"}
        )
        
        # Process payment
        transaction = await handler.process_payment(payment_request, channel.channel_id)
        
        print(f"   ✅ Payment processed: {transaction.tx_id}")
        print(f"   💳 Amount: {transaction.amount_link} LINK")
        print(f"   ⛽ Gas fee: {transaction.gas_fee} ETH")
        print(f"   🔗 TX Hash: {transaction.blockchain_tx_hash}")
        print(f"   ⏱️ Status: {transaction.status.value}")
        
        # Test standalone payment
        print(f"\n🔄 Testing Standalone Payment")
        
        standalone_request = X402PaymentRequest(
            request_id=f"standalone_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            workflow_id="standalone_workflow",
            payer_address="0x742D35Cc6678d5c6e7b3c3F4F6e52738a8c1D5d3",
            receiver_address="0x1234567890123456789012345678901234567890",
            amount_link=Decimal("1.0"),  # 1 LINK
            payment_type="one_time"
        )
        
        standalone_tx = await handler.process_payment(standalone_request)
        print(f"   ✅ Standalone payment: {standalone_tx.tx_id} ({standalone_tx.status.value})")
        
        # Get channel status
        print(f"\n📊 Channel Status Report")
        channel_status = handler.get_channel_status(channel.channel_id)
        
        print(f"   Channel ID: {channel_status['channel_info']['channel_id']}")
        print(f"   Status: {channel_status['channel_info']['status']}")
        print(f"   Remaining Balance: {channel_status['balances']['current_balance']} LINK")
        print(f"   Utilization: {channel_status['balances']['utilization_rate']:.1%}")
        print(f"   Success Rate: {channel_status['statistics']['success_rate']:.1%}")
        
        # Get payment analytics
        print(f"\n📈 Payment Analytics Dashboard")
        analytics = handler.get_payment_analytics()
        
        print(f"   Total Channels: {analytics['channel_metrics']['active_channels']}")
        print(f"   Total Transactions: {analytics['transaction_metrics']['completed_transactions']}")
        print(f"   Success Rate: {analytics['transaction_metrics']['success_rate']:.1%}")
        print(f"   Total Volume: {analytics['transaction_metrics']['total_volume']} LINK")
        print(f"   Agent Performance: {analytics['agent_performance']['performance']['average_property_score']:.3f}")
        
        print(f"\n✅ x402 Payment Handler operational and ready!")
        return analytics
        
    except Exception as e:
        print(f"❌ x402 Payment Handler test failed: {str(e)}")
        import traceback

# Claude Advanced Tools Integration
try:
    from claude_advanced_tools import ToolRegistry, ToolSearchEngine, PatternLearner
    ADVANCED_TOOLS_ENABLED = True
except ImportError:
    ADVANCED_TOOLS_ENABLED = False

        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the x402 Payment Handler test
    result = asyncio.run(main())
    
    # Output JSON result for integration
    print(f"\n📤 Payment Handler Test Result:")
    print(json.dumps(result, indent=2, default=str))