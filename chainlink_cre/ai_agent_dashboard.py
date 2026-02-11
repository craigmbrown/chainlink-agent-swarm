#!/usr/bin/env python3
"""
AI Agent dApps Dashboard Generator
Comprehensive monitoring dashboard for Chainlink AI agent dApps with revenue tracking
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
import requests
import logging
from pathlib import Path

# Import the AI agent dApps system
from ai_agent_dapps import ChainlinkAIAgentDApps, AgentType, RevenueModel

logger = logging.getLogger(__name__)

class AIAgentDashboard:
    """Dashboard generator for AI agent dApps analytics"""
    
    def __init__(self, ai_dapps: ChainlinkAIAgentDApps):
        self.ai_dapps = ai_dapps
        self.output_dir = Path("/home/craigmbrown/Project/html-output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_comprehensive_dashboard(self) -> str:
        """Generate comprehensive AI agent dApps dashboard"""
        try:
            # Get system data
            overview = await self.ai_dapps.get_system_overview()
            revenue_report = await self.ai_dapps.generate_revenue_report()
            
            # Dashboard data
            dashboard_data = {
                'overview': overview,
                'revenue_report': revenue_report,
                'timestamp': datetime.now().isoformat(),
                'total_agents': len(self.ai_dapps.active_agents),
                'total_revenue': float(self.ai_dapps.total_revenue)
            }
            
            # Generate HTML
            html_content = self._generate_dashboard_html(dashboard_data)
            
            # Save file
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ai_agent_dapps_dashboard.html"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Create TinyURL and send notification
            await self._create_tinyurl_and_notify(filename, "AI Agent dApps Dashboard", dashboard_data)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            raise
    
    def _generate_dashboard_html(self, data: dict) -> str:
        """Generate HTML dashboard content"""
        
        overview = data['overview']
        revenue_report = data['revenue_report']
        timestamp = data['timestamp']
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chainlink AI Agent dApps Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #e0e0e0;
            background: #1a1a1a;
            margin: 0;
            padding: 20px;
        }}
        
        article {{
            max-width: 1200px;
            margin: 0 auto;
            background: #2d2d2d;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #4a9eff;
        }}
        
        h1 {{
            color: #4a9eff;
            font-size: 2.5em;
            margin: 0 0 10px 0;
            text-shadow: 0 0 10px rgba(74, 158, 255, 0.3);
        }}
        
        .subtitle {{
            color: #b0b0b0;
            font-size: 1.1em;
            margin: 0;
        }}
        
        .timestamp {{
            color: #808080;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: #1a1a1a;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card:hover {{
            background: #3a3a3a;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(74, 158, 255, 0.2);
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #4a9eff, #79c0ff);
        }}
        
        .metric-title {{
            color: #4a9eff;
            font-size: 1.1em;
            font-weight: 600;
            margin: 0 0 10px 0;
        }}
        
        .metric-value {{
            color: #e0e0e0;
            font-size: 2em;
            font-weight: bold;
            margin: 0 0 5px 0;
        }}
        
        .metric-label {{
            color: #b0b0b0;
            font-size: 0.9em;
        }}
        
        .revenue-card {{
            background: linear-gradient(135deg, #1e3a5f, #2d4f7a);
            border: 1px solid #4a9eff;
        }}
        
        .performance-card {{
            background: linear-gradient(135deg, #1b5e20, #2e7d32);
            border: 1px solid #4caf50;
        }}
        
        .agents-card {{
            background: linear-gradient(135deg, #332200, #4d3300);
            border: 1px solid #ff9800;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 10px;
            background: #404040;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4a9eff, #79c0ff);
            transition: width 0.3s ease;
        }}
        
        .agent-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .agent-card {{
            background: #1a1a1a;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 15px;
            position: relative;
        }}
        
        .agent-type {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        
        .yield-farmer {{
            background: #1e3a5f;
            color: #4a9eff;
        }}
        
        .trading-bot {{
            background: #1b5e20;
            color: #4caf50;
        }}
        
        .liquidation-protector {{
            background: #330000;
            color: #f44336;
        }}
        
        .portfolio-manager {{
            background: #332200;
            color: #ff9800;
        }}
        
        .arbitrage-agent {{
            background: #4a148c;
            color: #e1bee7;
        }}
        
        .agent-name {{
            color: #e0e0e0;
            font-weight: 600;
            margin: 0 0 5px 0;
        }}
        
        .agent-revenue {{
            color: #4caf50;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .chart-container {{
            background: #1a1a1a;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
        }}
        
        .chart-title {{
            color: #4a9eff;
            font-size: 1.3em;
            font-weight: 600;
            margin: 0 0 20px 0;
            text-align: center;
        }}
        
        .revenue-breakdown {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: space-between;
        }}
        
        .breakdown-item {{
            background: #2d2d2d;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #4a9eff;
            min-width: 150px;
            flex: 1;
        }}
        
        .breakdown-label {{
            color: #b0b0b0;
            font-size: 0.9em;
            margin: 0 0 5px 0;
        }}
        
        .breakdown-value {{
            color: #e0e0e0;
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-active {{ background: #4caf50; }}
        .status-warning {{ background: #ff9800; }}
        .status-error {{ background: #f44336; }}
        
        .system-health {{
            display: flex;
            align-items: center;
            margin-top: 15px;
        }}
        
        .health-status {{
            color: #4caf50;
            font-weight: 600;
        }}
        
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #404040;
            text-align: center;
            color: #808080;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
            
            .agent-grid {{
                grid-template-columns: 1fr;
            }}
            
            .revenue-breakdown {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <article>
        <header>
            <h1>🤖 Chainlink AI Agent dApps</h1>
            <p class="subtitle">Revenue-Generating AI Agents Dashboard</p>
            <div class="timestamp">Last Updated: {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
        </header>
        
        <main>
            <section class="dashboard-grid">
                <div class="metric-card revenue-card">
                    <div class="metric-title">💰 Total Revenue</div>
                    <div class="metric-value">${overview['system_info']['total_system_revenue']:,.2f}</div>
                    <div class="metric-label">Current Total</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(100, overview['system_info']['target_progress_percentage'])}%"></div>
                    </div>
                    <div class="metric-label">Target Progress: {overview['system_info']['target_progress_percentage']:.1f}%</div>
                </div>
                
                <div class="metric-card performance-card">
                    <div class="metric-title">📈 Monthly Projection</div>
                    <div class="metric-value">${revenue_report['revenue_summary']['monthly_projection']:,.0f}</div>
                    <div class="metric-label">Projected Monthly Revenue</div>
                </div>
                
                <div class="metric-card agents-card">
                    <div class="metric-title">🤖 Active Agents</div>
                    <div class="metric-value">{overview['system_info']['total_active_agents']}</div>
                    <div class="metric-label">AI Agents Deployed</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">⚡ Total Trades</div>
                    <div class="metric-value">{overview['system_info']['total_trades_executed']:,}</div>
                    <div class="metric-label">Executed Successfully</div>
                </div>
            </section>
            
            <section class="chart-container">
                <h2 class="chart-title">Revenue Breakdown by Model</h2>
                <div class="revenue-breakdown">
                    {''.join([f'''
                    <div class="breakdown-item">
                        <div class="breakdown-label">{model.replace('_', ' ').title()}</div>
                        <div class="breakdown-value">${amount:,.0f}</div>
                    </div>
                    ''' for model, amount in revenue_report['revenue_by_model'].items()])}
                </div>
            </section>
            
            <section class="chart-container">
                <h2 class="chart-title">Agent Type Distribution</h2>
                <div class="agent-grid">
                    {''.join([f'''
                    <div class="agent-card">
                        <span class="agent-type {agent_type.replace('_', '-')}">{agent_type.replace('_', ' ').title()}</span>
                        <div class="agent-name">{count} Active Agents</div>
                        <div class="agent-revenue">${revenue_report['revenue_by_agent_type'].get(agent_type, 0):,.0f} Revenue</div>
                    </div>
                    ''' for agent_type, count in overview['agent_distribution'].items()])}
                </div>
            </section>
            
            <section class="chart-container">
                <h2 class="chart-title">Top Performing Agents</h2>
                <div class="agent-grid">
                    {''.join([f'''
                    <div class="agent-card">
                        <span class="agent-type {agent['agent_type'].replace('_', '-')}">{agent['agent_type'].replace('_', ' ').title()}</span>
                        <div class="agent-name">{agent['agent_name']}</div>
                        <div class="agent-revenue">${agent['total_revenue']:,.2f}</div>
                        <div class="metric-label">Success Rate: {agent['success_rate']:.1f}%</div>
                    </div>
                    ''' for agent in revenue_report['top_performing_agents']])}
                </div>
            </section>
            
            <section class="chart-container">
                <h2 class="chart-title">System Health & Integration Status</h2>
                <div class="revenue-breakdown">
                    <div class="breakdown-item">
                        <div class="breakdown-label">Chainlink Price Feeds</div>
                        <div class="breakdown-value">{overview['chainlink_integrations']['price_feeds_active']}</div>
                        <div class="system-health">
                            <span class="status-indicator status-active"></span>
                            <span class="health-status">Operational</span>
                        </div>
                    </div>
                    
                    <div class="breakdown-item">
                        <div class="breakdown-label">CCIP Connections</div>
                        <div class="breakdown-value">{overview['chainlink_integrations']['ccip_connections']}</div>
                        <div class="system-health">
                            <span class="status-indicator status-active"></span>
                            <span class="health-status">Connected</span>
                        </div>
                    </div>
                    
                    <div class="breakdown-item">
                        <div class="breakdown-label">Yield Strategies</div>
                        <div class="breakdown-value">{overview['yield_strategies']}</div>
                        <div class="system-health">
                            <span class="status-indicator status-active"></span>
                            <span class="health-status">Active</span>
                        </div>
                    </div>
                    
                    <div class="breakdown-item">
                        <div class="breakdown-label">Supported Chains</div>
                        <div class="breakdown-value">{len(overview['supported_chains'])}</div>
                        <div class="system-health">
                            <span class="status-indicator status-active"></span>
                            <span class="health-status">Multi-Chain</span>
                        </div>
                    </div>
                </div>
            </section>
        </main>
        
        <footer>
            <p>🔗 Powered by Chainlink Runtime Environment (CRE)</p>
            <p>Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </article>
</body>
</html>"""
    
    async def _create_tinyurl_and_notify(self, filename: str, title: str, dashboard_data: dict):
        """Create TinyURL and send WhatsApp notification"""
        try:
            # Create the web URL
            external_url = f"http://34.58.114.120:8080/{filename}"
            
            # Generate TinyURL for easy mobile access
            try:
                tinyurl_response = requests.get(
                    "https://tinyurl.com/api-create.php",
                    params={"url": external_url},
                    timeout=5
                )
                if tinyurl_response.status_code == 200:
                    tiny_url = tinyurl_response.text.strip()
                    print(f"✅ TinyURL created: {tiny_url}")
                else:
                    tiny_url = external_url
                    print(f"⚠️ TinyURL creation failed, using full URL")
            except Exception as e:
                tiny_url = external_url
                print(f"⚠️ TinyURL error: {e}, using full URL")
            
            # Prepare WhatsApp message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            total_revenue = dashboard_data.get('total_revenue', 0)
            total_agents = dashboard_data.get('total_agents', 0)
            
            message = f"""🤖 AI Agent dApps Dashboard Ready!

📝 {title}
🕐 {timestamp}

🔗 Quick Access: {tiny_url}

💰 Total Revenue: ${total_revenue:,.2f}
🤖 Active Agents: {total_agents}
📁 {filename}
✨ Comprehensive AI agent performance and revenue analytics"""
            
            # Send WhatsApp notification
            try:
                import sys
                sys.path.append('/home/craigmbrown/Project/whatsapp-mcp')
                from whatsapp_mcp.server import send_message
                

# Claude Advanced Tools Integration
try:
    from claude_advanced_tools import ToolRegistry, ToolSearchEngine, PatternLearner
    ADVANCED_TOOLS_ENABLED = True
except ImportError:
    ADVANCED_TOOLS_ENABLED = False

                await send_message("15712781730", message)
                print(f"📱 WhatsApp notification sent with TinyURL: {tiny_url}")
                
            except ImportError:
                print("📱 WhatsApp integration not available - notification skipped")
            except Exception as e:
                print(f"📱 WhatsApp notification failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to create TinyURL and notify: {e}")
            print(f"Full error details: {e}")

async def main():
    """Generate AI agent dApps dashboard"""
    try:
        # Initialize AI dApps system
        ai_dapps = ChainlinkAIAgentDApps()
        
        # Create some sample data
        await ai_dapps.create_defi_yield_farmer({
            'name': 'DeFi Yield Maximizer Pro',
            'revenue_model': 'performance_based',
            'chains': ['ethereum', 'polygon']
        })
        
        await ai_dapps.create_trading_bot({
            'name': 'Chainlink Trading Bot Elite',
            'revenue_model': 'hybrid',
            'chains': ['ethereum', 'arbitrum']
        })
        
        # Generate dashboard
        dashboard = AIAgentDashboard(ai_dapps)
        dashboard_path = await dashboard.generate_comprehensive_dashboard()
        
        print(f"✅ AI Agent dApps Dashboard generated: {dashboard_path}")
        # print(f"🌐 Open command commented out - not available on server")
        
        return dashboard_path
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
        print(f"Full error details: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())