#!/usr/bin/env python3
"""
REQ-CRE-MAPPER-001: Comprehensive Requirement-to-Code Mapping System
REQ-CRE-MAPPER-002: Automated Code Coverage Analysis
REQ-CRE-MAPPER-003: Requirement Traceability Matrix
REQ-CRE-MAPPER-004: Gap Analysis and Missing Implementation Detection
REQ-CRE-MAPPER-005: Code Quality and Compliance Validation
REQ-CRE-MAPPER-006: Automated Documentation Generation
REQ-CRE-MAPPER-007: Integration Testing Verification
REQ-CRE-MAPPER-008: Performance Metrics Correlation
REQ-CRE-MAPPER-009: Deployment Readiness Assessment

Requirement-to-Code Mapping System for Chainlink CRE Implementation
Ensures complete traceability from requirements to implemented code.
"""

import asyncio
import json
import logging
import re
import ast
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from decimal import Decimal
import hashlib


# Claude Advanced Tools Integration
try:
    from claude_advanced_tools import ToolRegistry, ToolSearchEngine, PatternLearner
    ADVANCED_TOOLS_ENABLED = True
except ImportError:
    ADVANCED_TOOLS_ENABLED = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/craigmbrown/Project/chainlink-prediction-markets-mcp/logs/requirement_mapper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Requirement:
    """REQ-CRE-MAPPER-001: Requirement definition"""
    req_id: str
    category: str
    title: str
    description: str
    priority: str
    status: str
    acceptance_criteria: List[str]
    dependencies: List[str]
    created_at: datetime

@dataclass
class CodeImplementation:
    """REQ-CRE-MAPPER-002: Code implementation tracking"""
    file_path: str
    function_name: str
    class_name: Optional[str]
    line_start: int
    line_end: int
    requirement_tags: List[str]
    complexity_score: int
    test_coverage: float
    documentation_quality: int

@dataclass
class RequirementMapping:
    """REQ-CRE-MAPPER-003: Requirement to code mapping"""
    req_id: str
    implementations: List[CodeImplementation]
    test_files: List[str]
    documentation_files: List[str]
    coverage_percentage: float
    quality_score: int
    compliance_status: str
    gaps: List[str]

@dataclass
class GapAnalysis:
    """REQ-CRE-MAPPER-004: Gap analysis result"""
    gap_id: str
    gap_type: str
    requirement_id: str
    description: str
    severity: str
    recommendation: str
    estimated_effort: str

@dataclass
class QualityMetric:
    """REQ-CRE-MAPPER-005: Code quality metric"""
    metric_name: str
    file_path: str
    score: float
    threshold: float
    passed: bool
    details: Dict[str, Any]

class RequirementCodeMapper:
    """REQ-CRE-MAPPER-001: Main requirement-to-code mapping system"""
    
    def __init__(self, project_root: str = "/home/craigmbrown/Project/chainlink-prediction-markets-mcp"):
        """Initialize requirement mapping system"""
        try:
            self.project_root = Path(project_root)
            self.chainlink_cre_path = self.project_root / "chainlink_cre"
            
            # Storage
            self.requirements: Dict[str, Requirement] = {}
            self.code_implementations: List[CodeImplementation] = []
            self.requirement_mappings: Dict[str, RequirementMapping] = {}
            self.gap_analyses: List[GapAnalysis] = []
            self.quality_metrics: List[QualityMetric] = []
            
            # Analysis results
            self.coverage_report = {}
            self.traceability_matrix = {}
            
            logger.info("RequirementCodeMapper initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RequirementCodeMapper: {e}")
            print(f"Full error details: {e}")
            raise
    
    async def load_requirements(self) -> Dict[str, Requirement]:
        """REQ-CRE-MAPPER-001: Load requirements from documentation"""
        try:
            requirements_file = self.project_root / "CHAINLINK_CRE_REQUIREMENTS.md"
            
            if not requirements_file.exists():
                logger.warning("Requirements file not found, creating sample requirements")
                return await self._create_sample_requirements()
            
            with open(requirements_file, 'r') as f:
                content = f.read()
            
            # Parse requirements from markdown
            requirements = self._parse_requirements_markdown(content)
            self.requirements = requirements
            
            logger.info(f"Loaded {len(requirements)} requirements")
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to load requirements: {e}")
            print(f"Full error details: {e}")
            raise
    
    def _parse_requirements_markdown(self, content: str) -> Dict[str, Requirement]:
        """Parse requirements from markdown content"""
        try:
            requirements = {}
            
            # Extract requirement blocks using regex
            req_pattern = r'### (REQ-CRE-\w+-\d+): (.+?)\n(.+?)(?=###|$)'
            matches = re.findall(req_pattern, content, re.DOTALL)
            
            for req_id, title, description in matches:
                # Extract acceptance criteria if present
                criteria_match = re.search(r'Acceptance Criteria:(.*?)(?=\n\n|\n###|$)', description, re.DOTALL)
                acceptance_criteria = []
                if criteria_match:
                    criteria_text = criteria_match.group(1).strip()
                    acceptance_criteria = [c.strip('- ').strip() for c in criteria_text.split('\n') if c.strip().startswith('-')]
                
                requirement = Requirement(
                    req_id=req_id,
                    category=req_id.split('-')[2],  # Extract category from req_id
                    title=title.strip(),
                    description=description.strip(),
                    priority="High",  # Default priority
                    status="In Progress",
                    acceptance_criteria=acceptance_criteria,
                    dependencies=[],
                    created_at=datetime.now()
                )
                
                requirements[req_id] = requirement
            
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to parse requirements markdown: {e}")
            return {}
    
    async def _create_sample_requirements(self) -> Dict[str, Requirement]:
        """Create sample requirements for testing"""
        try:
            sample_requirements = {
                "REQ-CRE-ORCH-001": Requirement(
                    req_id="REQ-CRE-ORCH-001",
                    category="ORCHESTRATION",
                    title="Agent Orchestration System",
                    description="Implement master orchestrator for specialized AI agents",
                    priority="High",
                    status="Completed",
                    acceptance_criteria=[
                        "System coordinates multiple specialized agents",
                        "Task queue management implemented",
                        "Dependency tracking functional"
                    ],
                    dependencies=[],
                    created_at=datetime.now()
                ),
                "REQ-CRE-DAPPS-001": Requirement(
                    req_id="REQ-CRE-DAPPS-001",
                    category="DAPPS",
                    title="AI Agent dApps Implementation",
                    description="Develop revenue-generating AI agent dApps",
                    priority="High",
                    status="Completed",
                    acceptance_criteria=[
                        "Multiple agent types supported",
                        "Revenue models implemented",
                        "Multi-chain support enabled"
                    ],
                    dependencies=["REQ-CRE-ORCH-001"],
                    created_at=datetime.now()
                ),
                "REQ-CRE-CCIP-001": Requirement(
                    req_id="REQ-CRE-CCIP-001",
                    category="CCIP",
                    title="Cross-Chain Integration",
                    description="Implement Chainlink CCIP for cross-chain operations",
                    priority="Medium",
                    status="Completed",
                    acceptance_criteria=[
                        "Cross-chain message passing",
                        "Revenue bridging implemented",
                        "Multi-chain liquidity management"
                    ],
                    dependencies=["REQ-CRE-DAPPS-001"],
                    created_at=datetime.now()
                )
            }
            
            self.requirements = sample_requirements
            return sample_requirements
            
        except Exception as e:
            logger.error(f"Failed to create sample requirements: {e}")
            return {}
    
    async def scan_code_implementations(self) -> List[CodeImplementation]:
        """REQ-CRE-MAPPER-002: Scan codebase for requirement implementations"""
        try:
            implementations = []
            
            # Scan all Python files in chainlink_cre directory
            python_files = list(self.chainlink_cre_path.glob("*.py"))
            
            for file_path in python_files:
                file_implementations = await self._analyze_python_file(file_path)
                implementations.extend(file_implementations)
            
            self.code_implementations = implementations
            logger.info(f"Found {len(implementations)} code implementations")
            return implementations
            
        except Exception as e:
            logger.error(f"Failed to scan code implementations: {e}")
            print(f"Full error details: {e}")
            raise
    
    async def _analyze_python_file(self, file_path: Path) -> List[CodeImplementation]:
        """Analyze a single Python file for requirement implementations"""
        try:
            implementations = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find requirement tags in comments
            req_pattern = r'REQ-CRE-[\w]+-\d+'
            req_tags = re.findall(req_pattern, content)
            
            if not req_tags:
                return implementations
            
            # Parse AST to find functions and classes
            try:
                tree = ast.parse(content)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping AST analysis")
                return implementations
            
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Find requirement tags in docstring or nearby comments
                    node_req_tags = self._extract_requirement_tags_for_node(node, lines)
                    
                    if node_req_tags:
                        implementation = CodeImplementation(
                            file_path=str(file_path),
                            function_name=node.name if isinstance(node, ast.FunctionDef) else None,
                            class_name=node.name if isinstance(node, ast.ClassDef) else None,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            requirement_tags=node_req_tags,
                            complexity_score=self._calculate_complexity(node),
                            test_coverage=0.0,  # Will be calculated separately
                            documentation_quality=self._assess_documentation_quality(node, lines)
                        )
                        implementations.append(implementation)
            
            return implementations
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return []
    
    def _extract_requirement_tags_for_node(self, node, lines: List[str]) -> List[str]:
        """Extract requirement tags associated with a specific AST node"""
        try:
            req_tags = []
            
            # Check docstring
            if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and 
                node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)):
                docstring = node.body[0].value.value
                if isinstance(docstring, str):
                    req_pattern = r'REQ-CRE-[\w]+-\d+'
                    req_tags.extend(re.findall(req_pattern, docstring))
            
            # Check comments around the node
            start_line = max(0, node.lineno - 10)
            end_line = min(len(lines), node.lineno + 5)
            
            for line_num in range(start_line, end_line):
                line = lines[line_num]
                if 'REQ-CRE-' in line:
                    req_pattern = r'REQ-CRE-[\w]+-\d+'
                    req_tags.extend(re.findall(req_pattern, line))
            
            return list(set(req_tags))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to extract requirement tags: {e}")
            return []
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a code node"""
        try:
            complexity = 1  # Base complexity
            
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                    ast.With, ast.ExceptHandler, ast.Assert)):
                    complexity += 1
                elif isinstance(child, (ast.BoolOp, ast.Compare)):
                    complexity += 1
            
            return complexity
            
        except Exception as e:
            logger.error(f"Failed to calculate complexity: {e}")
            return 1
    
    def _assess_documentation_quality(self, node, lines: List[str]) -> int:
        """Assess documentation quality (0-100)"""
        try:
            score = 0
            
            # Check for docstring
            if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and 
                node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)):
                docstring = node.body[0].value.value
                if isinstance(docstring, str):
                    score += 30
                    if len(docstring) > 50:
                        score += 20
                    if 'Args:' in docstring or 'Parameters:' in docstring:
                        score += 15
                    if 'Returns:' in docstring:
                        score += 15
                    if 'REQ-CRE-' in docstring:
                        score += 20
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Failed to assess documentation quality: {e}")
            return 0
    
    async def create_traceability_matrix(self) -> Dict[str, RequirementMapping]:
        """REQ-CRE-MAPPER-003: Create requirement traceability matrix"""
        try:
            if not self.requirements:
                await self.load_requirements()
            
            if not self.code_implementations:
                await self.scan_code_implementations()
            
            mappings = {}
            
            for req_id, requirement in self.requirements.items():
                # Find implementations for this requirement
                implementations = [
                    impl for impl in self.code_implementations 
                    if req_id in impl.requirement_tags
                ]
                
                # Calculate coverage
                coverage = 100.0 if implementations else 0.0
                
                # Assess quality
                quality_score = self._calculate_requirement_quality(implementations)
                
                # Identify gaps
                gaps = self._identify_requirement_gaps(requirement, implementations)
                
                mapping = RequirementMapping(
                    req_id=req_id,
                    implementations=implementations,
                    test_files=[],  # Will be populated by test analysis
                    documentation_files=[],
                    coverage_percentage=coverage,
                    quality_score=quality_score,
                    compliance_status="Compliant" if coverage > 80 and quality_score > 70 else "Non-Compliant",
                    gaps=gaps
                )
                
                mappings[req_id] = mapping
            
            self.requirement_mappings = mappings
            self.traceability_matrix = mappings
            
            logger.info(f"Created traceability matrix for {len(mappings)} requirements")
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to create traceability matrix: {e}")
            print(f"Full error details: {e}")
            raise
    
    def _calculate_requirement_quality(self, implementations: List[CodeImplementation]) -> int:
        """Calculate overall quality score for a requirement"""
        try:
            if not implementations:
                return 0
            
            # Average documentation quality
            doc_score = sum(impl.documentation_quality for impl in implementations) / len(implementations)
            
            # Complexity penalty (lower is better)
            avg_complexity = sum(impl.complexity_score for impl in implementations) / len(implementations)
            complexity_penalty = max(0, avg_complexity - 10) * 5  # Penalty for complexity > 10
            
            # Implementation completeness
            completeness = len(implementations) * 20  # 20 points per implementation
            
            quality_score = int(doc_score + completeness - complexity_penalty)
            return max(0, min(100, quality_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate requirement quality: {e}")
            return 0
    
    def _identify_requirement_gaps(self, requirement: Requirement, 
                                 implementations: List[CodeImplementation]) -> List[str]:
        """Identify gaps in requirement implementation"""
        try:
            gaps = []
            
            # Check if requirement has any implementation
            if not implementations:
                gaps.append("No code implementation found")
                return gaps
            
            # Check acceptance criteria coverage
            if requirement.acceptance_criteria:
                for criteria in requirement.acceptance_criteria:
                    # Simple heuristic: check if criteria keywords appear in implementations
                    criteria_keywords = criteria.lower().split()
                    found_coverage = False
                    
                    for impl in implementations:
                        impl_text = f"{impl.function_name or ''} {impl.class_name or ''}".lower()
                        if any(keyword in impl_text for keyword in criteria_keywords):
                            found_coverage = True
                            break
                    
                    if not found_coverage:
                        gaps.append(f"Acceptance criteria not covered: {criteria}")
            
            # Check for test coverage
            has_tests = any("test" in impl.file_path.lower() for impl in implementations)
            if not has_tests:
                gaps.append("No test coverage found")
            
            # Check documentation quality
            avg_doc_quality = sum(impl.documentation_quality for impl in implementations) / len(implementations)
            if avg_doc_quality < 50:
                gaps.append("Insufficient documentation quality")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to identify requirement gaps: {e}")
            return ["Error analyzing gaps"]
    
    async def perform_gap_analysis(self) -> List[GapAnalysis]:
        """REQ-CRE-MAPPER-004: Perform comprehensive gap analysis"""
        try:
            if not self.requirement_mappings:
                await self.create_traceability_matrix()
            
            gaps = []
            gap_counter = 1
            
            for req_id, mapping in self.requirement_mappings.items():
                for gap_description in mapping.gaps:
                    gap = GapAnalysis(
                        gap_id=f"GAP-{gap_counter:03d}",
                        gap_type=self._classify_gap_type(gap_description),
                        requirement_id=req_id,
                        description=gap_description,
                        severity=self._assess_gap_severity(gap_description, mapping),
                        recommendation=self._generate_gap_recommendation(gap_description),
                        estimated_effort=self._estimate_gap_effort(gap_description)
                    )
                    gaps.append(gap)
                    gap_counter += 1
            
            self.gap_analyses = gaps
            logger.info(f"Identified {len(gaps)} gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to perform gap analysis: {e}")
            print(f"Full error details: {e}")
            raise
    
    def _classify_gap_type(self, gap_description: str) -> str:
        """Classify the type of gap"""
        gap_lower = gap_description.lower()
        
        if "implementation" in gap_lower:
            return "IMPLEMENTATION"
        elif "test" in gap_lower:
            return "TESTING"
        elif "documentation" in gap_lower:
            return "DOCUMENTATION"
        elif "acceptance" in gap_lower:
            return "REQUIREMENTS"
        else:
            return "OTHER"
    
    def _assess_gap_severity(self, gap_description: str, mapping: RequirementMapping) -> str:
        """Assess gap severity"""
        gap_lower = gap_description.lower()
        
        if "no code implementation" in gap_lower:
            return "CRITICAL"
        elif "test coverage" in gap_lower:
            return "HIGH"
        elif "documentation" in gap_lower and mapping.quality_score < 30:
            return "HIGH"
        elif "acceptance criteria" in gap_lower:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_gap_recommendation(self, gap_description: str) -> str:
        """Generate recommendation for addressing the gap"""
        gap_lower = gap_description.lower()
        
        if "no code implementation" in gap_lower:
            return "Implement the required functionality according to the requirement specification"
        elif "test coverage" in gap_lower:
            return "Create comprehensive unit and integration tests"
        elif "documentation" in gap_lower:
            return "Add detailed docstrings and inline documentation"
        elif "acceptance criteria" in gap_lower:
            return "Review and implement missing acceptance criteria"
        else:
            return "Review and address the identified gap"
    
    def _estimate_gap_effort(self, gap_description: str) -> str:
        """Estimate effort required to address the gap"""
        gap_lower = gap_description.lower()
        
        if "no code implementation" in gap_lower:
            return "HIGH (3-5 days)"
        elif "test coverage" in gap_lower:
            return "MEDIUM (1-2 days)"
        elif "documentation" in gap_lower:
            return "LOW (0.5-1 day)"
        elif "acceptance criteria" in gap_lower:
            return "MEDIUM (1-2 days)"
        else:
            return "LOW (0.5-1 day)"
    
    async def generate_coverage_report(self) -> Dict[str, Any]:
        """REQ-CRE-MAPPER-002: Generate comprehensive coverage report"""
        try:
            if not self.requirement_mappings:
                await self.create_traceability_matrix()
            
            total_requirements = len(self.requirements)
            implemented_requirements = len([m for m in self.requirement_mappings.values() if m.implementations])
            tested_requirements = len([m for m in self.requirement_mappings.values() if m.test_files])
            compliant_requirements = len([m for m in self.requirement_mappings.values() if m.compliance_status == "Compliant"])
            
            coverage_report = {
                'summary': {
                    'total_requirements': total_requirements,
                    'implemented_requirements': implemented_requirements,
                    'tested_requirements': tested_requirements,
                    'compliant_requirements': compliant_requirements,
                    'implementation_coverage': (implemented_requirements / total_requirements * 100) if total_requirements > 0 else 0,
                    'test_coverage': (tested_requirements / total_requirements * 100) if total_requirements > 0 else 0,
                    'compliance_rate': (compliant_requirements / total_requirements * 100) if total_requirements > 0 else 0
                },
                'by_category': self._generate_category_coverage(),
                'quality_metrics': self._generate_quality_summary(),
                'gap_summary': self._generate_gap_summary(),
                'recommendations': self._generate_recommendations()
            }
            
            self.coverage_report = coverage_report
            logger.info("Coverage report generated successfully")
            return coverage_report
            
        except Exception as e:
            logger.error(f"Failed to generate coverage report: {e}")
            print(f"Full error details: {e}")
            raise
    
    def _generate_category_coverage(self) -> Dict[str, Any]:
        """Generate coverage breakdown by category"""
        try:
            categories = {}
            
            for req_id, requirement in self.requirements.items():
                category = requirement.category
                if category not in categories:
                    categories[category] = {
                        'total': 0,
                        'implemented': 0,
                        'compliant': 0,
                        'average_quality': 0
                    }
                
                categories[category]['total'] += 1
                
                if req_id in self.requirement_mappings:
                    mapping = self.requirement_mappings[req_id]
                    if mapping.implementations:
                        categories[category]['implemented'] += 1
                    if mapping.compliance_status == "Compliant":
                        categories[category]['compliant'] += 1
                    categories[category]['average_quality'] += mapping.quality_score
            
            # Calculate averages
            for category_data in categories.values():
                if category_data['total'] > 0:
                    category_data['average_quality'] /= category_data['total']
                    category_data['implementation_rate'] = (category_data['implemented'] / category_data['total'] * 100)
                    category_data['compliance_rate'] = (category_data['compliant'] / category_data['total'] * 100)
            
            return categories
            
        except Exception as e:
            logger.error(f"Failed to generate category coverage: {e}")
            return {}
    
    def _generate_quality_summary(self) -> Dict[str, Any]:
        """Generate quality metrics summary"""
        try:
            if not self.requirement_mappings:
                return {
                    'average_quality_score': 0,
                    'highest_quality_score': 0,
                    'lowest_quality_score': 0,
                    'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
                }
            
            quality_scores = [mapping.quality_score for mapping in self.requirement_mappings.values()]
            
            return {
                'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'highest_quality_score': max(quality_scores) if quality_scores else 0,
                'lowest_quality_score': min(quality_scores) if quality_scores else 0,
                'quality_distribution': {
                    'excellent': len([q for q in quality_scores if q >= 90]),
                    'good': len([q for q in quality_scores if 70 <= q < 90]),
                    'fair': len([q for q in quality_scores if 50 <= q < 70]),
                    'poor': len([q for q in quality_scores if q < 50])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate quality summary: {e}")
            return {}
    
    def _generate_gap_summary(self) -> Dict[str, Any]:
        """Generate gap analysis summary"""
        try:
            if not self.gap_analyses:
                return {}
            
            return {
                'total_gaps': len(self.gap_analyses),
                'by_severity': {
                    'critical': len([g for g in self.gap_analyses if g.severity == 'CRITICAL']),
                    'high': len([g for g in self.gap_analyses if g.severity == 'HIGH']),
                    'medium': len([g for g in self.gap_analyses if g.severity == 'MEDIUM']),
                    'low': len([g for g in self.gap_analyses if g.severity == 'LOW'])
                },
                'by_type': {
                    gap_type: len([g for g in self.gap_analyses if g.gap_type == gap_type])
                    for gap_type in set(g.gap_type for g in self.gap_analyses)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate gap summary: {e}")
            return {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            if not self.coverage_report:
                return recommendations
            
            summary = self.coverage_report.get('summary', {})
            
            # Implementation coverage recommendations
            impl_coverage = summary.get('implementation_coverage', 0)
            if impl_coverage < 80:
                recommendations.append(f"Increase implementation coverage from {impl_coverage:.1f}% to at least 80%")
            
            # Test coverage recommendations
            test_coverage = summary.get('test_coverage', 0)
            if test_coverage < 70:
                recommendations.append(f"Improve test coverage from {test_coverage:.1f}% to at least 70%")
            
            # Quality recommendations
            quality = self.coverage_report.get('quality_metrics', {})
            avg_quality = quality.get('average_quality_score', 0)
            if avg_quality < 70:
                recommendations.append(f"Improve code quality score from {avg_quality:.1f} to at least 70")
            
            # Gap-specific recommendations
            gaps = self.coverage_report.get('gap_summary', {})
            critical_gaps = gaps.get('by_severity', {}).get('critical', 0)
            if critical_gaps > 0:
                recommendations.append(f"Address {critical_gaps} critical implementation gaps immediately")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def generate_deployment_readiness_assessment(self) -> Dict[str, Any]:
        """REQ-CRE-MAPPER-009: Assess deployment readiness"""
        try:
            if not self.coverage_report:
                await self.generate_coverage_report()
            
            # Readiness criteria
            criteria = {
                'implementation_coverage': {
                    'current': self.coverage_report['summary']['implementation_coverage'],
                    'threshold': 95.0,
                    'weight': 0.3
                },
                'test_coverage': {
                    'current': self.coverage_report['summary']['test_coverage'],
                    'threshold': 80.0,
                    'weight': 0.25
                },
                'compliance_rate': {
                    'current': self.coverage_report['summary']['compliance_rate'],
                    'threshold': 90.0,
                    'weight': 0.25
                },
                'quality_score': {
                    'current': self.coverage_report['quality_metrics']['average_quality_score'],
                    'threshold': 75.0,
                    'weight': 0.2
                }
            }
            
            # Calculate readiness score
            readiness_score = 0
            criteria_results = {}
            
            for criterion, config in criteria.items():
                current = config['current']
                threshold = config['threshold']
                weight = config['weight']
                
                criterion_score = min(100, (current / threshold) * 100) if threshold > 0 else 0
                readiness_score += criterion_score * weight
                
                criteria_results[criterion] = {
                    'current': current,
                    'threshold': threshold,
                    'score': criterion_score,
                    'passed': current >= threshold
                }
            
            # Determine readiness status
            if readiness_score >= 95:
                status = "READY"
            elif readiness_score >= 85:
                status = "NEARLY_READY"
            elif readiness_score >= 70:
                status = "NEEDS_IMPROVEMENT"
            else:
                status = "NOT_READY"
            
            assessment = {
                'readiness_score': readiness_score,
                'status': status,
                'criteria_results': criteria_results,
                'critical_blockers': self._identify_critical_blockers(),
                'deployment_recommendations': self._generate_deployment_recommendations(readiness_score, criteria_results)
            }
            
            logger.info(f"Deployment readiness assessment: {status} ({readiness_score:.1f}%)")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to generate deployment readiness assessment: {e}")
            print(f"Full error details: {e}")
            raise
    
    def _identify_critical_blockers(self) -> List[str]:
        """Identify critical blockers for deployment"""
        try:
            blockers = []
            
            if self.gap_analyses:
                critical_gaps = [g for g in self.gap_analyses if g.severity == 'CRITICAL']
                for gap in critical_gaps:
                    blockers.append(f"Critical gap: {gap.description} in {gap.requirement_id}")
            
            # Check for unimplemented high-priority requirements
            for req_id, mapping in self.requirement_mappings.items():
                requirement = self.requirements.get(req_id)
                if requirement and requirement.priority == "High" and not mapping.implementations:
                    blockers.append(f"High-priority requirement not implemented: {req_id}")
            
            return blockers
            
        except Exception as e:
            logger.error(f"Failed to identify critical blockers: {e}")
            return []
    
    def _generate_deployment_recommendations(self, readiness_score: float, criteria_results: Dict[str, Any]) -> List[str]:
        """Generate deployment-specific recommendations"""
        try:
            recommendations = []
            
            if readiness_score < 95:
                recommendations.append("Complete remaining implementation gaps before deployment")
            
            for criterion, result in criteria_results.items():
                if not result['passed']:
                    if criterion == 'implementation_coverage':
                        recommendations.append(f"Implement remaining requirements to reach {result['threshold']:.1f}% coverage")
                    elif criterion == 'test_coverage':
                        recommendations.append(f"Add tests to reach {result['threshold']:.1f}% test coverage")
                    elif criterion == 'compliance_rate':
                        recommendations.append(f"Fix compliance issues to reach {result['threshold']:.1f}% compliance rate")
                    elif criterion == 'quality_score':
                        recommendations.append(f"Improve code quality to reach {result['threshold']:.1f} quality score")
            
            # Add specific actionable recommendations
            recommendations.extend([
                "Perform security audit before mainnet deployment",
                "Set up monitoring and alerting systems",
                "Prepare rollback procedures",
                "Conduct load testing with production-like data"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate deployment recommendations: {e}")
            return []

async def main():
    """Main function for testing requirement mapping"""
    try:
        # Initialize the system
        mapper = RequirementCodeMapper()
        
        print("📋 Requirement-to-Code Mapping System")
        print("=" * 50)
        
        # Load requirements (use sample data for testing)
        requirements = await mapper._create_sample_requirements()
        print(f"✅ Loaded {len(requirements)} requirements")
        
        # Scan code implementations
        implementations = await mapper.scan_code_implementations()
        print(f"✅ Found {len(implementations)} code implementations")
        
        # Create traceability matrix
        mappings = await mapper.create_traceability_matrix()
        print(f"✅ Created traceability matrix for {len(mappings)} requirements")
        
        # Perform gap analysis
        gaps = await mapper.perform_gap_analysis()
        print(f"✅ Identified {len(gaps)} gaps")
        
        # Generate coverage report
        coverage_report = await mapper.generate_coverage_report()
        print(f"✅ Generated coverage report")
        print(f"   Implementation Coverage: {coverage_report['summary']['implementation_coverage']:.1f}%")
        print(f"   Compliance Rate: {coverage_report['summary']['compliance_rate']:.1f}%")
        
        # Assess deployment readiness
        readiness = await mapper.generate_deployment_readiness_assessment()
        print(f"✅ Deployment readiness: {readiness['status']} ({readiness['readiness_score']:.1f}%)")
        
        print("\n🚀 Requirement mapping system analysis complete!")
        
    except Exception as e:
        logger.error(f"Main function failed: {e}")
        print(f"Full error details: {e}")

if __name__ == "__main__":
    asyncio.run(main())