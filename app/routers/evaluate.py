from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

router = APIRouter(
    prefix="/evaluate",
    tags=["evaluate"],
    responses={404: {"description": "Not found"}},
)

# Models for request and response
class FinancialData(BaseModel):
    revenue: float = Field(..., description="Annual revenue in USD")
    profit_margin: float = Field(..., description="Profit margin as percentage")
    debt_to_equity: float = Field(..., description="Debt to equity ratio")
    cash_flow: float = Field(..., description="Annual cash flow in USD")
    assets: float = Field(..., description="Total assets in USD")
    liabilities: float = Field(..., description="Total liabilities in USD")
    growth_rate: float = Field(..., description="Annual growth rate as percentage")
    
class MarketData(BaseModel):
    market_share: float = Field(..., description="Market share as percentage")
    industry_growth: float = Field(..., description="Industry growth rate as percentage")
    competitors: List[str] = Field(..., description="List of main competitors")
    market_size: float = Field(..., description="Total addressable market size in USD")
    market_trends: List[str] = Field(..., description="Market trends affecting the business")

class ManagementData(BaseModel):
    ceo_experience: int = Field(..., description="CEO experience in years")
    management_turnover: float = Field(..., description="Management turnover rate as percentage")
    employee_satisfaction: float = Field(..., description="Employee satisfaction score out of 10")
    leadership_quality: float = Field(..., description="Leadership quality score out of 10")

class ProductData(BaseModel):
    product_portfolio: List[str] = Field(..., description="List of main products or services")
    r_and_d_investment: float = Field(..., description="R&D investment as percentage of revenue")
    product_lifecycle: str = Field(..., description="Product lifecycle stage (early, growth, mature, decline)")
    innovation_score: float = Field(..., description="Innovation score out of 10")
    customer_satisfaction: float = Field(..., description="Customer satisfaction score out of 10")

class RiskData(BaseModel):
    regulatory_risks: List[str] = Field(..., description="Regulatory risks facing the business")
    market_risks: List[str] = Field(..., description="Market risks facing the business")
    operational_risks: List[str] = Field(..., description="Operational risks facing the business")
    financial_risks: List[str] = Field(..., description="Financial risks facing the business")
    risk_mitigation_strategies: List[str] = Field(..., description="Risk mitigation strategies")

class HistoricalData(BaseModel):
    financial_history: Dict[str, Dict[str, float]] = Field(..., description="Financial metrics by year")
    major_events: List[Dict[str, str]] = Field(..., description="Major events in company history")

class BusinessAnalysisRequest(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    industry: str = Field(..., description="Industry of the company")
    founded_year: int = Field(..., description="Year the company was founded")
    company_size: int = Field(..., description="Number of employees")
    financial_data: FinancialData = Field(..., description="Financial metrics")
    market_data: MarketData = Field(..., description="Market metrics")
    management_data: ManagementData = Field(..., description="Management metrics")
    product_data: ProductData = Field(..., description="Product metrics")
    risk_data: RiskData = Field(..., description="Risk assessment")
    historical_data: HistoricalData = Field(..., description="Historical company data")

class ScoreDetails(BaseModel):
    score: float = Field(..., description="Score out of 10")
    strengths: List[str] = Field(..., description="Identified strengths")
    weaknesses: List[str] = Field(..., description="Identified weaknesses")
    
class BusinessAnalysisResponse(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    industry: str = Field(..., description="Industry of the company")
    analysis_date: str = Field(..., description="Date of analysis")
    overall_score: float = Field(..., description="Overall investment score out of 10")
    financial_health: ScoreDetails = Field(..., description="Financial health assessment")
    market_position: ScoreDetails = Field(..., description="Market position assessment")
    management_quality: ScoreDetails = Field(..., description="Management quality assessment")
    product_strength: ScoreDetails = Field(..., description="Product strength assessment")
    risk_profile: ScoreDetails = Field(..., description="Risk profile assessment")
    investment_recommendation: str = Field(..., description="Investment recommendation")
    swot_analysis: Dict[str, List[str]] = Field(..., description="SWOT analysis")
    valuation_metrics: Dict[str, float] = Field(..., description="Key valuation metrics")
    growth_projections: Dict[str, float] = Field(..., description="Growth projections")
    detailed_analysis: str = Field(..., description="Detailed analysis")
    key_performance_indicators: Dict[str, Any] = Field(..., description="Key performance indicators")
    competitor_comparison: Dict[str, Any] = Field(..., description="Comparison with competitors")
    industry_benchmark: Dict[str, Any] = Field(..., description="Industry benchmarks")

# Analysis engine
class BusinessAnalysisEngine:
    def __init__(self):
        # Industry benchmarks (simplified for example)
        self.industry_benchmarks = {
            "Technology": {
                "avg_profit_margin": 15.0,
                "avg_growth_rate": 12.0,
                "avg_r_and_d": 15.0,
                "avg_debt_to_equity": 0.4
            },
            "Healthcare": {
                "avg_profit_margin": 12.0,
                "avg_growth_rate": 8.0,
                "avg_r_and_d": 18.0,
                "avg_debt_to_equity": 0.5
            },
            "Retail": {
                "avg_profit_margin": 5.0,
                "avg_growth_rate": 4.0,
                "avg_r_and_d": 3.0,
                "avg_debt_to_equity": 0.7
            },
            "Manufacturing": {
                "avg_profit_margin": 8.0,
                "avg_growth_rate": 5.0,
                "avg_r_and_d": 6.0,
                "avg_debt_to_equity": 0.6
            },
            "Financial Services": {
                "avg_profit_margin": 20.0,
                "avg_growth_rate": 7.0,
                "avg_r_and_d": 5.0,
                "avg_debt_to_equity": 2.5
            }
        }
        
        # Default industry benchmark for unknown industries
        self.default_benchmark = {
            "avg_profit_margin": 10.0,
            "avg_growth_rate": 7.0,
            "avg_r_and_d": 8.0,
            "avg_debt_to_equity": 0.6
        }
    
    def analyze_financial_health(self, financial_data, industry):
        benchmark = self.industry_benchmarks.get(industry, self.default_benchmark)
        
        # Calculate financial health score
        score_components = [
            self._compare_to_benchmark(financial_data.profit_margin, benchmark["avg_profit_margin"], higher_better=True),
            self._score_debt_to_equity(financial_data.debt_to_equity, benchmark["avg_debt_to_equity"]),
            self._score_cash_flow_to_debt(financial_data.cash_flow, financial_data.liabilities),
            self._score_asset_to_liability(financial_data.assets, financial_data.liabilities)
        ]
        
        score = np.mean(score_components) * 10  # Convert to scale out of 10
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if financial_data.profit_margin > benchmark["avg_profit_margin"]:
            strengths.append(f"Above-average profit margin ({financial_data.profit_margin}% vs industry average {benchmark['avg_profit_margin']}%)")
        else:
            weaknesses.append(f"Below-average profit margin ({financial_data.profit_margin}% vs industry average {benchmark['avg_profit_margin']}%)")
            
        if financial_data.debt_to_equity < benchmark["avg_debt_to_equity"]:
            strengths.append(f"Lower debt-to-equity ratio than industry average ({financial_data.debt_to_equity} vs {benchmark['avg_debt_to_equity']})")
        else:
            weaknesses.append(f"Higher debt-to-equity ratio than industry average ({financial_data.debt_to_equity} vs {benchmark['avg_debt_to_equity']})")
            
        if financial_data.assets > financial_data.liabilities * 2:
            strengths.append(f"Strong asset coverage with assets-to-liabilities ratio of {financial_data.assets/financial_data.liabilities:.2f}")
        elif financial_data.assets < financial_data.liabilities * 1.2:
            weaknesses.append(f"Low asset coverage with assets-to-liabilities ratio of {financial_data.assets/financial_data.liabilities:.2f}")
            
        if financial_data.growth_rate > benchmark["avg_growth_rate"]:
            strengths.append(f"Above-average growth rate ({financial_data.growth_rate}% vs industry average {benchmark['avg_growth_rate']}%)")
        else:
            weaknesses.append(f"Below-average growth rate ({financial_data.growth_rate}% vs industry average {benchmark['avg_growth_rate']}%)")
            
        return ScoreDetails(
            score=min(max(score, 0), 10),  # Ensure score is between 0 and 10
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def analyze_market_position(self, market_data, industry, company_size):
        # Calculate market position score
        market_score_components = [
            market_data.market_share / 100 * 10,  # Market share as percentage of 10
            min(market_data.industry_growth / 20 * 10, 10),  # Industry growth contribution to score
            min(market_data.market_size / 1e9, 10) / 10 * 5  # Market size contribution (capped at 10B for max score)
        ]
        
        score = np.mean(market_score_components)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if market_data.market_share > 15:
            strengths.append(f"Strong market share ({market_data.market_share}%)")
        elif market_data.market_share < 5:
            weaknesses.append(f"Limited market share ({market_data.market_share}%)")
            
        if market_data.industry_growth > 10:
            strengths.append(f"Operating in high-growth industry ({market_data.industry_growth}% growth)")
        elif market_data.industry_growth < 2:
            weaknesses.append(f"Operating in slow-growth industry ({market_data.industry_growth}% growth)")
            
        if len(market_data.competitors) < 5:
            strengths.append(f"Limited competition with only {len(market_data.competitors)} major competitors")
        elif len(market_data.competitors) > 15:
            weaknesses.append(f"Highly competitive market with {len(market_data.competitors)} major competitors")
            
        if "digital transformation" in [t.lower() for t in market_data.market_trends]:
            strengths.append("Well-positioned for digital transformation trend")
            
        if "regulatory pressure" in [t.lower() for t in market_data.market_trends]:
            weaknesses.append("Facing increased regulatory pressure")
            
        return ScoreDetails(
            score=min(max(score, 0), 10),
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def analyze_management_quality(self, management_data):
        # Calculate management quality score
        mgmt_score_components = [
            min(management_data.ceo_experience / 20, 1) * 10,  # CEO experience (capped at 20 years for max score)
            (10 - management_data.management_turnover / 5),  # Lower turnover is better
            management_data.employee_satisfaction,  # Already on scale of 0-10
            management_data.leadership_quality  # Already on scale of 0-10
        ]
        
        score = np.mean(mgmt_score_components)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if management_data.ceo_experience > 10:
            strengths.append(f"Experienced CEO with {management_data.ceo_experience} years of experience")
        elif management_data.ceo_experience < 5:
            weaknesses.append(f"Relatively inexperienced CEO with only {management_data.ceo_experience} years of experience")
            
        if management_data.management_turnover < 10:
            strengths.append(f"Stable management team with low turnover ({management_data.management_turnover}%)")
        elif management_data.management_turnover > 20:
            weaknesses.append(f"High management turnover ({management_data.management_turnover}%)")
            
        if management_data.employee_satisfaction > 7.5:
            strengths.append(f"High employee satisfaction ({management_data.employee_satisfaction}/10)")
        elif management_data.employee_satisfaction < 6:
            weaknesses.append(f"Low employee satisfaction ({management_data.employee_satisfaction}/10)")
            
        if management_data.leadership_quality > 8:
            strengths.append(f"Exceptional leadership quality ({management_data.leadership_quality}/10)")
        elif management_data.leadership_quality < 6:
            weaknesses.append(f"Leadership quality concerns ({management_data.leadership_quality}/10)")
            
        return ScoreDetails(
            score=min(max(score, 0), 10),
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def analyze_product_strength(self, product_data, industry):
        benchmark = self.industry_benchmarks.get(industry, self.default_benchmark)
        
        # Calculate product strength score
        product_score_components = [
            product_data.innovation_score,  # Already on scale of 0-10
            product_data.customer_satisfaction,  # Already on scale of 0-10
            min(product_data.r_and_d_investment / benchmark["avg_r_and_d"] * 10, 10),  # R&D compared to industry
            self._score_product_lifecycle(product_data.product_lifecycle),
            min(len(product_data.product_portfolio) / 3, 1) * 10  # Product diversity (capped at 3 for max score)
        ]
        
        score = np.mean(product_score_components)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if product_data.innovation_score > 8:
            strengths.append(f"Strong innovation capabilities ({product_data.innovation_score}/10)")
        elif product_data.innovation_score < 5:
            weaknesses.append(f"Weak innovation capabilities ({product_data.innovation_score}/10)")
            
        if product_data.customer_satisfaction > 8:
            strengths.append(f"High customer satisfaction ({product_data.customer_satisfaction}/10)")
        elif product_data.customer_satisfaction < 6:
            weaknesses.append(f"Low customer satisfaction ({product_data.customer_satisfaction}/10)")
            
        if product_data.r_and_d_investment > benchmark["avg_r_and_d"]:
            strengths.append(f"Above-average R&D investment ({product_data.r_and_d_investment}% vs industry average {benchmark['avg_r_and_d']}%)")
        else:
            weaknesses.append(f"Below-average R&D investment ({product_data.r_and_d_investment}% vs industry average {benchmark['avg_r_and_d']}%)")
            
        if product_data.product_lifecycle == "growth":
            strengths.append("Products in growth phase of lifecycle")
        elif product_data.product_lifecycle == "decline":
            weaknesses.append("Products in decline phase of lifecycle")
            
        if len(product_data.product_portfolio) > 5:
            strengths.append(f"Diverse product portfolio with {len(product_data.product_portfolio)} products")
        elif len(product_data.product_portfolio) < 2:
            weaknesses.append(f"Limited product diversity with only {len(product_data.product_portfolio)} products")
            
        return ScoreDetails(
            score=min(max(score, 0), 10),
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def analyze_risk_profile(self, risk_data, industry, financial_data):
        # Calculate risk profile score (lower risks = higher score)
        num_risks = (len(risk_data.regulatory_risks) + 
                    len(risk_data.market_risks) + 
                    len(risk_data.operational_risks) + 
                    len(risk_data.financial_risks))
        
        num_mitigations = len(risk_data.risk_mitigation_strategies)
        
        # Base score starts at 10 and is reduced by risks, but increased by mitigations
        base_risk_score = 10 - (num_risks * 0.5) + (num_mitigations * 0.3)
        
        # Additional risk factors from financial data
        risk_modifiers = []
        
        if financial_data.debt_to_equity > 1.5:
            risk_modifiers.append(-1.0)  # High debt increases risk
        
        if financial_data.cash_flow < 0:
            risk_modifiers.append(-2.0)  # Negative cash flow is a significant risk
            
        score = base_risk_score + sum(risk_modifiers)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if num_mitigations > num_risks:
            strengths.append(f"Strong risk mitigation with {num_mitigations} strategies for {num_risks} identified risks")
        elif num_mitigations < num_risks / 2:
            weaknesses.append(f"Insufficient risk mitigation with only {num_mitigations} strategies for {num_risks} identified risks")
            
        if len(risk_data.regulatory_risks) > 3:
            weaknesses.append(f"High regulatory risk exposure with {len(risk_data.regulatory_risks)} identified risks")
            
        if len(risk_data.financial_risks) > 3:
            weaknesses.append(f"High financial risk exposure with {len(risk_data.financial_risks)} identified risks")
            
        if "Diversification strategy" in risk_data.risk_mitigation_strategies:
            strengths.append("Effective diversification strategy for risk mitigation")
            
        if financial_data.debt_to_equity > 1.5:
            weaknesses.append(f"High debt-to-equity ratio ({financial_data.debt_to_equity}) increases financial risk")
            
        if financial_data.cash_flow < 0:
            weaknesses.append("Negative cash flow increases financial vulnerability")
            
        return ScoreDetails(
            score=min(max(score, 0), 10),
            strengths=strengths,
            weaknesses=weaknesses
        )
        
    def create_swot_analysis(self, financial, market, management, product, risk):
        # Combine all strengths and weaknesses
        strengths = financial.strengths + market.strengths + management.strengths + product.strengths + risk.strengths
        weaknesses = financial.weaknesses + market.weaknesses + management.weaknesses + product.weaknesses + risk.weaknesses
        
        # Generate opportunities
        opportunities = []
        
        # Market-based opportunities
        for trend in market.strengths:
            if "growth" in trend.lower():
                opportunities.append("Expand market share in growing industry")
                
        if product.score > 7:
            opportunities.append("Leverage strong product portfolio for international expansion")
            
        if financial.score > 7:
            opportunities.append("Use strong financial position for strategic acquisitions")
            
        if "digital" in " ".join(market.strengths).lower():
            opportunities.append("Capitalize on digital transformation trends")
            
        # Management-based opportunities
        if management.score > 7:
            opportunities.append("Leverage strong leadership for organizational transformation")
            
        # Generate threats
        threats = []
        
        # Risk-based threats
        for weakness in risk.weaknesses:
            if "regulatory" in weakness.lower():
                threats.append("Increasing regulatory pressure")
            if "financial" in weakness.lower():
                threats.append("Financial market volatility")
                
        # Market-based threats
        if "competitive" in " ".join(market.weaknesses).lower():
            threats.append("Intensifying competition")
            
        # Product-based threats
        if "innovation" in " ".join(product.weaknesses).lower():
            threats.append("Disruptive technologies in the industry")
            
        if "decline" in " ".join(product.weaknesses).lower():
            threats.append("Product obsolescence")
            
        # Ensure we have some basic items in each category
        if not opportunities:
            opportunities = [
                "Industry consolidation opportunities",
                "New market entry possibilities",
                "Product line expansion"
            ]
            
        if not threats:
            threats = [
                "Changing consumer preferences",
                "Economic downturn impact",
                "Supply chain disruptions"
            ]
        
        return {
            "Strengths": strengths[:5],  # Limit to top 5
            "Weaknesses": weaknesses[:5],  # Limit to top 5
            "Opportunities": opportunities[:5],  # Limit to top 5
            "Threats": threats[:5]  # Limit to top 5
        }
        
    def calculate_valuation_metrics(self, financial_data, company_age):
        # Calculate common valuation metrics
        revenue = financial_data.revenue
        profit = revenue * (financial_data.profit_margin / 100)
        
        # Price-to-Earnings (P/E) ratio (estimated)
        industry_pe_multiple = 15  # Default industry P/E multiple
        estimated_pe = industry_pe_multiple * (1 + (financial_data.growth_rate / 100))
        
        # Enterprise Value (EV) estimation
        estimated_market_cap = profit * estimated_pe
        enterprise_value = estimated_market_cap + financial_data.liabilities - financial_data.cash_flow
        
        # Return on Assets (ROA)
        roa = profit / financial_data.assets * 100
        
        # Return on Equity (ROE)
        equity = financial_data.assets - financial_data.liabilities
        roe = profit / equity * 100 if equity > 0 else 0
        
        # EV/Revenue multiple
        ev_revenue = enterprise_value / revenue if revenue > 0 else 0
        
        # EV/EBITDA multiple (estimating EBITDA as 1.5x profit)
        estimated_ebitda = profit * 1.5
        ev_ebitda = enterprise_value / estimated_ebitda if estimated_ebitda > 0 else 0
        
        return {
            "estimated_pe_ratio": round(estimated_pe, 2),
            "enterprise_value_usd": round(enterprise_value, 2),
            "return_on_assets_pct": round(roa, 2),
            "return_on_equity_pct": round(roe, 2),
            "ev_to_revenue": round(ev_revenue, 2),
            "ev_to_ebitda": round(ev_ebitda, 2)
        }
        
    def calculate_growth_projections(self, financial_data, market_data, product_data):
        # Calculate growth projections
        base_growth = financial_data.growth_rate
        
        # Adjust based on industry growth
        industry_adjustment = market_data.industry_growth - base_growth
        
        # Adjust based on product lifecycle
        lifecycle_adjustments = {
            "early": 2.0,
            "growth": 1.5,
            "mature": 0.0,
            "decline": -2.0
        }
        
        lifecycle_factor = lifecycle_adjustments.get(product_data.product_lifecycle.lower(), 0)
        
        # R&D impact on future growth
        rd_impact = (product_data.r_and_d_investment - 10) / 5  # Comparing to 10% R&D benchmark
        
        # Market share growth potential
        market_potential = (100 - market_data.market_share) / 50  # Room to grow in market
        
        # Calculate projected growth rates for next three years
        year1_growth = base_growth + (industry_adjustment * 0.3) + (lifecycle_factor * 0.5) + (rd_impact * 0.5)
        year2_growth = year1_growth + (industry_adjustment * 0.5) + (lifecycle_factor * 0.7) + (rd_impact * 0.7)
        year3_growth = year2_growth + (industry_adjustment * 0.7) + (lifecycle_factor * 0.9) + (rd_impact * 0.9)
        
        # Calculate compound annual growth rate (CAGR)
        cagr_3yr = ((1 + year1_growth/100) * (1 + year2_growth/100) * (1 + year3_growth/100)) ** (1/3) - 1
        cagr_3yr *= 100  # Convert to percentage
        
        # Calculate revenue projections
        revenue_year1 = financial_data.revenue * (1 + year1_growth/100)
        revenue_year2 = revenue_year1 * (1 + year2_growth/100)
        revenue_year3 = revenue_year2 * (1 + year3_growth/100)
        
        return {
            "projected_growth_year1_pct": round(year1_growth, 2),
            "projected_growth_year2_pct": round(year2_growth, 2),
            "projected_growth_year3_pct": round(year3_growth, 2),
            "cagr_3yr_pct": round(cagr_3yr, 2),
            "projected_revenue_year1": round(revenue_year1, 2),
            "projected_revenue_year2": round(revenue_year2, 2),
            "projected_revenue_year3": round(revenue_year3, 2)
        }
        
    def generate_detailed_analysis(self, company_data, scores):
        """Generate a detailed textual analysis based on all analysis components"""
        company_name = company_data.company_name
        industry = company_data.industry
        founding_year = company_data.founded_year
        company_age = datetime.now().year - founding_year
        
        # Introduction
        analysis = f"## Detailed Analysis: {company_name}\n\n"
        analysis += f"{company_name} is a {company_age}-year-old company operating in the {industry} industry. "
        
        # Overall assessment
        if scores.overall_score >= 8:
            investment_outlook = "highly attractive"
        elif scores.overall_score >= 6:
            investment_outlook = "moderately attractive"
        elif scores.overall_score >= 4:
            investment_outlook = "neutral"
        else:
            investment_outlook = "relatively unattractive"
            
        analysis += f"Based on our comprehensive analysis, {company_name} represents a {investment_outlook} investment opportunity with an overall score of {scores.overall_score:.1f}/10.\n\n"
        
        # Financial Health
        analysis += "### Financial Health\n"
        analysis += f"The company demonstrates {self._score_to_descriptor(scores.financial_health.score)} financial health (score: {scores.financial_health.score:.1f}/10). "
        
        if scores.financial_health.score >= 7:
            analysis += f"{company_name} exhibits strong financial fundamentals with {company_data.financial_data.profit_margin}% profit margins "
            analysis += f"and a debt-to-equity ratio of {company_data.financial_data.debt_to_equity}. "
        else:
            analysis += f"{company_name}'s financial position shows some weaknesses with {company_data.financial_data.profit_margin}% profit margins "
            analysis += f"and a debt-to-equity ratio of {company_data.financial_data.debt_to_equity}. "
            
        analysis += "Key strengths: " + "; ".join(scores.financial_health.strengths[:2]) + ". "
        analysis += "Notable weaknesses: " + "; ".join(scores.financial_health.weaknesses[:2]) + ".\n\n"
        
        # Market Position
        analysis += "### Market Position\n"
        analysis += f"The company holds a {self._score_to_descriptor(scores.market_position.score)} market position (score: {scores.market_position.score:.1f}/10). "
        
        analysis += f"With {company_data.market_data.market_share}% market share in an industry growing at {company_data.market_data.industry_growth}% annually, "
        
        if scores.market_position.score >= 7:
            analysis += f"{company_name} demonstrates competitive strength against its {len(company_data.market_data.competitors)} major competitors. "
        else:
            analysis += f"{company_name} faces challenges in positioning against its {len(company_data.market_data.competitors)} major competitors. "
            
        analysis += "Key market trends affecting the company include: " + ", ".join(company_data.market_data.market_trends[:3]) + ".\n\n"
        
        # Management Quality
        analysis += "### Management Quality\n"
        analysis += f"Leadership quality is {self._score_to_descriptor(scores.management_quality.score)} (score: {scores.management_quality.score:.1f}/10). "
        
        analysis += f"The CEO brings {company_data.management_data.ceo_experience} years of experience, with employee satisfaction rated at {company_data.management_data.employee_satisfaction}/10. "
        
        if scores.management_quality.score >= 7:
            analysis += "The management team demonstrates strong leadership capabilities and operational execution. "
        else:
            analysis += "The management team shows some areas for improvement in leadership and execution. "
            
        analysis += "Key strengths: " + "; ".join(scores.management_quality.strengths[:2]) + ".\n\n"
        
        # Product Strength
        analysis += "### Product & Innovation\n"
        analysis += f"The company's product portfolio demonstrates {self._score_to_descriptor(scores.product_strength.score)} strength (score: {scores.product_strength.score:.1f}/10). "
        
        analysis += f"With {len(company_data.product_data.product_portfolio)} major products in the {company_data.product_data.product_lifecycle} stage of their lifecycle, "
        analysis += f"the company invests {company_data.product_data.r_and_d_investment}% of revenue in R&D with an innovation score of {company_data.product_data.innovation_score}/10. "
        
        if scores.product_strength.score >= 7:
            analysis += f"Customer satisfaction is strong at {company_data.product_data.customer_satisfaction}/10, indicating product-market fit. "
        else:
            analysis += f"Customer satisfaction is at {company_data.product_data.customer_satisfaction}/10, suggesting room for product improvements. "
            
        analysis += "Key product strengths: " + "; ".join(scores.product_strength.strengths[:2]) + ".\n\n"
        
        # Risk Profile
        analysis += "### Risk Assessment\n"
        analysis += f"The risk profile is {self._score_to_descriptor(scores.risk_profile.score)} (score: {scores.risk_profile.score:.1f}/10). "
        
        total_risks = (len(company_data.risk_data.regulatory_risks) + 
                      len(company_data.risk_data.market_risks) + 
                      len(company_data.risk_data.operational_risks) + 
                      len(company_data.risk_data.financial_risks))
                      
        analysis += f"The company faces {total_risks} identified risks across regulatory, market, operational, and financial dimensions. "
        
        if len(company_data.risk_data.risk_mitigation_strategies) > total_risks / 2:
            analysis += f"With {len(company_data.risk_data.risk_mitigation_strategies)} mitigation strategies in place, the company demonstrates proactive risk management. "
        else:
            analysis += f"With only {len(company_data.risk_data.risk_mitigation_strategies)} mitigation strategies in place, the company's risk management could be enhanced. "
            
        analysis += "Key risk concerns: " + "; ".join(scores.risk_profile.weaknesses[:2]) + ".\n\n"
        
        # Future Outlook
        analysis += "### Future Outlook & Investment Potential\n"
        analysis += f"Based on current growth trajectory of {company_data.financial_data.growth_rate}% and industry growth of {company_data.market_data.industry_growth}%, "
        analysis += f"{company_name} is positioned for {'sustainable long-term growth' if scores.overall_score >= 7 else 'moderate growth with some challenges'}. "
        
        # Investment recommendation summary
        if scores.overall_score >= 8:
            analysis += f"We strongly recommend {company_name} as an investment opportunity, particularly for investors seeking exposure to the {industry} sector. "
        elif scores.overall_score >= 6:
            analysis += f"{company_name} represents a moderate investment opportunity with some promising attributes but also areas that require monitoring. "
        else:
            analysis += f"We suggest caution when considering {company_name} as an investment, as there are significant areas requiring improvement before it becomes an attractive opportunity. "
        
        return analysis
        
    def create_kpi_dashboard(self, company_data, scores):
        """Create a dashboard of key performance indicators"""
        financial = company_data.financial_data
        market = company_data.market_data
        
        return {
            "financial_kpis": {
                "revenue_usd": financial.revenue,
                "profit_margin_pct": financial.profit_margin,
                "debt_to_equity": financial.debt_to_equity,
                "cash_flow_usd": financial.cash_flow,
                "asset_to_liability_ratio": financial.assets / financial.liabilities if financial.liabilities > 0 else "N/A",
                "financial_health_score": scores.financial_health.score
            },
            "market_kpis": {
                "market_share_pct": market.market_share,
                "industry_growth_pct": market.industry_growth,
                "market_size_usd": market.market_size,
                "competitor_count": len(market.competitors),
                "market_position_score": scores.market_position.score
            },
            "product_kpis": {
                "product_count": len(company_data.product_data.product_portfolio),
                "r_and_d_investment_pct": company_data.product_data.r_and_d_investment,
                "customer_satisfaction": company_data.product_data.customer_satisfaction,
                "innovation_score": company_data.product_data.innovation_score,
                "product_lifecycle": company_data.product_data.product_lifecycle,
                "product_strength_score": scores.product_strength.score
            },
            "risk_kpis": {
                "total_identified_risks": (len(company_data.risk_data.regulatory_risks) + 
                                        len(company_data.risk_data.market_risks) + 
                                        len(company_data.risk_data.operational_risks) + 
                                        len(company_data.risk_data.financial_risks)),
                "risk_mitigation_strategies": len(company_data.risk_data.risk_mitigation_strategies),
                "risk_profile_score": scores.risk_profile.score
            },
            "overall_investment_score": scores.overall_score
        }
        
    def create_competitor_comparison(self, company_data):
        """Create a comparison with competitors based on market data"""
        # For this example, we'll create a simulated comparison
        competitors = company_data.market_data.competitors[:5]  # Take top 5 competitors
        
        # Create simulated data for comparison
        comparison_data = {
            "companies": [company_data.company_name] + competitors,
            "market_share": [
                company_data.market_data.market_share,
                *[round(np.random.uniform(1, 20), 1) for _ in range(len(competitors))]
            ],
            "growth_rate": [
                company_data.financial_data.growth_rate,
                *[round(np.random.uniform(0, 20), 1) for _ in range(len(competitors))]
            ],
            "profit_margin": [
                company_data.financial_data.profit_margin,
                *[round(np.random.uniform(1, 25), 1) for _ in range(len(competitors))]
            ],
            "r_and_d_investment": [
                company_data.product_data.r_and_d_investment,
                *[round(np.random.uniform(1, 20), 1) for _ in range(len(competitors))]
            ]
        }
        
        # Calculate relative positions
        market_share_rank = self._calculate_rank(comparison_data["market_share"])
        growth_rate_rank = self._calculate_rank(comparison_data["growth_rate"])
        profit_margin_rank = self._calculate_rank(comparison_data["profit_margin"])
        r_and_d_rank = self._calculate_rank(comparison_data["r_and_d_investment"])
        
        return {
            "raw_data": comparison_data,
            "ranks": {
                "market_share_rank": market_share_rank,
                "growth_rate_rank": growth_rate_rank,
                "profit_margin_rank": profit_margin_rank,
                "r_and_d_rank": r_and_d_rank,
                "overall_rank": self._calculate_overall_rank([market_share_rank, growth_rate_rank, profit_margin_rank, r_and_d_rank])
            }
        }
        
    def create_industry_benchmark(self, company_data):
        """Compare company metrics with industry benchmarks"""
        industry = company_data.industry
        benchmark = self.industry_benchmarks.get(industry, self.default_benchmark)
        
        # Create comparison to industry benchmarks
        return {
            "industry": industry,
            "metrics": {
                "profit_margin": {
                    "company": company_data.financial_data.profit_margin,
                    "industry_avg": benchmark["avg_profit_margin"],
                    "percentile": self._percentile_estimate(company_data.financial_data.profit_margin, benchmark["avg_profit_margin"], 5)
                },
                "growth_rate": {
                    "company": company_data.financial_data.growth_rate,
                    "industry_avg": benchmark["avg_growth_rate"],
                    "percentile": self._percentile_estimate(company_data.financial_data.growth_rate, benchmark["avg_growth_rate"], 4)
                },
                "r_and_d_investment": {
                    "company": company_data.product_data.r_and_d_investment,
                    "industry_avg": benchmark["avg_r_and_d"],
                    "percentile": self._percentile_estimate(company_data.product_data.r_and_d_investment, benchmark["avg_r_and_d"], 6)
                },
                "debt_to_equity": {
                    "company": company_data.financial_data.debt_to_equity,
                    "industry_avg": benchmark["avg_debt_to_equity"],
                    "percentile": self._percentile_estimate(company_data.financial_data.debt_to_equity, benchmark["avg_debt_to_equity"], 0.2, lower_better=True)
                }
            }
        }

    def _compare_to_benchmark(self, value, benchmark, higher_better=True):
        """Compare a value to a benchmark and return a normalized score (0-1)"""
        ratio = value / benchmark if benchmark > 0 else 1
        
        if higher_better:
            return min(ratio, 2) / 2  # Cap at 2x benchmark for max score of 1
        else:
            return min(benchmark / value if value > 0 else 2, 2) / 2  # Inverse ratio, capped at 2x
    
    def _score_debt_to_equity(self, debt_to_equity, benchmark):
        """Score debt to equity ratio (lower is better)"""
        if debt_to_equity <= 0:
            return 1.0  # No debt is optimal
        
        ratio = benchmark / debt_to_equity if debt_to_equity > 0 else 2
        return min(ratio, 2) / 2  # Cap at 2x benchmark for max score of 1
    
    def _score_cash_flow_to_debt(self, cash_flow, liabilities):
        """Score cash flow to debt ratio"""
        if liabilities <= 0:
            return 1.0  # No liabilities is optimal
            
        if cash_flow <= 0:
            return 0.0  # Negative cash flow is problematic
            
        ratio = cash_flow / liabilities
        return min(ratio * 5, 1)  # Normalize to 0-1 range
    
    def _score_asset_to_liability(self, assets, liabilities):
        """Score asset to liability ratio"""
        if liabilities <= 0:
            return 1.0  # No liabilities is optimal
            
        ratio = assets / liabilities
        return min((ratio - 1) / 2, 1)  # Normalize to 0-1 range, with 1 being the minimum acceptable ratio
    
    def _score_product_lifecycle(self, lifecycle):
        """Score based on product lifecycle stage"""
        lifecycle_scores = {
            "early": 0.7,  # High potential but not proven
            "growth": 1.0,  # Optimal stage
            "mature": 0.5,  # Stable but limited growth
            "decline": 0.2   # Problematic stage
        }
        
        return lifecycle_scores.get(lifecycle.lower(), 0.5)
        
    def _score_to_descriptor(self, score):
        """Convert a numerical score to a descriptive term"""
        if score >= 8.5:
            return "exceptional"
        elif score >= 7.5:
            return "excellent"
        elif score >= 6.5:
            return "strong"
        elif score >= 5.5:
            return "good"
        elif score >= 4.5:
            return "moderate"
        elif score >= 3.5:
            return "fair"
        elif score >= 2.5:
            return "weak"
        else:
            return "poor"
            
    def _calculate_rank(self, values):
        """Calculate ranking (1 = highest) for a list of values"""
        # Create a sorted list of indices (descending order)
        sorted_indices = np.argsort(values)[::-1]
        
        # Create ranks (1-based)
        ranks = np.zeros_like(sorted_indices)
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
            
        return ranks.tolist()
        
    def _calculate_overall_rank(self, rank_lists):
        """Calculate overall rank based on multiple rank lists"""
        # Sum all ranks for each position
        total_ranks = [sum(ranks[i] for ranks in rank_lists) for i in range(len(rank_lists[0]))]
        
        # Calculate ranking based on total ranks (lower is better)
        sorted_indices = np.argsort(total_ranks)
        
        # Create ranks (1-based)
        overall_ranks = np.zeros_like(sorted_indices)
        for i, idx in enumerate(sorted_indices):
            overall_ranks[idx] = i + 1
            
        return overall_ranks.tolist()
        
    def _percentile_estimate(self, value, mean, std_dev, lower_better=False):
        """Estimate percentile based on assumed normal distribution"""
        z_score = (value - mean) / std_dev
        
        if lower_better:
            z_score = -z_score
            
        # Convert z-score to percentile
        percentile = (1 + np.tanh(z_score * 0.7)) / 2 * 100
        return round(percentile, 1)
        
    def analyze_business(self, business_data: BusinessAnalysisRequest) -> BusinessAnalysisResponse:
        """Main method to analyze a business and generate comprehensive report"""
        # Calculate company age
        company_age = datetime.now().year - business_data.founded_year
        
        # Perform individual analyses
        financial_health = self.analyze_financial_health(business_data.financial_data, business_data.industry)
        market_position = self.analyze_market_position(business_data.market_data, business_data.industry, business_data.company_size)
        management_quality = self.analyze_management_quality(business_data.management_data)
        product_strength = self.analyze_product_strength(business_data.product_data, business_data.industry)
        risk_profile = self.analyze_risk_profile(business_data.risk_data, business_data.industry, business_data.financial_data)
        
        # Calculate overall score
        overall_score = np.mean([
            financial_health.score * 0.25,  # Financial health weighted higher
            market_position.score * 0.2,
            management_quality.score * 0.2,
            product_strength.score * 0.2,
            risk_profile.score * 0.15
        ])
        
        # Create SWOT analysis
        swot_analysis = self.create_swot_analysis(financial_health, market_position, management_quality, product_strength, risk_profile)
        
        # Calculate valuation metrics
        valuation_metrics = self.calculate_valuation_metrics(business_data.financial_data, company_age)
        
        # Calculate growth projections
        growth_projections = self.calculate_growth_projections(business_data.financial_data, business_data.market_data, business_data.product_data)
        
        # Determine investment recommendation
        if overall_score >= 8:
            recommendation = "Strong Buy - The company demonstrates exceptional fundamentals and growth potential"
        elif overall_score >= 7:
            recommendation = "Buy - The company shows strong performance with minor concerns"
        elif overall_score >= 6:
            recommendation = "Moderate Buy - The company has positive attributes but some areas need improvement"
        elif overall_score >= 5:
            recommendation = "Hold - The company has balanced strengths and weaknesses"
        elif overall_score >= 4:
            recommendation = "Moderate Sell - The company shows concerning weaknesses that outweigh strengths"
        else:
            recommendation = "Sell - The company demonstrates significant weaknesses across multiple dimensions"
            
        # Create detailed analysis
        scores = type('Scores', (), {
            'overall_score': overall_score,
            'financial_health': financial_health,
            'market_position': market_position,
            'management_quality': management_quality,
            'product_strength': product_strength,
            'risk_profile': risk_profile
        })
        
        detailed_analysis = self.generate_detailed_analysis(business_data, scores)
        
        # Create KPI dashboard
        kpi_dashboard = self.create_kpi_dashboard(business_data, scores)
        
        # Create competitor comparison
        competitor_comparison = self.create_competitor_comparison(business_data)
        
        # Create industry benchmark comparison
        industry_benchmark = self.create_industry_benchmark(business_data)
        
        # Create response
        return BusinessAnalysisResponse(
            company_name=business_data.company_name,
            industry=business_data.industry,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            overall_score=round(overall_score, 1),
            financial_health=financial_health,
            market_position=market_position,
            management_quality=management_quality,
            product_strength=product_strength,
            risk_profile=risk_profile,
            investment_recommendation=recommendation,
            swot_analysis=swot_analysis,
            valuation_metrics=valuation_metrics,
            growth_projections=growth_projections,
            detailed_analysis=detailed_analysis,
            key_performance_indicators=kpi_dashboard,
            competitor_comparison=competitor_comparison,
            industry_benchmark=industry_benchmark
        )

# Initialize the business analysis engine
analysis_engine = BusinessAnalysisEngine()

@router.post("/analyze", response_model=BusinessAnalysisResponse)
async def analyze_business(business_data: BusinessAnalysisRequest):
    """Analyze a business and provide comprehensive investment analysis"""
    try:
        result = analysis_engine.analyze_business(business_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/industries")
async def get_industries():
    """Get list of supported industries with their benchmarks"""
    return {
        "industries": list(analysis_engine.industry_benchmarks.keys()),
        "benchmarks": analysis_engine.industry_benchmarks
    }
