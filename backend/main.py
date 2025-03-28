from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import random
import io
# Optional ML imports (comment out if not using to avoid needing heavy deps when not necessary)
try:
    from tensorflow import keras
except ImportError:
    keras = None

app = FastAPI(title="AI Real Estate Platform", description="Backend API for AI-Powered Real Estate Platform")

# Enable CORS for all origins (so that the HTML/JS frontend can call the API from any host)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static frontend files (serving index.html at root)
app.mount("/", StaticFiles(directory="C:/Users/Dell/real-estate-platform/frontend", html=True), name="frontend")

# Define Pydantic models for request and response data where appropriate

class PropertyInput(BaseModel):
    address: str
    city: str
    state: str
    size_sqft: Optional[float] = None  # e.g. land or building size
    current_price: Optional[float] = None  # current property price if known

class EvaluationResult(BaseModel):
    median_price: float
    price_per_sqft: float
    trend_1yr: float
    trend_5yr: float
    demand_supply_ratio: Optional[float] = None
    conclusion: str

class ZoningInput(BaseModel):
    zone_code: str
    lot_size_sqft: float

class ZoningRecommendation(BaseModel):
    suggested_use: str
    max_height_ft: float
    floors: int
    est_units: int
    notes: str

class MarketingPlanInput(BaseModel):
    project_type: str  # e.g. "Luxury Condo", "Office", etc.
    budget: float
    target_audience: str

class MarketingPlan(BaseModel):
    channels: List[str]
    budget_allocation: dict
    kpi_targets: dict
    summary: str

class FinancialInput(BaseModel):
    project_cost: float
    land_cost: float
    construction_cost: float
    financing_rate: float  # e.g. annual interest rate in % for loans
    sell_price: Optional[float] = None  # if project is build-to-sell
    hold_rent: Optional[float] = None   # if project is build-to-rent (annual rent income)
    hold_years: Optional[int] = None

class FinancialOutput(BaseModel):
    total_cost: float
    total_revenue: float
    profit: float
    irr: Optional[float] = None
    roi: float
    cashflow: List[float]  # simplistic cashflow list for each period (e.g., year)
    recommendation: str

class ScenarioInput(BaseModel):
    base_case: FinancialInput
    scenarios: List[dict]  # each scenario can override some FinancialInput fields

class ScenarioOutcome(BaseModel):
    scenario_name: str
    irr: float
    profit: float

class MaterialsInput(BaseModel):
    building_size_sqft: float
    building_type: str  # e.g. "Residential", "Commercial"

class MaterialEstimate(BaseModel):
    material: str
    quantity: float
    cost: float

class ChecklistItem(BaseModel):
    task: str
    due_month: int  # e.g., month number in project timeline
    completed: bool = False

class AppealInput(BaseModel):
    # If using image file, we would handle via UploadFile in endpoint instead.
    style: Optional[str] = None  # e.g. "modern", "traditional"
    features: Optional[List[str]] = None  # e.g. ["Open Floor Plan", "Large Windows"]

class AppealResult(BaseModel):
    score: float
    interpretation: str

class DemandInput(BaseModel):
    area: str  # e.g. neighborhood or city
    property_type: str  # e.g. "Residential"
    years: int = 5  # forecast horizon

class DemandForecast(BaseModel):
    year: int
    projected_demand: float

class RiskInput(BaseModel):
    iterations: int = 1000
    # base assumptions:
    base_roi: float
    cost_variance_pct: float = 0.1  # e.g. 0.1 = 10% cost variance
    demand_variance_pct: float = 0.1
    price_variance_pct: float = 0.1

class RiskAnalysis(BaseModel):
    probable_roi: float
    worst_case_roi: float
    best_case_roi: float
    probability_roi_positive: float

# Sample data for demonstration (in real case, replace with actual data sources or API calls)
# e.g., a fake dataset of median prices for demonstration
sample_price_trends = {
    "city": {"median_price": 300000, "price_sqft": 200, "1yr_growth": 0.05, "5yr_growth": 0.20},
}

# ---------- Endpoint Implementations ----------

@app.post("/evaluate", response_model=EvaluationResult)
def evaluate_property(input: PropertyInput):
    """Evaluate property cost relative to market trends and supply/demand."""
    city_key = input.city.lower()
    if city_key in sample_price_trends:
        data = sample_price_trends[city_key]
    else:
        # In real implementation, fetch data from an API or dataset (Zillow, Redfin, etc.)
        data = {"median_price": 250000, "price_sqft": 180, "1yr_growth": 0.03, "5yr_growth": 0.15}
    # Calculate demand-supply ratio (dummy value for now)
    demand_supply = round(random.uniform(0.8, 1.2), 2)  # e.g., 1.1 means demand slightly exceeds supply
    # Determine conclusion
    conclusion = ""
    if input.current_price:
        if input.current_price > data["median_price"] * 1.1:
            conclusion = "The property is priced above the typical market range for the area."
        elif input.current_price < data["median_price"] * 0.9:
            conclusion = "The property is priced below the market median, indicating a potential bargain."
        else:
            conclusion = "The property is in line with the median market pricing."
    else:
        conclusion = "Market trends are stable. 5-year growth in the area was {}%, and 1-year growth {}%.".format(int(data["5yr_growth"]*100), int(data["1yr_growth"]*100))
    return EvaluationResult(
        median_price=data["median_price"],
        price_per_sqft=data["price_sqft"],
        trend_1yr=data["1yr_growth"],
        trend_5yr=data["5yr_growth"],
        demand_supply_ratio=demand_supply,
        conclusion=conclusion
    )

@app.post("/optimize-zoning", response_model=ZoningRecommendation)
def optimize_zoning(input: ZoningInput):
    """Suggest optimal building configuration for given zoning and lot size."""
    # Dummy logic: use zone_code to decide use, and lot_size for scale
    zone = input.zone_code.upper()
    if zone.startswith("R"):  # residential zone
        use = "Residential Multi-family"
        max_height = 40.0  # feet, e.g., 3-4 stories
    elif zone.startswith("C"):  # commercial zone
        use = "Commercial Mixed-Use"
        max_height = 60.0
    else:
        use = "Mixed-Use Development"
        max_height = 50.0
    # Estimate floors and units
    floors = int(max_height // 10)  # assume ~10 feet per floor
    est_units = int(input.lot_size_sqft // 800)  # assume each unit ~800 sqft of lot (just a rough guess)
    notes = "Zoning code {} allows {} use up to ~{} ft. Suggested {} floors, approx {} units.".format(
        input.zone_code, use, max_height, floors, est_units
    )
    return ZoningRecommendation(suggested_use=use, max_height_ft=max_height, floors=floors, est_units=est_units, notes=notes)

@app.post("/marketing-plan", response_model=MarketingPlan)
def generate_marketing_plan(input: MarketingPlanInput):
    """Generate a marketing plan based on project type, budget, and target audience."""
    # Simple rule-based plan: allocate budget across channels depending on project_type
    channels = []
    allocation = {}
    kpis = {}
    if "Luxury" in input.project_type or "Condo" in input.project_type:
        channels = ["Social Media", "Real Estate Portals", "VIP Events"]
        allocation = {"Social Media": input.budget * 0.4, "Real Estate Portals": input.budget * 0.3, "VIP Events": input.budget * 0.3}
        kpis = {"Leads": 100, "Site Visits": 50, "Conversions": 10}
        summary = (f"For a {input.project_type} targeting {input.target_audience}, focus on high-end channels. "
                   f"Allocate 40% of budget to social media campaigns (Instagram, Facebook ads), 30% to listings on premium real estate portals, "
                   f"and 30% to invite-only VIP open house events. Aim for {kpis['Leads']} leads, {kpis['Site Visits']} site visits, and {kpis['Conversions']} sales conversions.")
    else:
        # Default plan
        channels = ["Social Media", "Online Listings"]
        allocation = {"Social Media": input.budget * 0.5, "Online Listings": input.budget * 0.5}
        kpis = {"Leads": 50, "Site Visits": 20, "Conversions": 5}
        summary = (f"The marketing plan for {input.project_type} will target {input.target_audience} primarily through digital channels. "
                   f"Half the budget goes to social media advertising and content creation, and half to ensuring presence on online listing platforms. "
                   f"Expected outcomes are {kpis['Leads']} inquiries, leading to {kpis['Site Visits']} property visits and roughly {kpis['Conversions']} deals closed.")
    return MarketingPlan(channels=channels, budget_allocation=allocation, kpi_targets=kpis, summary=summary)

@app.post("/financial-model", response_model=FinancialOutput)
def financial_model(input: FinancialInput):
    """Calculate project financials including ROI and IRR based on costs and revenues."""
    # Summing costs
    total_cost = input.project_cost if input.project_cost else (input.land_cost + input.construction_cost)
    # Determine revenue
    if input.sell_price:
        total_revenue = input.sell_price
    elif input.hold_rent and input.hold_years:
        total_revenue = input.hold_rent * input.hold_years
    else:
        total_revenue = 0.0
    profit = total_revenue - total_cost
    roi = (profit / total_cost) * 100 if total_cost else 0.0
    # Simple IRR calculation: if sell_price given, treat it as cash inflow at end of project; if rental, treat annual rent as cashflow.
    cashflow = []
    irr = None
    try:
        if input.sell_price:
            # Assume 0 cashflow until sale
            cashflow = [ - total_cost, input.sell_price ]
        elif input.hold_rent and input.hold_years:
            annual_cost = total_cost / input.hold_years
            # Cashflow: negative costs for construction period (year 0 maybe) then rental income each year
            cashflow = [- total_cost] + [input.hold_rent] * input.hold_years
        # Use numpyfFinancial or manual IRR
        irr = np.irr(cashflow) * 100.0  # in percent
        irr = float(round(irr, 2))
    except Exception:
        irr = None
    recommendation = "Profitable" if profit > 0 else "Likely not profitable"
    return FinancialOutput(total_cost=round(total_cost,2),
                            total_revenue=round(total_revenue,2),
                            profit=round(profit,2),
                            irr=irr,
                            roi=round(roi,2),
                            cashflow=[round(x,2) for x in cashflow] if cashflow else [],
                            recommendation=recommendation)

@app.post("/scenario-planner", response_model=List[ScenarioOutcome])
def scenario_planner(input: ScenarioInput):
    """Compare multiple scenarios for value engineering (cost savings, etc.)."""
    outcomes = []
    # Base case outcome for reference
    base = input.base_case
    base_result = financial_model(base)  # reuse the financial_model logic
    outcomes.append({"scenario_name": "Base Case", "irr": base_result.irr if base_result.irr else 0.0, "profit": base_result.profit})
    # Iterate through scenarios
    for i, sc in enumerate(input.scenarios):
        # Override base case with scenario values
        sc_input_data = base.dict()
        sc_input_data.update(sc)  # replace with scenario overrides
        sc_input = FinancialInput(**sc_input_data)
        result = financial_model(sc_input)
        outcomes.append({"scenario_name": sc.get("name", f"Scenario {i+1}"), "irr": result.irr if result.irr else 0.0, "profit": result.profit})
    return outcomes

@app.post("/materials-estimate", response_model=List[MaterialEstimate])
def materials_estimate(input: MaterialsInput):
    """Provide a list of materials and cost estimates based on building size and type."""
    # Basic material requirements per 1000 sqft for demo
    ratios = {
        "concrete_cy": 5,   # cubic yards of concrete per 1000 sqft
        "steel_tons": 1,    # tons of steel per 1000 sqft
        "drywall_sf": 800,  # sqft drywall per 1000 sqft
    }
    factor = input.building_size_sqft / 1000.0
    # Assume unit costs (could be from data or API)
    unit_costs = {
        "concrete_cy": 100,   # $100 per cubic yard
        "steel_tons": 500,    # $500 per ton
        "drywall_sf": 2       # $2 per sqft
    }
    estimates = []
    for material, qty_per_unit in ratios.items():
        qty = qty_per_unit * factor
        cost = qty * unit_costs.get(material, 0)
        # Prettify material name
        mat_name = material.replace("_", " ")
        estimates.append({"material": mat_name, "quantity": round(qty, 2), "cost": round(cost, 2)})
    return estimates

@app.get("/project-checklist", response_model=List[ChecklistItem])
def project_checklist():
    """Get a generic project management checklist for a standard development timeline."""
    # Static checklist for demo. In real scenario, could adjust based on project specifics.
    checklist = [
        {"task": "Land Acquisition", "due_month": 1, "completed": False},
        {"task": "Design & Architecture Plans", "due_month": 3, "completed": False},
        {"task": "Permits Approval", "due_month": 6, "completed": False},
        {"task": "Groundbreaking Ceremony", "due_month": 7, "completed": False},
        {"task": "Foundation Completion", "due_month": 9, "completed": False},
        {"task": "Framing Completion", "due_month": 12, "completed": False},
        {"task": "Roof & Exterior Done", "due_month": 15, "completed": False},
        {"task": "Interior Finishes", "due_month": 18, "completed": False},
        {"task": "Final Inspections", "due_month": 20, "completed": False},
        {"task": "Marketing & Pre-Sales", "due_month": 5, "completed": False},
        {"task": "Project Handover/Sale", "due_month": 24, "completed": False}
    ]
    return checklist

@app.get("/3d-model")
def get_3d_model():
    """Endpoint to serve a 3D model file (e.g., glTF or similar)."""
    # In a real app, this would return a file (using FileResponse) for a 3D model.
    # Here, we'll just return a placeholder message or could serve a static file if available.
    return {"message": "3D model would be served here (e.g., a glTF file)."}

@app.get("/roi-heatmap")
def get_roi_heatmap():
    """Compute ROI data for property sections (for heatmap visualization)."""
    # Dummy implementation: return ROI values for a 5x5 grid (e.g., representing building sections)
    grid_size = 5
    roi_data = [[round(random.uniform(5, 15), 2) for _ in range(grid_size)] for _ in range(grid_size)]
    # In a real app, we'd base this on actual unit profits. We could also return an image.
    return {"roi_matrix": roi_data}

@app.post("/tax-optimization")
def tax_optimization():
    """Provide tax optimization suggestions (static content for demo)."""
    suggestions = [
        "Consider a 1031 Exchange to defer capital gains by reinvesting sale proceeds into a similar property.",
        "Utilize accelerated depreciation (cost segregation) to increase paper losses in early years and reduce taxable income.",
        "Check for local property tax abatements or incentives for new developments in the area.",
        "If holding the property, consider a REIT structure or LLC for pass-through benefits.",
        "Explore energy-efficient building credits or grants (solar installation credits, etc.)."
    ]
    return {"strategies": suggestions}

@app.post("/compliance-checker")
def compliance_checker(checklist: List[ChecklistItem]):
    """Check project plan for common regulatory compliance issues."""
    issues = []
    # Example checks:
    tasks = [item.task.lower() for item in checklist]
    # Ensure environmental review if project is large
    if "environmental review" not in tasks and "permits approval" in tasks:
        issues.append("Environmental review process might be required and is not listed.")
    # Ensure permit is accounted for
    if "permits approval" not in tasks:
        issues.append("Building permits approval is missing in the plan.")
    # Check timeline realism (if final handover is too early for listed tasks)
    latest_month = max(item.due_month for item in checklist) if checklist else 0
    if latest_month < 12:
        issues.append("Project timeline seems too short; it typically takes 18-24 months for completion.")
    return {"issues": issues if issues else ["All checks passed."]}

@app.post("/appeal-predict", response_model=AppealResult)
def predict_appeal(input: AppealInput):
    """Predict design appeal score based on style/features or image (if implemented)."""
    # If an image file was provided, we would load it and run through a CNN model.
    # Here we simulate using the described style/features.
    score = 50.0  # base
    interp = []
    if input.style:
        if input.style.lower() == "modern":
            score += 20
            interp.append("Modern style is generally popular, increasing appeal.")
        elif input.style.lower() == "traditional":
            interp.append("Traditional style has a steady appeal.")
        else:
            interp.append(f"Style {input.style} noted.")
    if input.features:
        for feat in input.features:
            if "Open Floor Plan".lower() in feat.lower():
                score += 10
                interp.append("Open floor plans tend to be attractive to buyers.")
            if "Natural Light".lower() in feat.lower():
                score += 5
                interp.append("Good natural light improves appeal.")
    # Clamp score between 0 and 100
    score = max(0, min(100, score))
    interpretation = " ".join(interp) if interp else "No specific high-impact design features noted."
    return AppealResult(score=round(score,2), interpretation=interpretation)

@app.post("/forecast-demand", response_model=List[DemandForecast])
def forecast_demand(input: DemandInput):
    """Forecast hyperlocal demand for given area and property type."""
    # Dummy implementation: simple linear growth projection
    base_demand = 100.0  # assume current demand index = 100
    growth_rate = 0.05  # 5% yearly growth as a placeholder
    forecasts = []
    for year in range(1, input.years+1):
        projected = base_demand * ((1 + growth_rate) ** year)
        forecasts.append({"year": year, "projected_demand": round(projected, 2)})
    return forecasts

@app.post("/risk-simulation", response_model=RiskAnalysis)
def risk_simulation(input: RiskInput):
    """Run Monte Carlo simulation for project ROI given uncertainties."""
    simulations = []
    base_roi = input.base_roi
    for _ in range(input.iterations):
        # Randomly vary costs and demand within given variance
        cost_factor = random.uniform(1 - input.cost_variance_pct, 1 + input.cost_variance_pct)
        demand_factor = random.uniform(1 - input.demand_variance_pct, 1 + input.demand_variance_pct)
        price_factor = random.uniform(1 - input.price_variance_pct, 1 + input.price_variance_pct)
        # Simplified: ROI changes inversely with cost and directly with demand/price
        roi_sim = base_roi * demand_factor * price_factor / cost_factor
        simulations.append(roi_sim)
    simulations.sort()
    worst = simulations[0]
    best = simulations[-1]
    # probability of positive ROI (>0)
    positive_count = sum(1 for r in simulations if r > 0)
    prob_positive = positive_count / len(simulations) if simulations else 0.0
    # Use median or mean as "most probable"
    probable = np.median(simulations) if simulations else 0.0
    return RiskAnalysis(
        probable_roi=round(float(probable), 2),
        worst_case_roi=round(worst, 2),
        best_case_roi=round(best, 2),
        probability_roi_positive=round(prob_positive * 100, 2)
    )
 
