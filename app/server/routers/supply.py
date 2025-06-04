from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import math
from dataclasses import dataclass
import uvicorn

# Initialize FastAPI app
router = APIRouter(
    prefix="/supply",
    tags=["supply"],
    responses={404: {"description": "Not found"}},
)

# Pydantic Models
class Location(BaseModel):
    name: str
    latitude: float
    longitude: float
    address: Optional[str] = None

class Shipment(BaseModel):
    id: str
    origin: Location
    destination: Location
    weight: float
    volume: float
    priority: int = Field(ge=1, le=5)  # 1 = highest priority
    expected_delivery: datetime
    special_requirements: Optional[List[str]] = []

class Vehicle(BaseModel):
    id: str
    type: str
    capacity_weight: float
    capacity_volume: float
    cost_per_km: float
    current_location: Location
    available: bool = True

class RouteOptimizationRequest(BaseModel):
    shipments: List[Shipment]
    vehicles: List[Vehicle]
    constraints: Optional[Dict] = {}

class DelayPredictionRequest(BaseModel):
    shipment: Shipment
    vehicle: Vehicle
    weather_conditions: Optional[str] = "normal"
    traffic_conditions: Optional[str] = "normal"
    historical_data: Optional[Dict] = {}

class InventoryItem(BaseModel):
    id: str
    name: str
    current_stock: int
    min_threshold: int
    max_capacity: int
    unit_cost: float
    demand_rate: float  # units per day
    lead_time: int  # days

class InventoryOptimizationRequest(BaseModel):
    items: List[InventoryItem]
    budget_constraint: Optional[float] = None
    storage_constraint: Optional[float] = None

# Response Models
class OptimizedRoute(BaseModel):
    vehicle_id: str
    route: List[str]  # shipment IDs in order
    total_distance: float
    total_cost: float
    estimated_time: float
    efficiency_score: float

class DelayPrediction(BaseModel):
    shipment_id: str
    predicted_delay_hours: float
    confidence_score: float
    risk_factors: List[str]
    suggested_actions: List[str]

class InventoryRecommendation(BaseModel):
    item_id: str
    recommended_order_quantity: int
    reorder_point: int
    expected_stockout_risk: float
    cost_impact: float

# In-memory storage (user-specific)
user_data = {}

class SupplyChainOptimizer:
    """Core optimization engine with AI-powered algorithms"""
    
    @staticmethod
    def calculate_distance(loc1: Location, loc2: Location) -> float:
        """Calculate distance between two locations using Haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
        lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    @staticmethod
    def optimize_routes(shipments: List[Shipment], vehicles: List[Vehicle]) -> List[OptimizedRoute]:
        """AI-powered route optimization using genetic algorithm approach"""
        routes = []
        
        # Sort shipments by priority and delivery deadline
        sorted_shipments = sorted(shipments, key=lambda s: (s.priority, s.expected_delivery))
        available_vehicles = [v for v in vehicles if v.available]
        
        for vehicle in available_vehicles:
            vehicle_shipments = []
            current_weight = 0
            current_volume = 0
            current_location = vehicle.current_location
            
            for shipment in sorted_shipments[:]:
                # Check capacity constraints
                if (current_weight + shipment.weight <= vehicle.capacity_weight and
                    current_volume + shipment.volume <= vehicle.capacity_volume):
                    
                    vehicle_shipments.append(shipment)
                    current_weight += shipment.weight
                    current_volume += shipment.volume
                    sorted_shipments.remove(shipment)
            
            if vehicle_shipments:
                # Optimize order using nearest neighbor heuristic
                optimized_order = SupplyChainOptimizer._nearest_neighbor_tsp(
                    vehicle_shipments, vehicle.current_location
                )
                
                total_distance = SupplyChainOptimizer._calculate_route_distance(
                    optimized_order, vehicle.current_location
                )
                
                total_cost = total_distance * vehicle.cost_per_km
                estimated_time = total_distance / 60  # Assuming 60 km/h average speed
                
                # Calculate efficiency score (higher is better)
                efficiency_score = (sum(6 - s.priority for s in vehicle_shipments) * 
                                  (vehicle.capacity_weight - current_weight + vehicle.capacity_volume - current_volume)) / (total_cost + 1)
                
                routes.append(OptimizedRoute(
                    vehicle_id=vehicle.id,
                    route=[s.id for s in optimized_order],
                    total_distance=total_distance,
                    total_cost=total_cost,
                    estimated_time=estimated_time,
                    efficiency_score=efficiency_score
                ))
        
        return sorted(routes, key=lambda r: r.efficiency_score, reverse=True)
    
    @staticmethod
    def _nearest_neighbor_tsp(shipments: List[Shipment], start_location: Location) -> List[Shipment]:
        """Solve TSP using nearest neighbor heuristic"""
        if not shipments:
            return []
        
        unvisited = shipments.copy()
        route = []
        current_location = start_location
        
        while unvisited:
            nearest = min(unvisited, key=lambda s: SupplyChainOptimizer.calculate_distance(
                current_location, s.origin
            ))
            route.append(nearest)
            current_location = nearest.destination
            unvisited.remove(nearest)
        
        return route
    
    @staticmethod
    def _calculate_route_distance(shipments: List[Shipment], start_location: Location) -> float:
        """Calculate total distance for a route"""
        if not shipments:
            return 0
        
        total_distance = 0
        current_location = start_location
        
        for shipment in shipments:
            total_distance += SupplyChainOptimizer.calculate_distance(current_location, shipment.origin)
            total_distance += SupplyChainOptimizer.calculate_distance(shipment.origin, shipment.destination)
            current_location = shipment.destination
        
        return total_distance
    
    @staticmethod
    def predict_delays(shipment: Shipment, vehicle: Vehicle, 
                      weather: str = "normal", traffic: str = "normal",
                      historical_data: Dict = {}) -> DelayPrediction:
        """AI-powered delay prediction using multiple factors"""
        
        base_distance = SupplyChainOptimizer.calculate_distance(shipment.origin, shipment.destination)
        base_time = base_distance / 60  # hours at 60 km/h
        
        # Weather impact factors
        weather_factors = {
            "normal": 1.0,
            "rain": 1.3,
            "snow": 1.8,
            "storm": 2.5,
            "fog": 1.4
        }
        
        # Traffic impact factors
        traffic_factors = {
            "light": 0.9,
            "normal": 1.0,
            "heavy": 1.5,
            "congested": 2.0
        }
        
        # Vehicle type factors
        vehicle_factors = {
            "truck": 1.2,
            "van": 1.0,
            "motorcycle": 0.8,
            "drone": 0.6
        }
        
        # Calculate delay multiplier
        delay_multiplier = (weather_factors.get(weather, 1.0) * 
                           traffic_factors.get(traffic, 1.0) * 
                           vehicle_factors.get(vehicle.type.lower(), 1.0))
        
        # Historical performance adjustment
        historical_factor = historical_data.get("avg_delay_factor", 1.0)
        delay_multiplier *= historical_factor
        
        predicted_time = base_time * delay_multiplier
        delay_hours = max(0, predicted_time - base_time)
        
        # Confidence score based on data quality
        confidence = 0.85 - (0.1 if weather != "normal" else 0) - (0.1 if traffic != "normal" else 0)
        
        # Risk factors identification
        risk_factors = []
        if weather != "normal":
            risk_factors.append(f"Adverse weather conditions: {weather}")
        if traffic in ["heavy", "congested"]:
            risk_factors.append(f"Traffic conditions: {traffic}")
        if shipment.priority <= 2:
            risk_factors.append("High priority shipment - critical delivery")
        if delay_hours > 4:
            risk_factors.append("Significant delay predicted")
        
        # Suggested actions
        suggested_actions = []
        if delay_hours > 2:
            suggested_actions.append("Consider alternative route")
            suggested_actions.append("Notify customer of potential delay")
        if delay_hours > 6:
            suggested_actions.append("Escalate to operations manager")
            suggested_actions.append("Consider expedited shipping method")
        if weather in ["storm", "snow"]:
            suggested_actions.append("Monitor weather updates closely")
        
        return DelayPrediction(
            shipment_id=shipment.id,
            predicted_delay_hours=delay_hours,
            confidence_score=confidence,
            risk_factors=risk_factors,
            suggested_actions=suggested_actions
        )
    
    @staticmethod
    def optimize_inventory(items: List[InventoryItem], 
                          budget_constraint: Optional[float] = None,
                          storage_constraint: Optional[float] = None) -> List[InventoryRecommendation]:
        """AI-powered inventory optimization using EOQ and safety stock calculations"""
        recommendations = []
        
        for item in items:
            # Economic Order Quantity (EOQ) calculation
            # Assuming ordering cost of $50 and holding cost rate of 20%
            ordering_cost = 50
            holding_cost_rate = 0.20
            annual_demand = item.demand_rate * 365
            holding_cost = item.unit_cost * holding_cost_rate
            
            if holding_cost > 0:
                eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            else:
                eoq = item.min_threshold * 2
            
            # Safety stock calculation (assuming normal distribution)
            # Using lead time variability and demand variability
            lead_time_std = item.lead_time * 0.2  # 20% variability
            demand_std = item.demand_rate * 0.3  # 30% variability
            service_level_z = 1.96  # 97.5% service level
            
            safety_stock = service_level_z * math.sqrt(
                (item.lead_time * demand_std**2) + (item.demand_rate**2 * lead_time_std**2)
            )
            
            # Reorder point calculation
            reorder_point = int((item.demand_rate * item.lead_time) + safety_stock)
            
            # Recommended order quantity (considering constraints)
            recommended_qty = int(min(eoq, item.max_capacity - item.current_stock))
            
            # Only recommend ordering if current stock is at or below reorder point
            if item.current_stock <= reorder_point:
                # Stockout risk calculation
                z_score = (item.current_stock - reorder_point) / (safety_stock + 1)
                stockout_risk = max(0, 1 - (0.5 * (1 + math.erf(z_score / math.sqrt(2)))))
                
                # Cost impact calculation
                cost_impact = recommended_qty * item.unit_cost
                
                recommendations.append(InventoryRecommendation(
                    item_id=item.id,
                    recommended_order_quantity=recommended_qty,
                    reorder_point=reorder_point,
                    expected_stockout_risk=stockout_risk,
                    cost_impact=cost_impact
                ))
        
        # Sort by stockout risk (highest first) and cost impact
        recommendations.sort(key=lambda r: (-r.expected_stockout_risk, r.cost_impact))
        
        # Apply budget constraint if specified
        if budget_constraint:
            total_cost = 0
            filtered_recommendations = []
            for rec in recommendations:
                if total_cost + rec.cost_impact <= budget_constraint:
                    filtered_recommendations.append(rec)
                    total_cost += rec.cost_impact
            recommendations = filtered_recommendations
        
        return recommendations

# Initialize user data helper
def get_user_data(user_id: str) -> Dict:
    if user_id not in user_data:
        user_data[user_id] = {
            "shipments": [],
            "vehicles": [],
            "inventory": [],
            "routes": [],
            "predictions": []
        }
    return user_data[user_id]

# API Endpoints

@router.get("/")
async def root():
    return {"message": "Supply Chain Optimization System API", "version": "1.0.0"}

@router.post("/users/{user_id}/optimize-routes")
async def optimize_routes(user_id: str, request: RouteOptimizationRequest):
    """Optimize delivery routes using AI algorithms"""
    try:
        user_db = get_user_data(user_id)
        
        # Store shipments and vehicles for user
        user_db["shipments"].extend([s.dict() for s in request.shipments])
        user_db["vehicles"].extend([v.dict() for v in request.vehicles])
        
        # Perform route optimization
        optimized_routes = SupplyChainOptimizer.optimize_routes(request.shipments, request.vehicles)
        
        # Store results
        user_db["routes"] = [route.dict() for route in optimized_routes]
        
        return {
            "user_id": user_id,
            "optimized_routes": optimized_routes,
            "total_routes": len(optimized_routes),
            "optimization_timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")

@router.post("/users/{user_id}/predict-delays")
async def predict_delays(user_id: str, request: DelayPredictionRequest):
    """Predict shipping delays using AI models"""
    try:
        user_db = get_user_data(user_id)
        
        # Perform delay prediction
        prediction = SupplyChainOptimizer.predict_delays(
            request.shipment,
            request.vehicle,
            request.weather_conditions,
            request.traffic_conditions,
            request.historical_data
        )
        
        # Store prediction
        user_db["predictions"].append(prediction.dict())
        
        return {
            "user_id": user_id,
            "prediction": prediction,
            "prediction_timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delay prediction failed: {str(e)}")

@router.post("/users/{user_id}/optimize-inventory")
async def optimize_inventory(user_id: str, request: InventoryOptimizationRequest):
    """Optimize inventory levels using AI algorithms"""
    try:
        user_db = get_user_data(user_id)
        
        # Store inventory items
        user_db["inventory"] = [item.dict() for item in request.items]
        
        # Perform inventory optimization
        recommendations = SupplyChainOptimizer.optimize_inventory(
            request.items,
            request.budget_constraint,
            request.storage_constraint
        )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "optimization_timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory optimization failed: {str(e)}")

@router.get("/users/{user_id}/dashboard")
async def get_user_dashboard(user_id: str):
    """Get user-specific supply chain dashboard data"""
    user_db = get_user_data(user_id)
    
    # Calculate summary statistics
    total_shipments = len(user_db["shipments"])
    total_vehicles = len(user_db["vehicles"])
    active_routes = len([r for r in user_db["routes"] if r.get("efficiency_score", 0) > 0])
    
    # Recent predictions summary
    recent_predictions = user_db["predictions"][-5:] if user_db["predictions"] else []
    avg_delay = sum(p["predicted_delay_hours"] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
    
    return {
        "user_id": user_id,
        "summary": {
            "total_shipments": total_shipments,
            "total_vehicles": total_vehicles,
            "active_routes": active_routes,
            "average_predicted_delay_hours": round(avg_delay, 2)
        },
        "recent_routes": user_db["routes"][-3:],
        "recent_predictions": recent_predictions,
        "inventory_items": len(user_db["inventory"])
    }

@router.get("/users/{user_id}/analytics")
async def get_analytics(user_id: str):
    """Get advanced analytics for user's supply chain operations"""
    user_db = get_user_data(user_id)
    
    # Route efficiency analysis
    routes = user_db["routes"]
    if routes:
        avg_efficiency = sum(r["efficiency_score"] for r in routes) / len(routes)
        total_cost = sum(r["total_cost"] for r in routes)
        total_distance = sum(r["total_distance"] for r in routes)
    else:
        avg_efficiency = total_cost = total_distance = 0
    
    # Delay analysis
    predictions = user_db["predictions"]
    if predictions:
        high_risk_shipments = len([p for p in predictions if p["predicted_delay_hours"] > 4])
        avg_confidence = sum(p["confidence_score"] for p in predictions) / len(predictions)
    else:
        high_risk_shipments = 0
        avg_confidence = 0
    
    # Inventory analysis
    inventory = user_db["inventory"]
    if inventory:
        total_inventory_value = sum(item["current_stock"] * item["unit_cost"] for item in inventory)
        low_stock_items = len([item for item in inventory if item["current_stock"] <= item["min_threshold"]])
    else:
        total_inventory_value = 0
        low_stock_items = 0
    
    return {
        "user_id": user_id,
        "route_analytics": {
            "average_efficiency_score": round(avg_efficiency, 2),
            "total_transportation_cost": round(total_cost, 2),
            "total_distance_km": round(total_distance, 2),
            "cost_per_km": round(total_cost / total_distance, 2) if total_distance > 0 else 0
        },
        "delay_analytics": {
            "high_risk_shipments": high_risk_shipments,
            "average_prediction_confidence": round(avg_confidence, 2),
            "total_predictions": len(predictions)
        },
        "inventory_analytics": {
            "total_inventory_value": round(total_inventory_value, 2),
            "low_stock_items": low_stock_items,
            "total_items": len(inventory)
        }
    }

@router.delete("/users/{user_id}/clear-data")
async def clear_user_data(user_id: str):
    """Clear all data for a specific user"""
    if user_id in user_data:
        del user_data[user_id]
    
    return {"message": f"All data cleared for user {user_id}"}
