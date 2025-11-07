from pydantic import BaseModel
from typing import List, Optional


# Mock Input Schema
class NodalAnalysisInput(BaseModel):
    well_id: str
    # Trajectory
    measured_depth: float  # MD (m or ft)
    true_vertical_depth: float  # TVD (m or ft)

    # Completion
    tubing_diameter: float  # inches
    casing_diameter: float  # inches

    # PVT Properties
    fluid_type: str  # "oil", "gas", "water"
    fluid_gravity: float  # API gravity or gas gravity
    viscosity: float  # cP

    # Reservoir
    reservoir_pressure: float  # psi
    reservoir_temperature: float  # °F or °C

    # Equipment
    pump_type: Optional[str]
    pump_specs: Optional[dict]


# Mock Output Schema
class NodalAnalysisOutput(BaseModel):
    production_rate: float  # bbl/day or Mscf/day
    wellhead_pressure: float  # psi
    bottomhole_pressure: float  # psi
    ipr_curve: List[dict]  # [{pressure: x, rate: y}, ...]
    vlp_curve: List[dict]
    operating_point: dict  # {pressure: x, rate: y}
    status: str  # "success" or "error"
    warnings: List[str]
