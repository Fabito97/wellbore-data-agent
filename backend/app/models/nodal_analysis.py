from typing import List, Optional, Dict

from pydantic import BaseModel, Field


# ==================== Input/Output Models ====================

class WellTrajectoryPoint(BaseModel):
    """Single point in well trajectory."""
    MD: float = Field(description="Measured depth in meters")
    TVD: float = Field(description="True vertical depth in meters")
    ID: float = Field(description="Inner diameter in meters")


class PumpCurve(BaseModel):
    """ESP pump curve data."""
    flow: List[float] = Field(description="Flow rates in m3/hr")
    head: List[float] = Field(description="Head values in meters")


class NodalAnalysisInput(BaseModel):
    """
    Input parameters for nodal analysis.
    Agent extracts these from completion reports.
    """
    # Fluid properties
    rho: float = Field(1000.0, description="Fluid density [kg/m3]")
    mu: float = Field(1e-3, description="Viscosity [Pa.s]")

    # Well parameters
    reservoir_pressure: float = Field(..., description="Reservoir pressure [bar]")
    wellhead_pressure: float = Field(10.0, description="Wellhead pressure [bar]")
    PI: float = Field(..., description="Productivity Index [m3/hr per bar]")

    # Equipment
    esp_depth: float = Field(500.0, description="ESP intake depth [m]")
    roughness: float = Field(1e-5, description="Pipe roughness [m]")

    # Well trajectory (simplified - agent extracts from completion data)
    well_trajectory: List[WellTrajectoryPoint] = Field(
        default_factory=lambda: [
            WellTrajectoryPoint(MD=0.0, TVD=0.0, ID=0.3397),
            WellTrajectoryPoint(MD=500.0, TVD=500.0, ID=0.2445),
            WellTrajectoryPoint(MD=1500.0, TVD=1500.0, ID=0.1778),
            WellTrajectoryPoint(MD=2500.0, TVD=2500.0, ID=0.1778),
        ]
    )

    # Pump curve (can be extracted or use default)
    pump_curve: Optional[PumpCurve] = Field(
        default=PumpCurve(
            flow=[0, 100, 200, 300, 400],
            head=[600, 550, 450, 300, 100]
        )
    )


class NodalAnalysisResult(BaseModel):
    """Results from nodal analysis calculation."""
    success: bool = Field(description="Whether solution was found")

    # Operating point
    flowrate: Optional[float] = Field(None, description="Optimal flowrate [m3/hr]")
    bottomhole_pressure: Optional[float] = Field(None, description="Bottomhole pressure [bar]")
    pump_head: Optional[float] = Field(None, description="Required pump head [m]")

    # Additional info
    convergence_tolerance: float = Field(description="Convergence tolerance [bar]")
    message: str = Field(description="Result message")

    # For plotting/analysis
    vlp_curve: Optional[Dict[str, List[float]]] = None
    ipr_curve: Optional[Dict[str, List[float]]] = None