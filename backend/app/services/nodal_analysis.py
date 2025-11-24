"""
Nodal Analysis Tool for AI Agent.

The agent will:
1. Extract parameters from documents via RAG
2. Call this tool with extracted parameters
3. Get back nodal analysis results
"""
# from app.models.nodal_analysis import NodalAnalysisInput, NodalAnalysisResult, WellTrajectoryPoint, PumpCurve
import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Nodal Analysis Calculator ====================

class OrganizerNodalAnalysis:
    """
       The organizer's nodal analysis model.
       DO NOT MODIFY - treat as black box.
       """

    def __init__(
            self,
            rho: float = 1000.0,
            mu: float = 1e-3,
            g: float = 9.81,
            roughness: float = 1e-5,
            reservoir_pressure: float = 230.0,
            wellhead_pressure: float = 10.0,
            PI: float = 5.0,
            esp_depth: float = 500.0,
            pump_curve: Dict[str, List[float]] = None,
            well_trajectory: List[Dict[str, float]] = None
    ):
        """Initialize with their parameters."""
        self.rho = rho
        self.mu = mu
        self.g = g
        self.roughness = roughness
        self.reservoir_pressure = reservoir_pressure
        self.wellhead_pressure = wellhead_pressure
        self.PI = PI
        self.esp_depth = esp_depth

        # Default pump curve
        if pump_curve is None:
            self.pump_curve = {
                "flow": [0, 100, 200, 300, 400],
                "head": [600, 550, 450, 300, 100],
            }
        else:
            self.pump_curve = pump_curve

        # Default well trajectory
        if well_trajectory is None:
            self.well_trajectory = [
                {"MD": 0.0, "TVD": 0.0, "ID": 0.3397},
                {"MD": 500.0, "TVD": 500.0, "ID": 0.2445},
                {"MD": 1500.0, "TVD": 1500.0, "ID": 0.1778},
                {"MD": 2500.0, "TVD": 2500.0, "ID": 0.1778},
            ]
        else:
            self.well_trajectory = well_trajectory

        # Build segments
        self.segments = self._build_segments()

    def _build_segments(self) -> List[Tuple[float, float, float]]:
        """Build well segments from trajectory."""
        segments = []
        for i in range(1, len(self.well_trajectory)):
            MD = self.well_trajectory[i]["MD"] - self.well_trajectory[i - 1]["MD"]
            TVD = self.well_trajectory[i]["TVD"] - self.well_trajectory[i - 1]["TVD"]
            D = self.well_trajectory[i]["ID"]
            L = MD
            theta = math.atan2(TVD, MD)
            segments.append((L, D, theta))
        return segments

    def swamee_jain(self, Re: float, D: float) -> float:
        """Friction factor calculation."""
        if Re <= 0:
            return 0.0
        return 0.25 / (math.log10((self.roughness / (3.7 * D)) + (5.74 / (Re ** 0.9)))) ** 2

    def pump_interp(self, flow: float, key: str) -> float:
        """Interpolate pump curve."""
        return np.interp(flow, self.pump_curve["flow"], self.pump_curve[key])

    def vlp(self, flow_m3hr: float) -> float:
        """Vertical Lift Performance."""
        q = flow_m3hr / 3600.0  # m3/hr to m3/s
        dp_total = 0.0
        depth_accum = 0.0

        for (L, D, theta) in self.segments:
            A = math.pi * D ** 2 / 4.0
            u = q / A
            Re = self.rho * abs(u) * D / self.mu
            f = self.swamee_jain(Re, D)

            dp_fric = f * (L / D) * (self.rho * u ** 2 / 2.0)
            dp_grav = self.rho * self.g * L * math.sin(theta)
            dp_total += dp_fric + dp_grav
            depth_accum += L * math.sin(theta)

        if depth_accum >= self.esp_depth:
            dp_total -= self.rho * self.g * self.pump_interp(flow_m3hr, "head")

        return self.wellhead_pressure + dp_total / 1e5

    def ipr(self, flow_m3hr: float) -> float:
        """Inflow Performance Relationship."""
        pbh = self.reservoir_pressure - flow_m3hr / self.PI
        return max(pbh, 0.0)

    def calculate(self) -> Dict[str, Any]:
        """
        Run the nodal analysis calculation.
        This is their main calculation method.
        """
        flows = np.linspace(1, 400, 200)
        p_vlp = np.array([self.vlp(f) for f in flows])
        p_ipr = np.array([self.ipr(f) for f in flows])

        # Find solution
        diff = np.abs(p_vlp - p_ipr)
        idx = np.argmin(diff)

        tolerance = 3.0  # bar

        if diff[idx] < tolerance:
            sol_flow = flows[idx]
            sol_pbh = p_vlp[idx]
            sol_head = self.pump_interp(sol_flow, "head")

            return {
                "success": True,
                "flowrate": float(sol_flow),
                "bottomhole_pressure": float(sol_pbh),
                "pump_head": float(sol_head),
                "convergence": float(diff[idx]),
                "flows": flows.tolist(),
                "p_vlp": p_vlp.tolist(),
                "p_ipr": p_ipr.tolist()
            }
        else:
            return {
                "success": False,
                "convergence": float(diff[idx]),
                "message": "No solution found",
                "flows": flows.tolist(),
                "p_vlp": p_vlp.tolist(),
                "p_ipr": p_ipr.tolist()
            }


def run_nodal_analysis(
        reservoir_pressure: float,
        productivity_index: float,
        well_name: Optional[str] = None,
        wellhead_pressure: float = 10.0,
        esp_depth: float = 500.0,
        fluid_density: float = 1000.0,
        viscosity: float = 1e-3,
        roughness: float = 1e-5,
        well_trajectory: Optional[List[Dict[str, float]]] = None,
        pump_curve: Optional[Dict[str, List[float]]] = None,
        return_curves: bool = False
) -> Dict[str, Any]:
    """
    Agent-callable wrapper for organizer's nodal analysis.

    This function:
    1. Takes agent-extracted parameters
    2. Calls organizer's model (unmodified)
    3. Returns agent-friendly results

    Args:
        reservoir_pressure: Reservoir pressure [bar]
        productivity_index: PI [m3/hr per bar]
        well_name: Well identifier
        wellhead_pressure: Wellhead pressure [bar]
        esp_depth: ESP depth [m]
        fluid_density: Fluid density [kg/m3]
        viscosity: Viscosity [Pa.s]
        roughness: Pipe roughness [m]
        well_trajectory: Well trajectory points
        pump_curve: Pump curve data
        return_curves: Include VLP/IPR curves

    Returns:
        Dict with results
    """
    logger.info(f"Running nodal analysis for {well_name or 'unknown well'}")

    try:
        # Create organizer's model with parameters
        model = OrganizerNodalAnalysis(
            rho=fluid_density,
            mu=viscosity,
            roughness=roughness,
            reservoir_pressure=reservoir_pressure,
            wellhead_pressure=wellhead_pressure,
            PI=productivity_index,
            esp_depth=esp_depth,
            pump_curve=pump_curve,
            well_trajectory=well_trajectory
        )

        # Run their calculation (unmodified)
        result = model.calculate()

        # Format for agent
        output = {
            "success": result["success"],
            "well_name": well_name,
            "convergence": result["convergence"]
        }

        if result["success"]:
            output.update({
                "flowrate": result["flowrate"],
                "bottomhole_pressure": result["bottomhole_pressure"],
                "pump_head": result["pump_head"],
                "message": f"Flowrate: {result['flowrate']:.2f} m3/hr "
                           f"Bottomhole Pressure: {result['bottomhole_pressure']:.2f} bar "
                           f"pump head: {result['pump_head']:.1f} m"
            })
            logger.info(output["message"])
        else:
            output["message"] = result.get("message", "No solution found")
            logger.warning(output["message"])

        # Include curves if requested
        if return_curves:
            output["curves"] = {
                "flows": result["flows"],
                "vlp": result["p_vlp"],
                "ipr": result["p_ipr"]
            }

        return output

    except Exception as e:
        logger.error(f"Nodal analysis failed: {e}", exc_info=True)
        return {
            "success": False,
            "well_name": well_name,
            "convergence": 999.0,
            "message": f"Calculation error: {str(e)}"
        }

    # ==================== UNIT CONVERSION HELPERS ====================


def convert_completion_params(
        completion_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert extracted completion parameters to nodal analysis inputs.

    Handles:
    - Unit conversions (psi→bar, ft→m, API→density)
    - Parameter mapping
    - Default values
    """

    def psi_to_bar(psi: float) -> float:
        return psi * 0.0689476

    def feet_to_meters(feet: float) -> float:
        return feet * 0.3048

    def inches_to_meters(inches: float) -> float:
        return inches * 0.0254

    def api_to_density(api: float) -> float:
        """Convert API gravity to density [kg/m3]."""
        sg = 141.5 / (131.5 + api)
        return sg * 1000.0

    nodal_params = {}

    # Map reservoir pressure
    if 'reservoir_pressure' in completion_params:
        value = completion_params['reservoir_pressure']
        # Check if already in bar or needs conversion
        if value > 100:  # Likely psi
            nodal_params['reservoir_pressure'] = psi_to_bar(value)
        else:  # Already bar
            nodal_params['reservoir_pressure'] = value

    # Map productivity index
    if 'productivity_index' in completion_params:
        # Assume in correct units (m3/hr/bar)
        nodal_params['productivity_index'] = completion_params['productivity_index']

    # Map ESP depth
    if 'tubing_depth' in completion_params:
        value = completion_params['tubing_depth']
        if value > 1000:  # Likely feet
            nodal_params['esp_depth'] = feet_to_meters(value)
        else:  # Already meters
            nodal_params['esp_depth'] = value

    # Map fluid density from oil gravity
    if 'oil_gravity' in completion_params:
        nodal_params['fluid_density'] = api_to_density(
            completion_params['oil_gravity']
        )

    # Map wellhead pressure
    if 'wellhead_pressure' in completion_params:
        value = completion_params['wellhead_pressure']
        if value > 100:  # Likely psi
            nodal_params['wellhead_pressure'] = psi_to_bar(value)
        else:
            nodal_params['wellhead_pressure'] = value

    # Build well trajectory if we have tubing data
    if 'tubing_id' in completion_params and 'tubing_depth' in completion_params:
        tubing_id_m = inches_to_meters(completion_params['tubing_id'])
        depth_m = feet_to_meters(completion_params['tubing_depth']) if completion_params['tubing_depth'] > 1000 else \
        completion_params['tubing_depth']

        # Simple vertical well trajectory
        nodal_params['well_trajectory'] = [
            {"MD": 0.0, "TVD": 0.0, "ID": 0.3397},  # Surface casing
            {"MD": depth_m / 3, "TVD": depth_m / 3, "ID": 0.2445},  # Intermediate
            {"MD": depth_m, "TVD": depth_m, "ID": tubing_id_m},  # Tubing
        ]

    return nodal_params

    # ==================== VALIDATION ====================

def validate_nodal_inputs(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters before running analysis.

    Returns: {valid: bool, errors: [], warnings: []}
    """
    errors = []
    warnings = []

    # Check required parameters
    required = ['reservoir_pressure', 'productivity_index']
    for param in required:
        if param not in params or params[param] is None:
            errors.append(f"Missing required parameter: {param}")

    # Check realistic ranges
    if 'reservoir_pressure' in params:
        p = params['reservoir_pressure']
        if p < 10 or p > 500:  # bar
            warnings.append(f"Unusual reservoir pressure: {p} bar (typical: 50-400 bar)")

    if 'productivity_index' in params:
        pi = params['productivity_index']
        if pi < 0.1 or pi > 50:
            warnings.append(f"Unusual PI: {pi} m3/hr/bar (typical: 1-20)")

    if 'esp_depth' in params:
        depth = params['esp_depth']
        if depth < 100 or depth > 5000:
            warnings.append(f"Unusual ESP depth: {depth} m (typical: 300-3000 m)")

    # Check well trajectory if provided
    if 'well_trajectory' in params:
        for i, point in enumerate(params['well_trajectory']):
            if point['MD'] < point['TVD']:
                errors.append(f"Trajectory point {i}: MD ({point['MD']}) < TVD ({point['TVD']}) - impossible!")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


if __name__ == '__main__':
    from pathlib import Path
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    run_nodal_analysis(reservoir_pressure=230.0, productivity_index= 5.0, well_name='Well 4', return_curves=True)
    print("Calculation finished")