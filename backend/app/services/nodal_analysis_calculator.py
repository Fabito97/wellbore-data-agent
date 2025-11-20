"""
Nodal Analysis Tool for AI Agent.

The agent will:
1. Extract parameters from documents via RAG
2. Call this tool with extracted parameters
3. Get back nodal analysis results
"""
from app.models.nodal_analysis import NodalAnalysisInput, NodalAnalysisResult, WellTrajectoryPoint, PumpCurve
import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Nodal Analysis Calculator ====================

class NodalAnalysisCalculator:
    """
    Performs nodal analysis calculations.
    Based on organizer's reference code.
    """

    def __init__(self):
        self.g = 9.81  # gravity [m/s2]

    def calculate(
            self,
            params: NodalAnalysisInput,
            return_curves: bool = False
    ) -> NodalAnalysisResult:
        """
        Run nodal analysis calculation.

        Args:
            params: Input parameters (extracted by agent)
            return_curves: Whether to return VLP/IPR curves for plotting

        Returns:
            NodalAnalysisResult with operating point
        """
        logger.info("Running nodal analysis calculation")

        try:
            # Build well segments from trajectory
            segments = self._build_segments(params.well_trajectory)

            # Calculate VLP and IPR curves
            flows = np.linspace(1, 400, 200)

            p_vlp = np.array([
                self._vlp(f, segments, params) for f in flows
            ])

            p_ipr = np.array([
                self._ipr(f, params) for f in flows
            ])

            # Find operating point (intersection)
            diff = np.abs(p_vlp - p_ipr)
            idx = np.argmin(diff)

            tolerance = 3.0  # bar

            if diff[idx] < tolerance:
                # Solution found
                sol_flow = flows[idx]
                sol_pbh = p_vlp[idx]
                sol_head = self._pump_interp(
                    sol_flow,
                    params.pump_curve.flow,
                    params.pump_curve.head
                )

                result = NodalAnalysisResult(
                    success=True,
                    flowrate=float(sol_flow),
                    bottomhole_pressure=float(sol_pbh),
                    pump_head=float(sol_head),
                    convergence_tolerance=float(diff[idx]),
                    message=f"Solution found: Q={sol_flow:.1f} m3/hr, BHP={sol_pbh:.1f} bar"
                )

                logger.info(result.message)
            else:
                # No solution
                result = NodalAnalysisResult(
                    success=False,
                    convergence_tolerance=float(diff[idx]),
                    message="No solution found - VLP and IPR curves do not intersect"
                )

                logger.warning(result.message)

            # Optionally include curves for plotting
            if return_curves:
                result.vlp_curve = {
                    "flow": flows.tolist(),
                    "pressure": p_vlp.tolist()
                }
                result.ipr_curve = {
                    "flow": flows.tolist(),
                    "pressure": p_ipr.tolist()
                }

            return result

        except Exception as e:
            logger.error(f"Nodal analysis failed: {e}", exc_info=True)
            return NodalAnalysisResult(
                success=False,
                convergence_tolerance=999.0,
                message=f"Calculation error: {str(e)}"
            )

    def _build_segments(
            self,
            trajectory: List[WellTrajectoryPoint]
    ) -> List[Tuple[float, float, float]]:
        """Build well segments from trajectory points."""
        segments = []

        for i in range(1, len(trajectory)):
            MD = trajectory[i].MD - trajectory[i - 1].MD
            TVD = trajectory[i].TVD - trajectory[i - 1].TVD
            D = trajectory[i].ID
            L = MD

            # Handle vertical sections
            if MD == 0:
                theta = math.pi / 2  # 90 degrees
            else:
                theta = math.atan2(TVD, MD)

            segments.append((L, D, theta))

        return segments

    def _vlp(
            self,
            flow_m3hr: float,
            segments: List[Tuple[float, float, float]],
            params: NodalAnalysisInput
    ) -> float:
        """Calculate Vertical Lift Performance."""
        q = flow_m3hr / 3600.0  # m3/hr to m3/s
        dp_total = 0.0
        depth_accum = 0.0

        for (L, D, theta) in segments:
            A = math.pi * D ** 2 / 4.0
            u = q / A if A > 0 else 0.0

            # Reynolds number
            Re = params.rho * abs(u) * D / params.mu

            # Friction factor (Swamee-Jain)
            f = self._swamee_jain(Re, D, params.roughness)

            # Pressure drops
            dp_fric = f * (L / D) * (params.rho * u ** 2 / 2.0) if D > 0 else 0.0
            dp_grav = params.rho * self.g * L * math.sin(theta)

            dp_total += dp_fric + dp_grav
            depth_accum += L * math.sin(theta)

        # Add pump contribution if below ESP depth
        if depth_accum >= params.esp_depth and params.pump_curve:
            pump_head = self._pump_interp(
                flow_m3hr,
                params.pump_curve.flow,
                params.pump_curve.head
            )
            dp_total -= params.rho * self.g * pump_head

        # Return pressure in bar
        return params.wellhead_pressure + dp_total / 1e5

    def _ipr(self, flow_m3hr: float, params: NodalAnalysisInput) -> float:
        """Calculate Inflow Performance Relationship."""
        pbh = params.reservoir_pressure - flow_m3hr / params.PI
        return max(pbh, 0.0)

    def _swamee_jain(self, Re: float, D: float, roughness: float) -> float:
        """Swamee-Jain friction factor correlation."""
        if Re <= 0:
            return 0.0

        term1 = roughness / (3.7 * D)
        term2 = 5.74 / (Re ** 0.9)

        try:
            return 0.25 / (math.log10(term1 + term2) ** 2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _pump_interp(
            self,
            flow: float,
            flow_points: List[float],
            head_points: List[float]
    ) -> float:
        """Interpolate pump curve."""
        return float(np.interp(flow, flow_points, head_points))


# ==================== Agent Tool Function ====================

def run_nodal_analysis(
        reservoir_pressure: float,
        productivity_index: float,
        well_trajectory: Optional[List[Dict[str, float]]] = None,
        wellhead_pressure: float = 10.0,
        esp_depth: float = 500.0,
        fluid_density: float = 1000.0,
        viscosity: float = 1e-3,
        **kwargs
) -> Dict[str, Any]:
    """
    Agent-callable function for nodal analysis.

    The agent extracts parameters from documents and calls this function.

    Args:
        reservoir_pressure: Reservoir pressure [bar]
        productivity_index: PI [m3/hr per bar]
        well_trajectory: Optional trajectory data
        wellhead_pressure: Wellhead pressure [bar]
        esp_depth: ESP depth [m]
        fluid_density: Fluid density [kg/m3]
        viscosity: Viscosity [Pa.s]

    Returns:
        Dict with analysis results

    Example agent usage:
        # Agent extracts from document
        reservoir_pressure = 230.0  # extracted from completion report
        PI = 5.0  # extracted from well test

        # Agent calls tool
        result = run_nodal_analysis(
            reservoir_pressure=reservoir_pressure,
            productivity_index=PI
        )

        # Agent uses result in response
        if result['success']:
            return f"Optimal production: {result['flowrate']:.1f} m3/hr"
    """
    # Build input from agent-extracted parameters
    input_params = NodalAnalysisInput(
        reservoir_pressure=reservoir_pressure,
        PI=productivity_index,
        wellhead_pressure=wellhead_pressure,
        esp_depth=esp_depth,
        rho=fluid_density,
        mu=viscosity
    )

    # Override trajectory if provided
    if well_trajectory:
        try:
            input_params.well_trajectory = [
                WellTrajectoryPoint(**point) for point in well_trajectory
            ]
        except Exception as e:
            logger.warning(f"Failed to parse trajectory: {e}")

    # Run calculation
    calculator = NodalAnalysisCalculator()
    result = calculator.calculate(input_params, return_curves=True)

    # Return as dict for agent
    return result.model_dump()


# ==================== Helper for Parameter Mapping ====================

def map_completion_params_to_nodal(
        completion_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Map completion report parameters to nodal analysis input.

    Handles unit conversions and parameter mapping.

    Args:
        completion_params: Parameters extracted from completion report

    Returns:
        Dict ready for run_nodal_analysis()
    """

    # Unit conversions
    def psi_to_bar(psi: float) -> float:
        return psi * 0.0689476

    def feet_to_meters(feet: float) -> float:
        return feet * 0.3048

    def api_to_density(api: float) -> float:
        """Convert API gravity to density [kg/m3]."""
        sg = 141.5 / (131.5 + api)  # specific gravity
        return sg * 1000.0

    nodal_params = {}

    # Map reservoir pressure
    if 'reservoir_pressure' in completion_params:
        nodal_params['reservoir_pressure'] = psi_to_bar(
            completion_params['reservoir_pressure']
        )

    # Map productivity index (needs conversion if in different units)
    if 'productivity_index' in completion_params:
        # Assuming PI is in STB/day/psi, convert to m3/hr/bar
        pi_stb = completion_params['productivity_index']
        # 1 STB = 0.1589873 m3, 1 day = 24 hr, 1 psi = 0.0689476 bar
        nodal_params['productivity_index'] = pi_stb * 0.1589873 * 24 / 0.0689476

    # Map ESP depth
    if 'tubing_depth' in completion_params:
        nodal_params['esp_depth'] = feet_to_meters(
            completion_params['tubing_depth']
        )

    # Map fluid density from oil gravity
    if 'oil_gravity' in completion_params:
        nodal_params['fluid_density'] = api_to_density(
            completion_params['oil_gravity']
        )

    # Map wellhead pressure
    if 'wellhead_pressure' in completion_params:
        nodal_params['wellhead_pressure'] = psi_to_bar(
            completion_params['wellhead_pressure']
        )

    return nodal_params