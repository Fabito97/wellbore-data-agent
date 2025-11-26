from typing import Dict, Any, Optional
from langchain_core.tools import tool
from app.services.nodal_analysis import (
    run_nodal_analysis,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@tool('nodal_analysis')
def run_nodal_analysis_tool(
    reservoir_pressure: float,
    productivity_index: float,
    tubing_id: Optional[float] = None,
    tubing_depth: Optional[float] = None,
    oil_gravity: Optional[float] = None,
    wellhead_pressure: Optional[float] = None,
    pump_curve: Optional[Dict[str, Any]] = None,
    return_curves: bool = False,
) -> Dict[str, Any]:
    """
    Run nodal analysis with explicit parameters.

    This tool requires the following mandatory parameters:
      - reservoir_pressure (float) : reservoir pressure (bar or psi)
      - productivity_index (float) : productivity index (m3/hr per bar)

    Optional parameters:
      - tubing_id (float) : tubing inner diameter (inches)
      - tubing_depth (float) : tubing depth (ft or m)
      - oil_gravity (float) : API gravity
      - wellhead_pressure (float) : wellhead pressure (bar or psi)
      - pump_curve (dict) : {'flow': [...], 'head': [...]} or None
      - return_curves (bool) : include VLP/IPR curves in the result

    On success the tool returns a dict where 'message' EXACTLY matches the
    string format required by the caller:

    "Flowrate: {flow:.2f} m3/hr Bottomhole Pressure: {bhp:.2f} bar pump head: {head:.1f} m"

    The numeric values are taken from the organizer's model outputs.
    """
    try:
        # Inline conversion helpers (avoid external dict conversions)
        def psi_to_bar(psi: float) -> float:
            return psi * 0.0689476

        def feet_to_meters(feet: float) -> float:
            return feet * 0.3048

        def inches_to_meters(inches: float) -> float:
            return inches * 0.0254

        def api_to_density(api: float) -> float:
            sg = 141.5 / (131.5 + api)
            return sg * 1000.0

        # --- Normalize inputs ---
        # reservoir_pressure: detect psi vs bar
        rp = float(reservoir_pressure)
        if rp > 100:  # heuristic: likely psi
            rp = psi_to_bar(rp)

        pi = float(productivity_index)

        # tubing id (inches -> meters)
        tubing_id_m = None
        if tubing_id is not None:
            tubing_id_m = inches_to_meters(float(tubing_id))

        # tubing / esp depth
        esp_depth_m = None
        if tubing_depth is not None:
            td = float(tubing_depth)
            if td > 1000:  # likely feet
                esp_depth_m = feet_to_meters(td)
            else:
                esp_depth_m = td

        # fluid density from oil gravity
        fluid_density = None
        if oil_gravity is not None:
            fluid_density = api_to_density(float(oil_gravity))

        # wellhead pressure normalize
        whp = None
        if wellhead_pressure is not None:
            w = float(wellhead_pressure)
            if w > 100:  # likely psi
                whp = psi_to_bar(w)
            else:
                whp = w

        # well trajectory: simple vertical if tubing info present
        well_trajectory = None
        if tubing_id_m is not None and esp_depth_m is not None:
            depth_m = esp_depth_m
            well_trajectory = [
                {"MD": 0.0, "TVD": 0.0, "ID": 0.3397},
                {"MD": depth_m / 3.0, "TVD": depth_m / 3.0, "ID": 0.2445},
                {"MD": depth_m, "TVD": depth_m, "ID": tubing_id_m},
            ]

        # --- Inline validation (mirror validate_nodal_inputs logic) ---
        errors = []
        warnings = []

        if rp is None:
            errors.append("Missing required parameter: reservoir_pressure")
        if pi is None:
            errors.append("Missing required parameter: productivity_index")

        # realistic ranges
        if rp is not None and (rp < 10 or rp > 500):
            warnings.append(f"Unusual reservoir pressure: {rp} bar (typical: 10-400 bar)")
        if pi is not None and (pi < 0.1 or pi > 50):
            warnings.append(f"Unusual PI: {pi} m3/hr/bar (typical: 1-20)")
        if esp_depth_m is not None and (esp_depth_m < 100 or esp_depth_m > 5000):
            warnings.append(f"Unusual ESP depth: {esp_depth_m} m (typical: 300-3000 m)")

        if well_trajectory is not None:
            for i, point in enumerate(well_trajectory):
                if point['MD'] < point['TVD']:
                    errors.append(f"Trajectory point {i}: MD ({point['MD']}) < TVD ({point['TVD']}) - impossible!")

        if errors:
            return {
                'success': False,
                'message': 'Validation failed',
                'errors': errors,
                'warnings': warnings,
            }

        # Run nodal analysis with explicit keyword arguments
        result = run_nodal_analysis(
            reservoir_pressure=rp,
            productivity_index=pi,
            well_name=None,
            wellhead_pressure=whp,
            esp_depth=esp_depth_m,
            fluid_density=fluid_density,
            # viscosity=None,
            # roughness=None,
            well_trajectory=well_trajectory,
            pump_curve=pump_curve,
            return_curves=return_curves,
        )

        response = {
            'success': bool(result.get('success')),
            'convergence': float(result.get('convergence', 999.0)),
            'warnings': warnings,
            'errors': errors,
            'well_name': result.get('well_name'),
        }

        if result.get('success'):
            flow = float(result.get('flowrate'))
            bhp = float(result.get('bottomhole_pressure'))
            head = float(result.get('pump_head'))

            message = (
                f"Flowrate: {flow:.2f} m3/hr "
                f"Bottomhole Pressure: {bhp:.2f} bar "
                f"pump head: {head:.1f} m"
            )

            response.update({
                'flowrate_m3hr': flow,
                'bottomhole_pressure_bar': bhp,
                'pump_head_m': head,
                'message': message,
            })

            if return_curves:
                response['curves'] = result.get('curves', {})
        else:
            response['message'] = result.get('message', 'No solution found')

        return response

    except Exception as e:
        logger.exception(f"Nodal analysis tool error: {e}")
        return {
            'success': False,
            'message': f'Nodal analysis tool error: {str(e)}',
            'errors': [str(e)]
        }
