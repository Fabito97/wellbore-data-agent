def extraction_prompt(well_name, context):
    return f"""
Extract the following nodal analysis parameters from {well_name} completion report.

Look for these parameters in the tables and text:

CRITICAL PARAMETERS:
- Tubing ID (inner diameter in inches)
- Tubing depth (measured depth in feet or meters)
- Reservoir pressure (psi or bar)
- Productivity Index (PI)
- Oil gravity (API)

ADDITIONAL PARAMETERS:
- Gas gravity (specific gravity)
- Water cut (%)
- Gas-oil ratio (GOR in scf/stb)
- Production rate (bbl/day or m3/hr)
- Flowing pressure (psi or bar)

Context from {well_name}:
{context}

Extract all available parameters and format them clearly.
For each parameter, include:
1. Parameter name
2. Value with units
3. Source (which table or page)

If a parameter is not found, state "Not found in documents".
"""

def generate_summary_prompt(well_name, context):
    return f"""
Based on the following document excerpts from {well_name}, provide a comprehensive summary in approximately 200 words.

Focus on:
- Well identification and location
- Completion details (tubing, casing, perforations)
- Key operational parameters
- Important observations or findings

Context from documents:
{context}

Generate a clear, concise summary:
"""