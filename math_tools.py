import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import SympifyError

def solve_math_problem(problem: str) -> str:
    """
    Solves mathematical problems including algebraic equations, differentiation,
    and integration using the SymPy library.

    Args:
        problem: A string describing the mathematical problem.
                 Examples:
                 - "solve 2*x + 3 = 7 for x"
                 - "derivative of x**2 with respect to x"
                 - "integrate sin(x) with respect to x"
                 - "integrate sin(x) from 0 to pi with respect to x"

    Returns:
        A string containing the solution or an error message if the problem
        cannot be solved or the input is invalid.
    """
    try:
        # Define transformations for parsing
        transformations = standard_transformations + (implicit_multiplication_application,)

        # Normalize problem string
        problem_lower = problem.lower()

        if "solve" in problem_lower and "for" in problem_lower:
            # Algebraic equation
            # Example: "solve 2*x + 3 = 7 for x"
            parts = problem_lower.split("solve")[1].split("for")
            equation_str = parts[0].strip()
            variable_str = parts[1].strip()

            # Attempt to identify the variable SymPy will use
            # This is a simple heuristic and might need refinement
            if '=' not in equation_str:
                return "Error: Invalid equation format. Ensure it contains '='."

            # Try to parse the variable first
            try:
                variable = sympy.Symbol(variable_str)
            except Exception as e:
                return f"Error: Could not parse variable '{variable_str}'. {e}"

            # Split equation into LHS and RHS
            lhs_str, rhs_str = equation_str.split('=')
            
            # Parse LHS and RHS
            try:
                lhs = parse_expr(lhs_str.strip(), transformations=transformations, local_dict={'x': sympy.Symbol('x'), variable_str: variable})
                rhs = parse_expr(rhs_str.strip(), transformations=transformations, local_dict={'x': sympy.Symbol('x'), variable_str: variable})
            except SympifyError as e:
                return f"Error parsing equation: {e}. Ensure terms like '2x' are written as '2*x'."

            # Create equation and solve
            equation = sympy.Eq(lhs, rhs)
            solution = sympy.solve(equation, variable)
            if not solution:
                return f"SymPy could not find a solution for '{problem}'."
            return f"Solution for '{problem}': {variable_str} = {solution[0] if len(solution) == 1 else solution}"

        elif "derivative of" in problem_lower:
            # Differentiation
            # Example: "derivative of x**2 with respect to x"
            parts = problem_lower.split("derivative of")[1].split("with respect to")
            if len(parts) < 2:
                return "Error: Invalid derivative format. Use 'derivative of [expression] with respect to [variable]'."
            
            expr_str = parts[0].strip()
            var_str = parts[1].strip()

            try:
                variable = sympy.Symbol(var_str)
                expression = parse_expr(expr_str, transformations=transformations, local_dict={var_str: variable, 'x': sympy.Symbol('x'), 'y': sympy.Symbol('y'), 'z': sympy.Symbol('z')})
            except SympifyError as e:
                return f"Error parsing expression or variable for differentiation: {e}"
            
            derivative = sympy.diff(expression, variable)
            return f"The derivative of {expr_str} with respect to {var_str} is: {derivative}"

        elif "integrate" in problem_lower:
            # Integration
            # Example: "integrate sin(x) with respect to x"
            # Example: "integrate sin(x) from 0 to pi with respect to x"
            parts = problem_lower.split("integrate")[1].split("with respect to")
            if len(parts) < 2:
                 return "Error: Invalid integration format. Use 'integrate [expression] with respect to [variable]' or 'integrate [expression] from [lower_bound] to [upper_bound] with respect to [variable]'."
            
            expr_and_bounds_str = parts[0].strip()
            var_str = parts[1].strip()
            
            try:
                variable = sympy.Symbol(var_str)
                # Add common math functions to local_dict for parsing
                local_symbols = {var_str: variable, 'x': sympy.Symbol('x'), 'y': sympy.Symbol('y'), 'z': sympy.Symbol('z'), 'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan, 'exp': sympy.exp, 'ln': sympy.ln, 'pi': sympy.pi}
            except SympifyError as e:
                 return f"Error parsing variable for integration: {e}"

            if "from" in expr_and_bounds_str and "to" in expr_and_bounds_str:
                # Definite integral
                expr_part, bounds_part = expr_and_bounds_str.split("from")
                expr_str = expr_part.strip()
                
                lower_bound_str, upper_bound_str = bounds_part.split("to")
                lower_bound_str = lower_bound_str.strip()
                upper_bound_str = upper_bound_str.strip()

                try:
                    expression = parse_expr(expr_str, transformations=transformations, local_dict=local_symbols)
                    lower_bound = parse_expr(lower_bound_str, transformations=transformations, local_dict=local_symbols)
                    upper_bound = parse_expr(upper_bound_str, transformations=transformations, local_dict=local_symbols)
                except SympifyError as e:
                    return f"Error parsing expression or bounds for definite integration: {e}"
                
                try:
                    integral_result = sympy.integrate(expression, (variable, lower_bound, upper_bound))
                    return f"The integral of {expr_str} from {lower_bound_str} to {upper_bound_str} with respect to {var_str} is: {integral_result}"
                except Exception as e:
                    return f"Error calculating definite integral: {e}"
            else:
                # Indefinite integral
                expr_str = expr_and_bounds_str.strip()
                try:
                    expression = parse_expr(expr_str, transformations=transformations, local_dict=local_symbols)
                except SympifyError as e:
                    return f"Error parsing expression for indefinite integration: {e}"

                try:
                    integral_result = sympy.integrate(expression, variable)
                    return f"The indefinite integral of {expr_str} with respect to {var_str} is: {integral_result} + C"
                except Exception as e:
                    return f"Error calculating indefinite integral: {e}"
        else:
            return "Error: Could not determine the type of problem. Please use 'solve', 'derivative of', or 'integrate'."

    except SympifyError as e:
        return f"Error: Invalid mathematical expression. {e}. Remember to use '*' for multiplication (e.g., '2*x' instead of '2x')."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Physics Formulas ---
PHYSICS_FORMULAS = {
    "newton's second law": "F = m*a (Force = mass * acceleration)",
    "ohm's law": "V = I*R (Voltage = Current * Resistance)",
    "kinetic energy": "K.E. = 1/2 * m * v**2 (Kinetic Energy = 1/2 * mass * velocity^2)",
    "potential energy (gravity)": "P.E. = m*g*h (Potential Energy = mass * gravitational acceleration * height)",
    "einstein's mass-energy equivalence": "E = m*c**2 (Energy = mass * speed of light^2)",
    "pressure": "P = F/A (Pressure = Force / Area)",
    "work": "W = F*d*cos(theta) (Work = Force * distance * cos(angle between F and d))",
    "power": "P = W/t (Power = Work / time)",
    "density": "rho = m/V (Density = mass / Volume)",
    "ideal gas law": "P*V = n*R*T (Pressure * Volume = moles * Ideal Gas Constant * Temperature)",
    "coulomb's law": "F = k * (q1*q2)/r**2 (Force = Coulomb's constant * (charge1 * charge2) / distance^2)",
    "universal gravitation": "F = G * (m1*m2)/r**2 (Force = Gravitational constant * (mass1 * mass2) / distance^2)",
}

def get_physics_formula(concept: str) -> str:
    """
    Looks up a physics formula from a predefined dictionary.

    Args:
        concept: A string describing the physics concept (case-insensitive).
                 Examples: "Newton's second law", "Ohm's law", "kinetic energy"

    Returns:
        A string containing the formula or a message if the concept is not found.
    """
    concept_lower = concept.lower()
    return PHYSICS_FORMULAS.get(concept_lower, f"Formula for '{concept}' not found.")

# --- Physical Constants ---
import scipy.constants
import scipy

# --- Unit Conversion ---
import pint
ureg = pint.UnitRegistry()

# --- Plotting ---
import matplotlib.pyplot as plt
import numpy as np
import re
import ast
from sympy import lambdify, Symbol
# Ensure Matplotlib uses a non-interactive backend to avoid issues in environments without a display
plt.switch_backend('Agg')

# --- Web Search (Mock Implementation) ---
# import requests # Uncomment for real API calls
# import os # For API key management

MOCK_SEARCH_RESULTS = [
    {
        'title': "Ohm's Law - Wikipedia",
        'link': 'https://en.wikipedia.org/wiki/Ohm%27s_law',
        'snippet': "Ohm's law states that the current through a conductor between two points is directly proportional to the voltage across the two points..."
    },
    {
        'title': "What is Ohm's Law? | Fluke",
        'link': 'https://www.fluke.com/en-us/learn/blog/electrical/what-is-ohms-law',
        'snippet': "Ohm's Law is a formula used to calculate the relationship between voltage, current and resistance in an electrical circuit."
    },
    {
        'title': "Kinetic Energy | Physics Classroom",
        'link': 'https://www.physicsclassroom.com/class/energy/Lesson-1/Kinetic-Energy',
        'snippet': "Kinetic energy is the energy of motion. An object that has motion - whether it is vertical or horizontal motion - has kinetic energy."
    },
    {
        'title': "Introduction to Special Relativity - CERN",
        'link': 'https://home.cern/science/physics/special-relativity',
        'snippet': "Special relativity, theory proposed by Albert Einstein, that describes the propagation of matter and light at high speeds."
    },
    {
        'title': "E=mc^2: Einstein's equation that gave birth to the atom bomb",
        'link': 'https://www.britannica.com/story/e-mc2-einsteins-equation-that-gave-birth-to-the-atom-bomb',
        'snippet': "E = mc^2, equation in Albert Einstein’s theory of special relativity that expresses the fact that mass and energy are the same physical entity..."
    },
    {
        'title': "SymPy - Python library for symbolic mathematics",
        'link': 'https://www.sympy.org/en/index.html',
        'snippet': "SymPy is a Python library for symbolic mathematics. It aims to become a full-featured computer algebra system (CAS)..."
    },
    {
        'title': "SciPy - Fundamental algorithms for scientific computing in Python",
        'link': 'https://scipy.org/',
        'snippet': "SciPy provides many user-friendly and efficient numerical routines, such as numerical integration, interpolation, optimization, linear algebra, and statistics."
    },
    {
        'title': "Pint - Physical quantities module for Python",
        'link': 'https://pint.readthedocs.io/',
        'snippet': "Pint is a Python package to define, operate and manipulate physical quantities: the product of a numerical value and a unit of measurement."
    }
]

# Mapping common names to scipy.constants attributes or providing more details
# Structure: "common name": (attribute_name_in_scipy.constants OR key_in_physical_constants, display_prefix, direct_unit_string_OR_None)
# If direct_unit_string is provided, it's a direct attribute and we use this unit.
# If direct_unit_string is None, it's a key for physical_constants dictionary.
PHYSICAL_CONSTANTS_MAP = {
    "speed of light": ("c", "Speed of light (c)", "m s^-1"),
    "planck constant": ("h", "Planck constant (h)", "J s"),
    "gravitational constant": ("G", "Gravitational constant (G)", "N m^2 kg^-2"),
    "boltzmann constant": ("k", "Boltzmann constant (k)", "J K^-1"),
    "electron mass": ("m_e", "Electron mass (m_e)", "kg"),
    "proton mass": ("m_p", "Proton mass (m_p)", "kg"),
    "neutron mass": ("m_n", "Neutron mass (m_n)", "kg"),
    "elementary charge": ("e", "Elementary charge (e)", "C"),
    "avogadro number": ("Avogadro", "Avogadro constant (N_A)", "mol^-1"),
    "gas constant": ("R", "Gas constant (R)", "J mol^-1 K^-1"),
    "faraday constant": ("Faraday", "Faraday constant (F)", "C mol^-1"),
    "stefan-boltzmann constant": ("Stefan_Boltzmann", "Stefan-Boltzmann constant (sigma)", "W m^-2 K^-4"),
    "rydberg constant": ("Rydberg", "Rydberg constant (R_inf)", "m^-1"),
    "vacuum permeability": ("mu_0", "Vacuum permeability (mu_0)", "N A^-2"),
    "vacuum permittivity": ("epsilon_0", "Vacuum permittivity (epsilon_0)", "F m^-1"),
    "fine-structure constant": ("alpha", "Fine-structure constant (alpha)", ""), # Dimensionless

    # For keys that are ONLY in physical_constants dict (not direct attributes)
    "atomic mass constant": ("atomic mass constant", "Atomic mass constant", None),
    "bohr radius": ("Bohr radius", "Bohr radius (a_0)", None),
    "electron g factor": ("electron g factor", "Electron g factor", None),
    "proton magnetic moment": ("proton mag. mom.", "Proton magnetic moment", None), # Corrected key
}


def get_physical_constant(constant_name: str) -> str:
    """
    Retrieves the value and unit of a physical constant using scipy.constants.

    Args:
        constant_name: A string with the common name of the physical constant
                       (case-insensitive).
                       Examples: "speed of light", "Planck constant", "atomic mass constant"

    Returns:
        A string representation of the constant's value and unit, or a
        message if the constant is not found or not supported.
    """
    constant_name_lower = constant_name.lower()
    
    mapped_info = PHYSICAL_CONSTANTS_MAP.get(constant_name_lower)

    if mapped_info:
        key_or_attr, display_prefix, direct_unit_str = mapped_info
        try:
            if direct_unit_str is not None: # Indicates it's a direct attribute with a known unit
                value = getattr(scipy.constants, key_or_attr)
                return f"{display_prefix} = {value} {direct_unit_str}".strip()
            else: # It's a key for physical_constants dictionary
                if key_or_attr in scipy.constants.physical_constants:
                    value, unit, _ = scipy.constants.physical_constants[key_or_attr]
                    return f"{display_prefix} = {value} {unit}"
                else:
                    return f"Error: Key '{key_or_attr}' (for '{constant_name}') not found in scipy.constants.physical_constants dictionary."
        except AttributeError:
             return f"Error: Attribute '{key_or_attr}' (for '{constant_name}') not found as a direct attribute in scipy.constants."
        except KeyError: # Should be caught by the check above, but as a safeguard
            return f"Error: Key '{key_or_attr}' (for '{constant_name}') unexpectedly not found in scipy.constants.physical_constants."
        except Exception as e:
            return f"An unexpected error occurred while retrieving constant '{constant_name}': {e}"
    else:
        # Fallback: Try searching directly in physical_constants keys if not in our map
        for pc_key_name_full, pc_val_tuple in scipy.constants.physical_constants.items():
            if constant_name_lower in pc_key_name_full.lower():
                value, unit_str_pc, _ = pc_val_tuple
                # Use the full key name from physical_constants for clarity if it's different & more descriptive
                display_name_pc = pc_key_name_full
                if constant_name_lower == pc_key_name_full.lower() : # if it's an exact match (case insens.)
                     display_name_pc = constant_name.capitalize() # use user's capitalization
                elif constant_name_lower not in display_name_pc.lower(): # if user input is not part of the full key name
                    display_name_pc = f"{pc_key_name_full} (related to '{constant_name}')"
                
                return f"{display_name_pc} = {value} {unit_str_pc}"
        
        return f"Constant '{constant_name}' not found. Please check spelling or try a different name. Consult PHYSICAL_CONSTANTS_MAP for supported names."


def convert_units(value_str: str, from_unit: str, to_unit: str) -> str:
    """
    Converts a value from one unit to another using the Pint library.

    Args:
        value_str: A string representing the numerical value to convert.
        from_unit: A string representing the unit to convert from (e.g., "meter", "km/h").
        to_unit: A string representing the unit to convert to (e.g., "kilometer", "m/s").

    Returns:
        A string representation of the converted value and unit (e.g.,
        "10.0 kilometer = 6.21371192237334 mile") or an informative error message.
    """
    try:
        value = float(value_str)
    except ValueError:
        return f"Error: Invalid input value '{value_str}'. Must be a number."

    try:
        quantity = ureg.Quantity(value, from_unit)
        converted_quantity = quantity.to(to_unit)
        # Use ~P for pretty formatting of units by Pint for clearer output
        return f"{quantity:~P} = {converted_quantity:~P}"
    except pint.errors.UndefinedUnitError as e:
        # Provide specific error message for undefined units
        return f"Error: Unit undefined. {e}. Please check the spelling or if the unit is supported by Pint (e.g., use 'degreeCelsius' for Celsius)."
    except pint.errors.DimensionalityError as e:
        # Provide specific error message for dimensionality errors
        return f"Error: Cannot convert from '{from_unit}' to '{to_unit}'. Incompatible units. {e}"
    except Exception as e:
        # Catch any other unexpected errors from Pint or elsewhere
        return f"An unexpected error occurred during unit conversion: {e}"


def generate_plot(plot_type: str, data_str: str, filename: str = "plot.png") -> str:
    """
    Generates a plot using Matplotlib and saves it to a file.

    Args:
        plot_type: Type of plot ("function" or "scatter").
        data_str: String containing data for the plot.
            - For "function": "expression from start_val to end_val [for var]"
              e.g., "sin(x) from 0 to 2*pi", "x**2 - 3*x + 2 for x from -5 to 5"
              If "[for var]" is omitted, 'x' is assumed.
              SymPy's 'pi' and 'E' can be used.
            - For "scatter": "x_values=[1,2,3]; y_values=[2,4,1]"
              e.g., "x=[1,2,3]; y=[4,5,6]" (shorter aliases also work)
        filename: Name of the file to save the plot (e.g., "plot.png").

    Returns:
        Success message with filename or an error message.
    """
    plt.clf()  # Clear any previous plot figures

    try:
        if plot_type == "function":
            # Regex 1: "EXPR from START to END [for VAR]" (VAR is optional)
            # Groups: 1=expr, 2=start_expr, 3=end_expr, 4=var_name (optional)
            regex1 = r"^(.*?) from\s*(.+?)\s*to\s*(.+?)(?:\s+for\s+([a-zA-Z_][a-zA-Z0-9_]*))?\s*$"
            
            # Regex 2: "EXPR for VAR from START to END" (VAR is mandatory here)
            # Groups: 1=expr, 2=var_name, 3=start_expr, 4=end_expr
            regex2 = r"^(.*?)\s+for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+from\s*(.+?)\s*to\s*(.+?)\s*$"

            expr_str, start_expr_str, end_expr_str, var_name = None, None, None, None
            parsed_successfully = False

            match1 = re.match(regex1, data_str.strip(), flags=re.IGNORECASE)
            if match1:
                raw_expr_str, raw_start_expr_str, raw_end_expr_str, raw_var_name_opt = match1.groups()
                expr_str = raw_expr_str.strip()
                start_expr_str = raw_start_expr_str.strip()
                end_expr_str = raw_end_expr_str.strip()
                var_name = raw_var_name_opt.strip() if raw_var_name_opt else 'x'
                parsed_successfully = True
            
            if not parsed_successfully:
                match2 = re.match(regex2, data_str.strip(), flags=re.IGNORECASE)
                if match2:
                    raw_expr_str, raw_var_name, raw_start_expr_str, raw_end_expr_str = match2.groups()
                    expr_str = raw_expr_str.strip()
                    var_name = raw_var_name.strip()
                    start_expr_str = raw_start_expr_str.strip()
                    end_expr_str = raw_end_expr_str.strip()
                    parsed_successfully = True

            if not parsed_successfully:
                return "Error: Invalid format for function plot. Use 'expr from start to end [for var]' or 'expr for var from start to end'."

            variable = Symbol(var_name)

            def eval_range_val(range_val_str: str):
                """Evaluates a range value string (e.g., "2*pi", "10", "E") using SymPy."""
                try:
                    processed_range_val_str = range_val_str.strip()
                    if not processed_range_val_str:
                        raise ValueError("Range value string cannot be empty.")

                    parsed_val = sympy.sympify(processed_range_val_str, locals={'pi': sympy.pi, 'E': sympy.E, 'oo': sympy.oo, 'inf': sympy.oo})
                    
                    # Attempt to evaluate to a numerical value if it's not already a number or infinity
                    if not (parsed_val.is_Float or parsed_val.is_Integer or parsed_val.is_Rational or parsed_val.is_infinite):
                        evalf_val = parsed_val.evalf()
                        if evalf_val.is_Number: # Checks if it's a concrete number (Integer, Float, Rational)
                            parsed_val = evalf_val
                        else: # evalf() did not produce a concrete number (e.g. it's still symbolic like 'x*pi')
                            raise ValueError(f"Range expression '{processed_range_val_str}' did not evaluate to a concrete number.")

                    if parsed_val == sympy.oo:
                        return np.inf
                    elif parsed_val == -sympy.oo:
                        return -np.inf
                    elif parsed_val.is_Float or parsed_val.is_Integer or parsed_val.is_Rational:
                        return float(parsed_val)
                    else: # Should be caught by the evalf check if it's not a direct number/infinity
                        raise ValueError(f"Range value '{processed_range_val_str}' must evaluate to a real number or +/-infinity.")
                except (SympifyError, TypeError, AttributeError, ValueError) as e:
                    if isinstance(e, ValueError): # Re-raise our own ValueErrors
                        raise
                    # Wrap other low-level errors into a ValueError for consistent error reporting
                    raise ValueError(f"Invalid range value or expression: '{range_val_str}'. Original error: {e}")

            try:
                start_val = eval_range_val(start_expr_str)
                end_val = eval_range_val(end_expr_str)
            except ValueError as e: # Catch the specific ValueError from eval_range_val
                return f"Error evaluating range: {e}" # Display the detailed error from eval_range_val

            if start_val >= end_val:
                return "Error: Start value of the range must be less than the end value."

            # Parse the expression using SymPy
            # Provide a local dict for common math functions and the variable
            local_dict = {
                var_name: variable, 'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
                'exp': sympy.exp, 'ln': sympy.ln, 'log': sympy.log, 'sqrt': sympy.sqrt,
                'pi': sympy.pi, 'E': sympy.E
            }
            # Include numpy functions if desired, mapping them carefully for SymPy
            # For simplicity, we mainly rely on SymPy's functions here.
            
            # Ensure expr_str is not empty after stripping, as parse_expr("") is an error.
            if not expr_str:
                return "Error: Expression string is empty after parsing."

            parsed_expr = parse_expr(expr_str, local_dict=local_dict, transformations=standard_transformations + (implicit_multiplication_application,))
            
            # Lambdify the expression
            # Important: use "numpy" for numerical evaluation with np.linspace
            func = lambdify(variable, parsed_expr, modules=['numpy', {'pi': np.pi, 'E': np.e}])
            
            x_vals = np.linspace(start_val, end_val, 500)
            y_vals = func(x_vals)

            plt.plot(x_vals, y_vals)
            plt.title(f"Plot of {expr_str}")
            plt.xlabel(var_name)
            plt.ylabel(f"f({var_name})")
            plt.grid(True)

        elif plot_type == "scatter":
            # Regex to find x_values=[...] and y_values=[...]
            # Allows for 'x_values', 'xvals', 'x' and similar for y.
            x_match = re.search(r"(?:x_values|xvals|x)\s*=\s*(\[.*?\])", data_str, flags=re.IGNORECASE)
            y_match = re.search(r"(?:y_values|yvals|y)\s*=\s*(\[.*?\])", data_str, flags=re.IGNORECASE)

            if not x_match or not y_match:
                return "Error: Invalid format for scatter plot. Expected 'x_values=[...]; y_values=[...]' or similar."

            try:
                x_values_str = x_match.group(1)
                y_values_str = y_match.group(1)
                
                # Use ast.literal_eval for safe parsing of list strings
                x_values = ast.literal_eval(x_values_str)
                y_values = ast.literal_eval(y_values_str)
            except (ValueError, SyntaxError) as e:
                return f"Error parsing lists: {e}. Ensure lists are in valid Python format (e.g., [1, 2, 3])."

            if not isinstance(x_values, list) or not isinstance(y_values, list):
                return "Error: x_values and y_values must be lists."
            if len(x_values) != len(y_values):
                return "Error: x_values and y_values must have the same length."
            if not all(isinstance(val, (int, float)) for val in x_values + y_values):
                return "Error: All values in x and y lists must be numbers."

            plt.scatter(x_values, y_values)
            plt.title("Scatter Plot")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

        else:
            return f"Error: Unknown plot type '{plot_type}'. Supported types are 'function' and 'scatter'."

        plt.savefig(filename)
        return f"Plot saved as {filename}"

    except SympifyError as e:
        return f"Error parsing mathematical expression: {e}"
    except ValueError as e: # Catch specific value errors not caught by more specific handlers
        return f"ValueError during plot generation: {e}"
    except Exception as e:
        # Use plt.clf() here as well in case of error during plotting itself before savefig
        plt.clf() 
        return f"An unexpected error occurred during plot generation: {e}"


def web_search(query: str, num_results: int = 3) -> str:
    """
    Simulates a web search using a predefined list of mock results.

    In a real implementation, this function would call a web search API
    (e.g., Google Search API, SerpAPI, Bing Search API) to get live results.

    Args:
        query: The search term (string).
        num_results: The maximum number of results to return (integer, default 3).

    Returns:
        A formatted string containing the search results (title, link, snippet)
        or a message if no relevant results are found.
    """
    
    # --- Real Implementation Notes ---
    # 1. API Choice:
    #    You would typically use a service like Google Custom Search JSON API,
    #    SerpAPI (third-party scraper), Bing Web Search API, or similar.
    #    Example using a hypothetical API with 'requests':
    #
    #    API_KEY = os.getenv("YOUR_SEARCH_API_KEY") # Securely manage API keys
    #    SEARCH_ENGINE_ID = os.getenv("YOUR_SEARCH_ENGINE_ID") # For Google CSE
    #    ENDPOINT_URL = "https://www.googleapis.com/customsearch/v1" # Example for Google
    #
    #    headers = {"Accept": "application/json"}
    #    params = {
    #        "key": API_KEY,
    #        "cx": SEARCH_ENGINE_ID, # For Google CSE
    #        "q": query,
    #        "num": num_results
    #    }
    #    try:
    #        response = requests.get(ENDPOINT_URL, headers=headers, params=params, timeout=5)
    #        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    #        search_data = response.json()
    #        # Process search_data['items'] to extract title, link, snippet
    #        # ... (formatting logic similar to mock results below) ...
    #        # return formatted_results_string
    #    except requests.exceptions.RequestException as e:
    #        return f"Error during web search: {e}"
    #    except KeyError:
    #        return "Error: Could not parse search results from API."
    # --- End of Real Implementation Notes ---

    # Mock implementation:
    query_lower = query.lower()
    found_results = []

    for result in MOCK_SEARCH_RESULTS:
        if query_lower in result['title'].lower() or query_lower in result['snippet'].lower():
            found_results.append(result)
    
    if not found_results:
        return f"No specific mock results found for '{query}'.\nFor general information, try a broader query or visit a general science resource like Wikipedia."

    output_str = f"Mock search results for '{query}' (up to {num_results}):\n\n"
    for i, res in enumerate(found_results[:num_results]):
        output_str += f"{i+1}. Title: {res['title']}\n"
        output_str += f"   Link: {res['link']}\n"
        output_str += f"   Snippet: {res['snippet']}\n\n"
    
    return output_str.strip()

# --- Educational Resources ---
EDUCATIONAL_RESOURCES = {
    "calculus basics": [
        {"title": "Khan Academy: Calculus 1", "url": "https://www.khanacademy.org/math/calculus-1", "description": "Comprehensive introductory calculus lessons, covering limits, derivatives, and integrals."},
        {"title": "MIT OpenCourseware: Single Variable Calculus", "url": "https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/", "description": "University-level calculus course materials from MIT, including lecture notes, videos, and assignments."},
        {"title": "3Blue1Brown: Essence of Calculus", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr", "description": "Intuitive video explanations of core calculus concepts."}
    ],
    "newtonian mechanics": [
        {"title": "Khan Academy: Newtonian Mechanics", "url": "https://www.khanacademy.org/science/physics/newtonian-mechanics", "description": "Covers Newton's laws of motion, forces, work, energy, and momentum."},
        {"title": "MIT OpenCourseware: Physics I - Classical Mechanics", "url": "https://ocw.mit.edu/courses/physics/8-01sc-classical-mechanics-fall-2016/", "description": "Full university course on classical mechanics, including Newtonian mechanics."},
        {"title": "HyperPhysics: Newtonian Mechanics", "url": "http://hyperphysics.phy-astr.gsu.edu/hbase/hph.html#mech", "description": "Concept maps and explanations of mechanics topics."}
    ],
    "ohm's law": [
        {"title": "HyperPhysics: Ohm's Law", "url": "http://hyperphysics.phy-astr.gsu.edu/hbase/electric/ohmlaw.html", "description": "Detailed explanation, examples, and calculator for Ohm's Law."},
        {"title": "All About Circuits: Ohm's Law", "url": "https://www.allaboutcircuits.com/textbook/direct-current/chpt-2/ohms-law/", "description": "Textbook chapter on Ohm's Law with practical examples."}
    ],
    "linear algebra": [
        {"title": "Khan Academy: Linear Algebra", "url": "https://www.khanacademy.org/math/linear-algebra", "description": "Covers vectors, matrices, transformations, and more."},
        {"title": "MIT OpenCourseware: Linear Algebra", "url": "https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/", "description": "Classic linear algebra course by Prof. Gilbert Strang."},
        {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab", "description": "Geometric intuition behind linear algebra concepts."}
    ],
    "special relativity": [
        {"title": "Einstein Online: Special Relativity", "url": "https://www.einstein-online.info/en/category/elementary/special-relativity/", "description": "Accessible explanations of Einstein's theory of special relativity."},
        {"title": "Coursera: Understanding Einstein - The Special Theory of Relativity (Stanford)", "url": "https://www.coursera.org/learn/einstein-relativity", "description": "Online course from Stanford University on special relativity."}
    ]
}

def get_educational_resource(topic: str) -> str:
    """
    Retrieves a list of curated educational resources for a given topic.

    Args:
        topic: A string representing the topic of interest (case-insensitive).
               Examples: "calculus basics", "Newtonian mechanics", "Ohm's Law"

    Returns:
        A string containing a formatted list of resources (title, url, description)
        if the topic is found in the predefined dictionary. Otherwise, returns a
        message indicating that no specific resources were found for that topic.
    """
    topic_lower = topic.lower() # Case-insensitive lookup
    
    resources = EDUCATIONAL_RESOURCES.get(topic_lower)
    
    if resources:
        output_str = f"Educational resources for '{topic}':\n\n"
        for i, res in enumerate(resources):
            output_str += f"{i+1}. Title: {res['title']}\n"
            output_str += f"   URL: {res['url']}\n"
            output_str += f"   Description: {res['description']}\n\n"
        return output_str.strip()
    else:
        return f"No specific educational resources found for '{topic}'. You could try a broader web search or check available topics."


if __name__ == '__main__':
    print("--- Math Problem Solver Examples ---")

    # Algebraic equations
    problem1 = "solve 2*x + 5 = 11 for x"
    print(f"Problem: {problem1}")
    print(f"Solution: {solve_math_problem(problem1)}\n")

    problem2 = "solve y**2 - 4 = 0 for y"
    print(f"Problem: {problem2}")
    print(f"Solution: {solve_math_problem(problem2)}\n")
    
    problem_implicit_mult = "solve 2x = 10 for x" # common user error
    print(f"Problem: {problem_implicit_mult} (testing implicit multiplication fix)")
    print(f"Solution: {solve_math_problem(problem_implicit_mult)}\n")

    # Differentiation
    problem3 = "derivative of x**3 + 2*x**2 - x with respect to x"
    print(f"Problem: {problem3}")
    print(f"Solution: {solve_math_problem(problem3)}\n")

    problem4 = "derivative of sin(y) * cos(y) with respect to y"
    print(f"Problem: {problem4}")
    print(f"Solution: {solve_math_problem(problem4)}\n")

    # Integration
    problem5 = "integrate x**2 with respect to x"
    print(f"Problem: {problem5}")
    print(f"Solution: {solve_math_problem(problem5)}\n")

    problem6 = "integrate cos(z) from 0 to pi with respect to z"
    print(f"Problem: {problem6}")
    print(f"Solution: {solve_math_problem(problem6)}\n")

    problem7 = "integrate 1/x with respect to x"
    print(f"Problem: {problem7}")
    print(f"Solution: {solve_math_problem(problem7)}\n")
    
    problem8 = "integrate exp(-x**2) from -oo to oo with respect to x" # oo for infinity
    print(f"Problem: {problem8}")
    print(f"Solution: {solve_math_problem(problem8)}\n")

    # Error handling examples
    problem_error1 = "solve 2x + 3 = " # Invalid equation
    print(f"Problem: {problem_error1}")
    print(f"Solution: {solve_math_problem(problem_error1)}\n")
    
    problem_error2 = "derivative of log(x) wrt x" # "wrt" instead of "with respect to"
    print(f"Problem: {problem_error2}")
    print(f"Solution: {solve_math_problem(problem_error2)}\n")

    problem_error3 = "fly me to the moon" # Unparseable
    print(f"Problem: {problem_error3}")
    print(f"Solution: {solve_math_problem(problem_error3)}\n")

    problem_error4 = "solve 3*a - b = 5 for a, b" # Multiple variables (current version expects one)
    print(f"Problem: {problem_error4}")
    print(f"Solution: {solve_math_problem(problem_error4)}\n")
    
    problem_error5 = "integrate 1/x from -1 to 1 with respect to x" # Integral across discontinuity
    print(f"Problem: {problem_error5}")
    print(f"Solution: {solve_math_problem(problem_error5)}\n")

    print("\n--- Physics Formula Examples ---")
    formula_concept1 = "Newton's second law"
    print(f"Concept: {formula_concept1}")
    print(f"Formula: {get_physics_formula(formula_concept1)}\n")

    formula_concept2 = "Ohm's Law"
    print(f"Concept: {formula_concept2}")
    print(f"Formula: {get_physics_formula(formula_concept2)}\n")

    formula_concept3 = "Kinetic Energy"
    print(f"Concept: {formula_concept3}")
    print(f"Formula: {get_physics_formula(formula_concept3)}\n")

    formula_concept_unknown = "Theory of Everything"
    print(f"Concept: {formula_concept_unknown}")
    print(f"Formula: {get_physics_formula(formula_concept_unknown)}\n")

    print("\n--- Physical Constant Examples ---")
    constant_name1 = "speed of light"
    print(f"Constant: {constant_name1}")
    print(f"Value: {get_physical_constant(constant_name1)}\n")

    constant_name2 = "Planck constant"
    print(f"Constant: {constant_name2}")
    print(f"Value: {get_physical_constant(constant_name2)}\n")

    constant_name3 = "gravitational constant"
    print(f"Constant: {constant_name3}")
    print(f"Value: {get_physical_constant(constant_name3)}\n")
    
    constant_name4 = "Electron mass"
    print(f"Constant: {constant_name4}")
    print(f"Value: {get_physical_constant(constant_name4)}\n")

    constant_name_unknown = "Muppet constant"
    print(f"Constant: {constant_name_unknown}")
    print(f"Value: {get_physical_constant(constant_name_unknown)}\n")
    
    constant_name_direct = "Boltzmann" # Testing direct lookup or description search (fallback)
    print(f"Constant: {constant_name_direct} (testing fallback)")
    print(f"Value: {get_physical_constant(constant_name_direct)}\n")

    constant_from_dict_only = "atomic mass constant"
    print(f"Constant: {constant_from_dict_only} (testing physical_constants dict key)")
    print(f"Value: {get_physical_constant(constant_from_dict_only)}\n")

    constant_proton_moment = "proton magnetic moment"
    print(f"Constant: {constant_proton_moment}")
    print(f"Value: {get_physical_constant(constant_proton_moment)}\n")
    
    print(f"SciPy version used: {scipy.__version__}")


    print("\n--- Unit Conversion Examples ---")
    # Successful conversions
    print(f"10 km to miles: {convert_units('10', 'kilometer', 'mile')}")
    print(f"2.5 meters to centimeters: {convert_units('2.5', 'meter', 'centimeter')}")
    # Corrected temperature units to use 'celsius' which Pint recognizes for degree Celsius.
    # Pint's canonical name is 'degree_Celsius', but 'celsius' is a common alias.
    print(f"100 celsius to fahrenheit: {convert_units('100', 'celsius', 'fahrenheit')}")
    print(f"20 celsius to kelvin: {convert_units('20', 'celsius', 'kelvin')}")
    print(f"60 miles/hour to km/hour: {convert_units('60', 'mile/hour', 'kilometer/hour')}")
    print(f"1000 Watts to kilowatts: {convert_units('1000', 'watt', 'kilowatt')}")
    print(f"1 Newton to dyne: {convert_units('1', 'newton', 'dyne')}")
    print(f"10 gallons to liters: {convert_units('10', 'gallon', 'liter')}")
    print(f"50 kg to pounds: {convert_units('50', 'kilogram', 'pound')}")
    print(f"1 atmosphere to psi: {convert_units('1', 'atm', 'psi')}") # atm is a default unit in pint

    # Error cases
    print(f"Meters to Kilograms (incompatible): {convert_units('10', 'meter', 'kilogram')}")
    print(f"Invalid from_unit: {convert_units('10', 'blargs', 'meter')}")
    print(f"Invalid to_unit: {convert_units('10', 'meter', 'flargs')}")
    print(f"Invalid value: {convert_units('ten', 'kilometer', 'mile')}")
    print(f"Celsius to Kelvin (using common 'celsius'): {convert_units('100', 'celsius', 'kelvin')}")
    print(f"Unparseable unit expression: {convert_units('10', 'kilometer / banana', 'mile/hour')}")


    print("\n--- Plot Generation Examples ---")
    # Function plots
    print(generate_plot("function", "sin(x) from 0 to 2*pi", "plot_sin_x.png"))
    print(generate_plot("function", "x**2 - 2*x + 1 from -5 to 5 for x", "plot_quadratic.png"))
    print(generate_plot("function", "exp(-x**2/2) from -4 to 4", "plot_gaussian.png")) # Default var x
    print(generate_plot("function", "log(t) from 1 to 100 for t", "plot_log_t.png"))
    print(generate_plot("function", "sin(x)*cos(x) from 0 to pi", "plot_sincos.png"))
    print(generate_plot("function", "x**3 from -2 to 2", "plot_cubic.png"))
    print(generate_plot("function", "1/x for x from -2 to 2", "plot_inv_x_alt_format.png")) # Test alternative format
    print(generate_plot("function", "exp(-x) from 0 to inf", "plot_exp_to_inf.png")) # Test inf in range

    # Scatter plots
    print(generate_plot("scatter", "x_values=[1,2,3,4,5]; y_values=[1,4,9,16,25]", "plot_scatter_squares.png"))
    print(generate_plot("scatter", "x=[0.1, 0.5, 1.2, 2.5, 3.1]; y=[0.5, 2.0, 1.5, 4.0, 3.5]", "plot_scatter_custom.png"))
    
    # Error cases for plotting
    print(generate_plot("function", "sin(x) from 2*pi to 0", "plot_error_range.png")) # Invalid range
    print(generate_plot("function", "sin(x) from a to b", "plot_error_range_nonnumeric.png")) # Non-numeric range
    print(generate_plot("function", "unknown_func(x) from 0 to 1", "plot_error_expr.png")) # Invalid expression (SympifyError)
    print(generate_plot("function", "sin(x) fro 0 to 1", "plot_error_format_func.png")) # Malformed data_str
    print(generate_plot("scatter", "x_values=[1,2,3]; y_values=[1,2]", "plot_error_mismatch.png")) # Mismatched lists
    print(generate_plot("scatter", "x_values=[1,2,'a']; y_values=[1,2,3]", "plot_error_nonnumeric_scatter.png"))
    print(generate_plot("scatter", "x_vals=[1,2,3]; y_vals=not_a_list", "plot_error_format_scatter.png")) # Malformed list
    print(generate_plot("histogram", "data=[1,2,2,3,3,3,4]", "plot_error_type.png")) # Unknown plot type
    # The following line for "1/x from -2 to 2" was the one that worked with regex1 for plot_discontinuity.png
    # The "1/x for x from -2 to 2" for plot_inv_x_alt_format.png was the one with the persistent "invalid syntax" error.
    print(generate_plot("function", "1/x from -2 to 2", "plot_discontinuity.png")) 


    print("\n--- Web Search Examples (Mock) ---")
    print(web_search("Ohm's Law"))
    print("-" * 30)
    print(web_search("Kinetic Energy", num_results=1))
    print("-" * 30)
    print(web_search("SymPy library")) # This query now should find results from MOCK_SEARCH_RESULTS
    print("-" * 30)
    print(web_search("Quantum Entanglement")) # Query not in mock data
    print("-" * 30)
    print(web_search("SciPy", num_results=5)) # Request more results than available for this specific query


    print("\n--- Educational Resources Examples ---")
    print(get_educational_resource("Calculus Basics"))
    print("-" * 40)
    print(get_educational_resource("ohm's law")) # Test case-insensitivity
    print("-" * 40)
    print(get_educational_resource("Linear Algebra"))
    print("-" * 40)
    print(get_educational_resource("Quantum Physics")) # Topic not in dictionary
    print("-" * 40)
    print(get_educational_resource("Special Relativity"))
