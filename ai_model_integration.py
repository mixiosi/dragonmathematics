"""
This file defines the declarations for tools available in math_tools.py,
formatted for use with Google's Gemini API for function calling.

Each tool declaration follows a structure similar to what the Gemini API
expects, typically a JSON-like object (represented here as Python dictionaries).
This allows the Gemini model to understand:
- The tool's name.
- A description of what the tool does.
- The parameters the tool accepts, including their types, descriptions,
  and whether they are required.

This information enables the model to decide when and how to call these
tools in response to user prompts.
"""

# List to hold all tool declarations for the Gemini API
tool_declarations = [
    {
        "name": "solve_math_problem",
        "description": "Solves mathematical problems including algebraic equations, differentiation, and integration using the SymPy library. For example, 'solve 2*x + 3 = 7 for x', 'derivative of x**2 with respect to x', or 'integrate sin(x) with respect to x'.",
        "parameters": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "A string describing the mathematical problem. Examples: 'solve 2*x + 3 = 7 for x', 'derivative of x**2 with respect to x', 'integrate sin(x) from 0 to pi with respect to x'."
                }
            },
            "required": ["problem"]
        }
    },
    {
        "name": "get_physics_formula",
        "description": "Retrieves the mathematical formula for a given physics concept (e.g., Newton's second law, Ohm's law, kinetic energy).",
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "The physics concept for which to find the formula (e.g., 'Ohm\\'s law', 'kinetic energy', 'einstein\\'s mass-energy equivalence')."
                }
            },
            "required": ["concept"]
        }
    },
    {
        "name": "get_physical_constant",
        "description": "Retrieves the value and unit of a physical constant (e.g., speed of light, Planck constant, electron mass).",
        "parameters": {
            "type": "object",
            "properties": {
                "constant_name": {
                    "type": "string",
                    "description": "The common name of the physical constant (e.g., 'speed of light', 'Planck constant', 'Avogadro number')."
                }
            },
            "required": ["constant_name"]
        }
    },
    {
        "name": "convert_units",
        "description": "Converts a value from one unit to another (e.g., kilometers to miles, Celsius to Fahrenheit). Uses the Pint library.",
        "parameters": {
            "type": "object",
            "properties": {
                "value_str": {
                    "type": "string",
                    "description": "The numerical value to convert, as a string (e.g., '10', '2.5')."
                },
                "from_unit": {
                    "type": "string",
                    "description": "The unit to convert from (e.g., 'kilometer', 'meter', 'degreeCelsius', 'mile/hour'). Refer to Pint library unit names."
                },
                "to_unit": {
                    "type": "string",
                    "description": "The unit to convert to (e.g., 'mile', 'centimeter', 'degreeFahrenheit', 'kilometer/hour'). Refer to Pint library unit names."
                }
            },
            "required": ["value_str", "from_unit", "to_unit"]
        }
    },
    {
        "name": "generate_plot",
        "description": "Generates a plot using Matplotlib and saves it to a file. Supports function plots and scatter plots.",
        "parameters": {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "description": "Type of plot to generate. Must be 'function' or 'scatter'."
                },
                "data_str": {
                    "type": "string",
                    "description": "String containing data for the plot. For 'function': 'expression from start_val to end_val [for var]' (e.g., 'sin(x) from 0 to 2*pi', 'x**2 for x from -5 to 5'). For 'scatter': 'x_values=[1,2,3]; y_values=[2,4,1]'."
                },
                "filename": {
                    "type": "string",
                    "description": "Optional. Name of the file to save the plot (e.g., 'plot.png'). Defaults to 'plot.png'."
                }
            },
            "required": ["plot_type", "data_str"]
        }
    },
    {
        "name": "web_search",
        "description": "Simulates a web search using a predefined list of mock results to find information on a given query. In a real scenario, this would query a search engine.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search term or question to look up."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Optional. The maximum number of search results to return. Defaults to 3."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_educational_resource",
        "description": "Retrieves a list of curated educational resources (articles, videos, courses) for a given scientific or mathematical topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic for which to find educational resources (e.g., 'calculus basics', 'Newtonian mechanics', 'Ohm\\'s Law')."
                }
            },
            "required": ["topic"]
        }
    }
]

# --- Example of how to use these declarations with the Gemini API ---
#
# import google.generativeai as genai
#
# # Assuming 'model' is an initialized GenerativeModel instance
# # e.g., model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', tools=tool_declarations)
#
# # response = model.generate_content("Could you plot sin(x) from 0 to 2*pi and save it as my_plot.png?")
# #
# # # If the model decides to call a function, the response will contain a FunctionCall part:
# # if response.parts[0].function_call.name == "generate_plot":
# #     args = response.parts[0].function_call.args
# #     plot_type = args.get("plot_type")
# #     data_str = args.get("data_str")
# #     filename = args.get("filename", "plot.png") # Handle optional arg
# #
# #     # ... (call your actual math_tools.generate_plot function with these args)
# #     # result = math_tools.generate_plot(plot_type, data_str, filename)
# #
# #     # Send the result back to the model:
# #     # response = model.generate_content(
# #     #     glm.Content(
# #     #         parts=[glm.Part(
# #     #             function_response=glm.FunctionResponse(name="generate_plot", response={"result": result})
# #     #         )]
# #     #     )
# #     # )
# #     # final_answer = response.text
#
# # For demonstration purposes, print the declarations:
# if __name__ == "__main__":
#     import json
#     print("Tool Declarations for Gemini API:\n")
#     print(json.dumps(tool_declarations, indent=2))
#
#     print("\n\nExample of how to pass to GenerativeModel (conceptual):")
#     print("# from google.generativeai.types import HarmCategory, HarmBlockThreshold")
#     print("# model = genai.GenerativeModel(")
#     print("#     model_name='gemini-1.5-flash-latest',")
#     print("#     tools=tool_declarations,")
#     print("#     safety_settings={")
#     print("# HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,")
#     print("# HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,")
#     print("# HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,")
#     print("# HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,")
#     print("#     }")
#     print("# )")
#
#     # Test one declaration (e.g. generate_plot)
#     generate_plot_decl = next((item for item in tool_declarations if item["name"] == "generate_plot"), None)
#     if generate_plot_decl:
#         print("\n\nExample: 'generate_plot' declaration structure:")
#         print(json.dumps(generate_plot_decl, indent=2))

"""
Note on parameter types for Gemini API:
- "string": For text inputs.
- "integer": For whole numbers.
- "number": For floating-point numbers (can also represent integers).
- "boolean": For true/false values.
- "object": For structured parameters (like the 'parameters' field itself).
- "array": For lists of items (e.g., a list of strings or numbers).
The 'type' under 'properties' for each parameter should be one of these primitive types.
"""
