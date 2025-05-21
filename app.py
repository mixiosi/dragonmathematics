"""
Main Flask application file for the Math and Physics AI Tutor backend.
"""
import os
from flask import Flask, request, jsonify

# Attempt to import tool declarations and math tools
# These will be used in later subtasks for actual AI model integration.
try:
    from ai_model_integration import tool_declarations
    import math_tools
except ImportError as e:
    # Handle cases where these files might not exist yet in early development stages
    # or in a minimal deployment. For this task, we assume they exist.
    print(f"Warning: Could not import ai_model_integration or math_tools: {e}")
    tool_declarations = [] # Default to empty list if import fails
    math_tools = None      # Default to None

app = Flask(__name__)

# --- Secret Key Configuration for Flask Session ---
# IMPORTANT: This key must be kept secret in production!
# Load from environment variable for security.
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

if not app.secret_key:
    print("ERROR: FLASK_SECRET_KEY environment variable not set. Session management will not be secure and may not work as expected.")
    # In a production environment, you might want to raise an error or exit if the secret key is not set.
    # For development, we can proceed with a warning, but sessions will be insecure.
    # app.secret_key = "dev_unsafe_secret_key" # Unsafe fallback for dev only if you absolutely must run without env var

# --- API Key Configuration ---
# IMPORTANT SECURITY NOTE:
# NEVER hardcode your API key directly in the source code for production.
# This key should be loaded from a secure source, such as:
# 1. Environment Variables (Recommended for most deployments):
#    API_KEY = os.getenv("GOOGLE_AI_API_KEY")
# 2. Configuration Files (e.g., .env files, JSON/YAML configs):
#    Load the key using a library like python-dotenv or a config parser.
# 3. Secret Management Services (e.g., Google Secret Manager, HashiCorp Vault):
#    For more robust security in cloud environments.
#
# For now, a placeholder is used. Replace this with a secure loading mechanism.
# ** WARNING: This is a placeholder and NOT for production. **
# API_KEY = "YOUR_GOOGLE_AI_API_KEY_REPLACE_ME" # Original placeholder
API_KEY = os.environ.get("GOOGLE_AI_API_KEY")

# --- Initialize Google Gemini Model ---
import google.generativeai as genai
model = None
initialization_error_message = None

if not API_KEY:
    error_msg = "ERROR: GOOGLE_AI_API_KEY environment variable not set. The AI tutor functionality will be disabled."
    print(error_msg)
    initialization_error_message = error_msg
else:
    try:
        genai.configure(api_key=API_KEY)
        # Target model could be "learnlm-2.0-flash-experimental" if available and access is granted.
        # Using "gemini-1.5-flash-latest" as a robust, generally available alternative.
        # Safety settings can be adjusted if needed, e.g., to allow more borderline content.
        # from google.generativeai.types import HarmCategory, HarmBlockThreshold
        # safety_settings = {
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        # }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", 
            tools=tool_declarations,
            # safety_settings=safety_settings # Uncomment to apply custom safety settings
        )
        print("Google Gemini model initialized successfully.")
    except Exception as e:
        error_msg = f"ERROR: Failed to initialize Google Gemini model: {e}"
        print(error_msg)
        initialization_error_message = error_msg
        model = None # Ensure model is None if initialization fails

# Mapping tool names to actual Python functions from math_tools
available_tools = {}
if math_tools:
    available_tools = {
        "solve_math_problem": math_tools.solve_math_problem,
        "get_physics_formula": math_tools.get_physics_formula,
        "get_physical_constant": math_tools.get_physical_constant,
        "convert_units": math_tools.convert_units,
        "generate_plot": math_tools.generate_plot,
        "web_search": math_tools.web_search,
        "get_educational_resource": math_tools.get_educational_resource,
    }
else:
    print("WARNING: math_tools module not loaded. Tool execution will not be possible.")
    # Define dummy functions if math_tools is not available, so the app can still run
    # This helps in scenarios where math_tools might be missing temporarily during dev
    def create_dummy_tool(name):
        def dummy_func(*args, **kwargs):
            return f"Error: Tool '{name}' is not available because math_tools module failed to load."
        return dummy_func
    
    for tool_decl in tool_declarations:
        if tool_decl["name"] not in available_tools:
             available_tools[tool_decl["name"]] = create_dummy_tool(tool_decl["name"])


# --- Endpoints ---

@app.route('/')
def home():
    """
    Serves the main HTML page for the frontend.
    """
    # Ensure the path is relative to the 'static' directory where index.html is located
    return app.send_static_file('index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Endpoint to ask a question to the AI tutor.
    Accepts POST requests with a JSON body containing a "question" field.
    Currently returns a mock response.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415 # Unsupported Media Type

    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Missing 'question' field in JSON body"}), 400

    if not model:
        if initialization_error_message:
            return jsonify({"error": f"AI Model not available: {initialization_error_message}"}), 503 # Service Unavailable
        return jsonify({"error": "AI Model not initialized. Check server logs."}), 503

    try:
        # Retrieve chat history from session or start a new one
        chat_history_from_session = session.get('chat_history', [])
        
        # Comment regarding history size management for production
        # For production, consider limiting history size to prevent oversized session cookies
        # and to manage token usage with the AI model.
        # Example: max_history_turns = 20; chat_history_from_session = chat_history_from_session[-max_history_turns*2:] # Assuming each turn has user + model part

        # Start chat using existing history if available.
        # Automatic function calling is enabled here to let the model decide when to call tools.
        # The subsequent logic will handle executing those calls.
        chat = model.start_chat(history=chat_history_from_session, enable_automatic_function_calling=True)
        
        print(f"INFO: Sending message to chat. Current history length: {len(chat.history)}")
        response = chat.send_message(question)

        answer = response.text
        
        # Log any function call requests from the model for debugging and next subtask prep
        # Gemini API typically puts function calls in response.candidates[0].content.parts
        # For version 1.5, it might be directly in response.function_calls if available
        # or response.candidates[0].function_calls
        
        # Check for function calls in the response parts
        # (Iterate, as there could be multiple parts, though usually one for a function call)
        function_calls_detected = []
        for part in response.parts:
            if part.function_call:
                function_calls_detected.append({
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                })
        
        if function_calls_detected:
            print(f"INFO: Model suggested function call(s): {function_calls_detected}")
            # For this subtask, we are not executing the function call yet.
            # We are just returning the text part, if any.
            # If the model *only* returns a function call, response.text might be empty or minimal.
        
        debug_info = {"initial_response_text": answer, "detected_initial_function_calls": function_calls_detected if function_calls_detected else None}

        # --- Function Call Execution Loop (handles one call for this subtask) ---
        # Process the first detected function call, if any.
        primary_function_call_details = None
        if function_calls_detected:
            primary_function_call_details = function_calls_detected[0] # Process the first one

        if primary_function_call_details:
            fc_name = primary_function_call_details["name"]
            fc_args = primary_function_call_details["args"]

            print(f"INFO: Model wants to call function '{fc_name}' with args: {fc_args}")
            
            executed_call_info = {"name": fc_name, "args": fc_args}
            debug_info["executed_function_call"] = executed_call_info

            tool_result_str = ""
            if fc_name in available_tools:
                python_function_to_call = available_tools[fc_name]
                try:
                    print(f"Attempting to execute local tool: {fc_name}")
                    # Execute the local tool using keyword arguments
                    tool_result = python_function_to_call(**fc_args)
                    tool_result_str = str(tool_result) # Ensure it's a string
                    executed_call_info["result"] = tool_result_str
                    print(f"Tool '{fc_name}' executed. Result (first 200 chars): {tool_result_str[:200]}...")
                except TypeError as te: # Catch errors if args don't match function signature
                    print(f"ERROR: TypeError during execution of tool '{fc_name}' with args {fc_args}: {te}")
                    tool_result_str = f"Error: Type error calling {fc_name}. Arguments provided by AI might be incorrect. Details: {str(te)}"
                    executed_call_info["error"] = tool_result_str
                except Exception as e:
                    print(f"ERROR: Exception during execution of tool '{fc_name}': {e}")
                    tool_result_str = f"Error executing tool {fc_name}: {str(e)}"
                    executed_call_info["error"] = tool_result_str
            else:
                print(f"ERROR: Tool '{fc_name}' requested by model is not available locally.")
                tool_result_str = f"Error: Function {fc_name} is not recognized or available."
                executed_call_info["error"] = tool_result_str
            
            # Send the tool's output back to the model
            print(f"INFO: Sending tool response for '{fc_name}' back to model.")
            # Construct the FunctionResponse correctly for the API
            # The API expects the 'response' field to be a dict, typically {"content": "your_tool_output_string"}
            function_response_part = genai.Part(
                function_response=genai.types.FunctionResponse(
                    name=fc_name,
                    response={"content": tool_result_str} 
                )
            )
            
            # Send the function response back to the chat
            response = chat.send_message(function_response_part)
            
            # The new response should contain the final text answer from the model
            answer = response.text # Update answer with the model's response after processing the tool output
            debug_info["response_after_tool_execution"] = answer

            # Check for any new function calls after sending tool response (for debug and future multi-turn)
            follow_up_calls_detected = []
            for part in response.parts: # Check new response parts
                if part.function_call:
                    follow_up_calls_detected.append({
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args)
                    })
            if follow_up_calls_detected:
                print(f"INFO: Model suggested further function call(s) after tool use: {follow_up_calls_detected}")
                debug_info["follow_up_suggestions"] = follow_up_calls_detected
                if not answer.strip(): # If no text answer, but another function call
                     answer = f"[Model suggested a follow-up tool call: {follow_up_calls_detected[0]['name']}. Further tool execution would be needed.]"
        
        # If there was no initial function call, 'answer' is already response.text from the first call.
        # If there was a function call, 'answer' is now the text after processing the tool response.
        
        # Update session with the new chat history
        # Ensure chat.history is serializable. Gemini's history objects typically are.
        session['chat_history'] = chat.history 
        print(f"INFO: Updated chat history in session. New length: {len(session['chat_history'])}")

        return jsonify({"answer": answer, "debug_info": debug_info})

    except genai.types.generation_types.BlockedPromptException as e:
        print(f"ERROR: Prompt blocked by API: {e}")
        return jsonify({"error": "Your question was blocked by the content safety filter.", "details": str(e)}), 400
    except genai.types.generation_types.StopCandidateException as e:
        print(f"ERROR: Generation stopped unexpectedly: {e}") # Often due to safety settings on response
        return jsonify({"error": "The AI response was stopped before completion, possibly due to content policies.", "details": str(e)}), 500
    except Exception as e:
        # Catching a broad exception for other potential API errors or issues
        print(f"ERROR: An unexpected error occurred while interacting with the AI model: {e}")
        # Log the full traceback for server-side debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred with the AI model.", "details": str(e)}), 500

@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    """
    Clears the current conversation history from the session,
    allowing the user to start a fresh conversation.
    """
    session.pop('chat_history', None)
    return jsonify({"message": "New chat session started. Conversation history cleared."}), 200


if __name__ == '__main__':
    # Note: For development, 'flask run' is preferred.
    # This direct app.run() is for environments where 'flask run' isn't as convenient.
    # Ensure the FLASK_ENV environment variable is set to 'development' for debug mode.
    # app.run(debug=True) # Example: app.run(host='0.0.0.0', port=5000, debug=True)
    print("Flask app initialized.")
    if not API_KEY:
        print("WARNING: GOOGLE_AI_API_KEY is not set. AI features will be disabled.")
    else:
        print("GOOGLE_AI_API_KEY is set.")
    
    if model:
        print(f"Gemini model ('{model.model_name}') initialized and ready.")
    else:
        print("Gemini model failed to initialize or API key is missing.")
    
    print("Use 'flask run' (after setting FLASK_APP=app.py and optionally FLASK_ENV=development) to start the development server.")
    # print(f"Example of tool_declarations (first one): {tool_declarations[0] if tool_declarations else 'Not loaded'}")
    # print(f"Math tools module loaded: {'Yes' if math_tools else 'No'}")
