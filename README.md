# Math and Physics AI Tutor Backend

This is a Flask-based backend for the Math and Physics AI Tutor application. It provides an API endpoint to interact with an AI model (currently mocked, to be integrated with Google's Gemini API) and various mathematical and scientific tools.

## Project Structure

```
.
├── app.py                  # Main Flask application
├── math_tools.py           # Core mathematical and scientific helper functions
├── ai_model_integration.py # Tool declarations for AI model function calling
├── requirements.txt        # Python package dependencies
├── plot.png                # Default filename for generated plots (if any)
└── README.md               # This file
```

## Setup

1.  **Clone the repository (if applicable).**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables (Future Use for API Key):**

    For actual AI model integration (not used by the current mock `/api/ask` endpoint), you will need to set your Google AI API key. **Never hardcode your API key in the source code.**

    The application **requires** the `GOOGLE_AI_API_KEY` environment variable to be set to communicate with the Google Gemini API. If this key is not set or is invalid, the `/api/ask` endpoint will return an error.

    The application uses Flask sessions to maintain conversation history. For session security, you **must** set a `FLASK_SECRET_KEY` environment variable.

    **Generate a Secret Key:**
    You can generate a strong secret key using Python:
    ```bash
    python -c 'import os; print(os.urandom(24).hex())'
    ```
    This will output a random string.

    **Set Environment Variables:**
    1.  `GOOGLE_AI_API_KEY`: Your API key for Google's Gemini API.
    2.  `FLASK_SECRET_KEY`: The secret key you generated above.

    **Linux/macOS:**
    ```bash
    export GOOGLE_AI_API_KEY="your_actual_api_key_here"
    ```
    To make it persistent across sessions, add this line to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`).

    **Windows (Command Prompt):**
    ```bash
    set GOOGLE_AI_API_KEY=your_actual_api_key_here
    ```
    **Windows (PowerShell):**
    ```bash
    $env:GOOGLE_AI_API_KEY="your_actual_api_key_here"
    ```
    For persistent storage on Windows, you can set it via System Properties -> Environment Variables.

## Running the Application

The application consists of a Flask backend and a static HTML/CSS/JavaScript frontend.

**1. Running the Flask Backend Server:**

*   Ensure all dependencies are installed (see "Setup" above).
*   Make sure the `GOOGLE_AI_API_KEY` and `FLASK_SECRET_KEY` environment variables are set.
*   **Set the `FLASK_APP` environment variable:**
    This tells Flask where your application is.
    ```bash
    export FLASK_APP=app.py  # On Windows use `set FLASK_APP=app.py`
    ```
*   **Set `FLASK_ENV` (optional, for development):**
    This enables debug mode, which provides more detailed error messages and auto-reloads the server on code changes.
    ```bash
    export FLASK_ENV=development
    ```
*   **Run the Flask development server:**
    ```bash
    flask run
    ```
    The backend server will typically start on `http://127.0.0.1:5000/`. Keep this terminal window open.

**2. Accessing the Frontend:**

*   Once the backend Flask server is running, it will also serve the frontend.
*   Open your web browser and navigate to the root URL of the Flask server:
    ```
    http://127.0.0.1:5000/
    ```
    This will load `index.html` and the associated CSS and JavaScript files. You can then interact with the AI Tutor through this web interface.

## API Endpoints

*   **`GET /`**:
    *   Returns a simple HTML message confirming the server is running.
    *   Example: `curl http://127.0.0.1:5000/`

*   **`POST /api/ask`**:
    *   Accepts a JSON request body with a "question" field.
        *   Interacts with the AI model, maintaining conversation history using Flask sessions.
        *   Supports tool/function calling with `math_tools.py`.
    *   **Request Body Example:**
        ```json
        {
          "question": "What is Ohm's law?"
        }
        ```
        *   **Example using `curl` (ensure you handle session cookies if testing via curl, e.g., with `-b cookie.txt -c cookie.txt`):**
        ```bash
        curl -X POST -H "Content-Type: application/json" -d '{"question":"What is Ohm law?"}' http://127.0.0.1:5000/api/ask
        ```
    *   **Expected Response Example (actual AI response will vary):**
        ```json
        {
          "answer": "Ohm's Law states that the current through a conductor between two points is directly proportional to the voltage across the two points, and inversely proportional to the resistance between them. It's commonly expressed as V = IR.",
              "debug_info": { /* ... details about model interaction and tool use ... */ }
        }
        ```

*   **`POST /api/new_chat`**:
    *   Clears the current conversation history from the session.
    *   Call this endpoint to start a fresh conversation with the AI.
    *   **Example using `curl` (ensure you handle session cookies):**
        ```bash
        curl -X POST http://127.0.0.1:5000/api/new_chat
        ```
    *   **Expected Response:**
        ```json
        {
          "message": "New chat session started. Conversation history cleared."
        }
        ```

## Next Steps (Future Subtasks)

*   Refine error handling for API calls and tool execution.
*   Add more comprehensive logging.
*   Consider implementing limits on conversation history size for production.
```
