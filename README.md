Poegroqi is a Poe server bot built with FastAPI that uses Groq AI as the model backend.  
This project lets you run a custom Poe bot with a carefully designed system prompt for concise, helpful, and language‑adaptive answers.  

## Features

- Integration with Poe using the `fastapi_poe` library. 
- Uses the official Groq Python client (`groq`) as the inference backend (default model: `qwen/qwen3-32b`).  
- Structured system prompt that:
  - Encourages short, direct answers.
  - Uses headings and lists when appropriate.
  - Automatically follows the user’s language.
  - Avoids defaulting to Chinese responses.  
- Streams tokens from Groq to Poe for responsive chat.  
- Simple logging to `stderr` for debugging (message count, roles, and payload size).  

## Architecture

The `main.py` file contains the core logic of the bot:  

- `SYSTEM_PROMPT` constant with the main instructions for the model.  
- **GroqBot** class extending `fp.PoeBot`:
  - Initializes the Groq client with `GROQ_API_KEY` from environment variables.  
  - Implements `get_response`, which:
    - Builds the `messages` list, inserting the system prompt as the first message.
    - Converts Poe’s `bot` role to the `assistant` role expected by Groq.  
    - Calls `client.chat.completions.create` with streaming enabled.
    - Yields each token delta as `fp.PartialResponse` back to Poe.  
- FastAPI app `app` created via `fp.make_app(GroqBot(), access_key=...)`.  

## Requirements

Make sure you have:  

- Python 3.9 or newer  
- A Groq account and valid API key (`GROQ_API_KEY`)  
- A Poe account with server bot access and `POE_ACCESS_KEY`  

### Environment variables

Set the following environment variables before running the server:  

```bash
export GROQ_API_KEY="your_groq_api_key_here"
export POE_ACCESS_KEY="your_poe_access_key_here"
```

On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:POE_ACCESS_KEY="your_poe_access_key_here"
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ferritopia/Poegroqi.git
   cd Poegroqi
   ```

2. (Optional but recommended) Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows PowerShell
   ```

3. Install dependencies (adjust to match your `requirements.txt` if present):

   ```bash
   pip install fastapi-poe groq uvicorn
   ```

## Running the server

Start the FastAPI server with Uvicorn:  

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will expose the endpoint Poe needs to connect your bot to Groq.  

## Configuring the bot on Poe

At a high level, you need to: 

- Open the **Server Bot** settings page on Poe.   
- Set the server URL to your deployed endpoint, for example:  
  `https://your-domain.com/poe` or `https://your-ngrok-url.io/poe` (depending on how you mount the app).   
- Ensure that `POE_ACCESS_KEY` on your server matches the access key configured in Poe.  
- Save and test the bot from within Poe’s interface. 

The exact endpoint path can be adjusted based on your `fastapi_poe` setup and your reverse proxy/hosting configuration. 

## Customization

You can easily customize several parts of the bot:

- **Groq model**

  Change the `model` parameter in `chat.completions.create`:

  ```python
  stream = self.client.chat.completions.create(
      model="qwen/qwen3-32b",
      ...
  )
  ```

- **System prompt**

  Edit the `SYSTEM_PROMPT` constant at the top of `main.py` to change the bot’s tone, default language rules, or constraints.  

- **Generation parameters**

  Tune `temperature`, `max_completion_tokens`, `top_p`, and `reasoning_effort` to fit your use case.  

## Debugging

The code includes basic logging to `stderr`:  

- Total message count (`Messages count`)  
- Roles for each message (`Roles`)  
- JSON payload size in bytes (`Payload size`)  

If you see unexpected behavior or errors, check the server logs (the terminal where you run Uvicorn) for details.  

