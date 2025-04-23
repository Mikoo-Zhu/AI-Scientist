import json
import os
import re

import anthropic
import backoff
import openai
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    # Anthropic models
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # OpenRouter models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    # Google Gemini models
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
]

# Map standard names to OpenRouter IDs
# Note: This mapping might need adjustments based on exact OpenRouter naming conventions.
OPENROUTER_MODEL_MAP = {
    # Anthropic
    "claude-3-5-sonnet-20240620": "anthropic/claude-3-5-sonnet",
    "claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet", # OpenRouter might use the base name
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0": "anthropic/claude-3-sonnet",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0": "anthropic/claude-3-5-sonnet",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic/claude-3-5-sonnet", # Base name likely
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0": "anthropic/claude-3-haiku",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0": "anthropic/claude-3-opus",
    "vertex_ai/claude-3-opus@20240229": "anthropic/claude-3-opus",
    "vertex_ai/claude-3-5-sonnet@20240620": "anthropic/claude-3-5-sonnet",
    "vertex_ai/claude-3-5-sonnet-v2@20241022": "anthropic/claude-3-5-sonnet", # Base name likely
    "vertex_ai/claude-3-sonnet@20240229": "anthropic/claude-3-sonnet",
    "vertex_ai/claude-3-haiku@20240307": "anthropic/claude-3-haiku",
    # OpenAI
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o-mini-2024-07-18": "openai/gpt-4o-mini-2024-07-18",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-2024-05-13": "openai/gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-2025-04-14": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14": "openai/gpt-4.1-mini",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14": "openai/gpt-4.1-nano",
    "o1": "openai/o1",
    "o1-2024-12-17": "openai/o1",
    "o1-preview-2024-09-12": "openai/o1-preview-2024-09-12",
    "o1-mini": "openai/o1-mini",
    "o1-mini-2024-09-12": "openai/o1-mini-2024-09-12",
    "o3-mini": "openai/o3-mini",
    "o3-mini-2025-01-31": "openai/o3-mini",
    # Llama via OpenRouter
    "llama3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    # DeepSeek
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-coder": "deepseek/deepseek-coder",
    "deepseek-reasoner": "deepseek/deepseek-reasoner",
    # Google Gemini
    "gemini-1.5-flash": "google/gemini-flash-1.5",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "gemini-2.0-flash": "google/gemini-flash-2.0", # Hypothetical
    "gemini-2.0-flash-lite": "google/gemini-flash-lite-2.0", # Hypothetical
    "gemini-2.0-flash-thinking-exp-01-21": "google/gemini-flash-thinking-exp-2.0", # Hypothetical
    "gemini-2.5-pro-preview-03-25": "google/gemini-pro-2.5-preview", # Hypothetical
    "gemini-2.5-pro-exp-03-25": "google/gemini-pro-2.5-exp", # Hypothetical
}

# Get N responses from a single message, used for ensembling.
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
        msg,
        client,
        model, # Original model name requested
        client_model, # Model ID to use with the client (could be OpenRouter ID)
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    if msg_history is None:
        msg_history = []

    # Check the type of client to determine API call style
    # Since OpenRouter provides an OpenAI-compatible API, we use the OpenAI client methods
    if isinstance(client, openai.OpenAI):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        # Use client_model which might be the OpenRouter ID
        response = client.chat.completions.create(
            model=client_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            # Add seed if supported by the specific model via OpenRouter
            # seed=0, # May cause errors depending on the underlying model
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    # Handle direct Anthropic calls (only if OpenRouter key wasn't set)
    elif isinstance(client, (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex)):
        content, new_msg_history_list = [], []
        for _ in range(n_responses):
             # Direct Anthropic doesn't support 'n' parameter directly, so loop
            c, hist = get_response_from_llm(
                msg,
                client,
                model, # Pass original model name
                client_model, # Pass model ID for this client type
                system_message,
                print_debug=False,
                msg_history=msg_history, # Use original history for each call
                temperature=temperature,
            )
            content.append(c)
            new_msg_history_list.append(hist)
        # For batch, we need a consistent structure, maybe just return the list of histories?
        # The original code returned a list of histories, let's stick to that.
        new_msg_history = new_msg_history_list
    else:
        # Fallback or error for unsupported client types
        raise TypeError(f"Unsupported client type for batch responses: {type(client)}")


    if print_debug:
        print()
        print("*" * 20 + " LLM BATCH START " + "*" * 20)
        # Print history from the first response for debugging
        if new_msg_history:
            for j, msg_item in enumerate(new_msg_history[0]):
                 print(f'{j}, {msg_item["role"]}: {msg_item["content"]}')
        print(content)
        print("*" * 21 + " LLM BATCH END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
        msg,
        client,
        model, # Original model name requested
        client_model, # Model ID to use with the client (could be OpenRouter ID)
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    # Check the type of client to determine API call style
    # Since OpenRouter provides an OpenAI-compatible API, we use the OpenAI client methods
    if isinstance(client, openai.OpenAI):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        # Use client_model which might be the OpenRouter ID
        response = client.chat.completions.create(
            model=client_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            # Add seed if supported by the specific model via OpenRouter
            # seed=0, # May cause errors depending on the underlying model
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    # Handle direct Anthropic calls (only if OpenRouter key wasn't set)
    elif isinstance(client, (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex)):
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=client_model, # Use the model ID appropriate for this client
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    else:
         # Fallback or error for unsupported client types
        raise TypeError(f"Unsupported client type for single response: {type(client)}")


    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg_item in enumerate(new_msg_history):
            print(f'{j}, {msg_item["role"]}: {msg_item["content"]}')
        print("Response Content:", content) # Added label for clarity
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        # Be careful with greedy matching here, make it non-greedy
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    valid_json = None
    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            valid_json = parsed_json # Store the first valid JSON found
            break
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues like trailing commas (simple fix)
            json_string_fixed = re.sub(r",\s*([\}\]])", r"\1", json_string)
            try:
                 parsed_json = json.loads(json_string_fixed)
                 valid_json = parsed_json
                 break
            except json.JSONDecodeError:
                 # Attempt to fix invalid control characters
                 try:
                    json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                    parsed_json = json.loads(json_string_clean)
                    valid_json = parsed_json
                    break
                 except json.JSONDecodeError:
                    continue # Try next match if cleaning didn't help

    # If no JSON found between markers or as fallback, try parsing the whole string
    if valid_json is None:
        try:
            llm_output_clean = re.sub(r"[\x00-\x1F\x7F]", "", llm_output)
            valid_json = json.loads(llm_output_clean.strip())
        except json.JSONDecodeError:
            pass # Cannot parse the whole string either

    return valid_json # Return the first valid JSON found or None


def create_client(model):
    """
    Creates an LLM client instance and returns the client and the model ID string
    to be used for API calls. Prioritizes OpenRouter if OPENROUTER_API_KEY is set.
    """
    # Prioritize OpenRouter if API key is available
    if "OPENROUTER_API_KEY" in os.environ:
        openrouter_model_id = OPENROUTER_MODEL_MAP.get(model)
        if openrouter_model_id:
            print(f"Using OpenRouter API (OpenAI compatible) for model {model} mapped to {openrouter_model_id}.")
            # Always return an OpenAI client when using OpenRouter
            return openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1"
            ), openrouter_model_id # Return the mapped OpenRouter ID
        else:
            # If model is not in the map, maybe it's already an OpenRouter ID or we should warn/error
            print(f"Warning: Model '{model}' not found in OPENROUTER_MODEL_MAP. Assuming it's a valid OpenRouter ID or direct model name.")
            # Try using the provided name directly with OpenRouter
            return openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1"
            ), model # Return the original model name

    # --- Fallback to original logic if OPENROUTER_API_KEY is not set ---
    elif model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        if "ANTHROPIC_API_KEY" not in os.environ: raise ValueError("ANTHROPIC_API_KEY missing for direct Anthropic call.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        # Add credential checks if needed
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
         # Add credential checks if needed
        return anthropic.AnthropicVertex(), client_model
    elif 'gpt' in model or "o1" in model or "o3" in model:
        print(f"Using OpenAI API with model {model}.")
        if "OPENAI_API_KEY" not in os.environ: raise ValueError("OPENAI_API_KEY missing for direct OpenAI call.")
        return openai.OpenAI(), model
    elif model in ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]:
        print(f"Using DeepSeek API (via OpenAI compatible endpoint) with {model}.")
        if "DEEPSEEK_API_KEY" not in os.environ: raise ValueError("DEEPSEEK_API_KEY missing for direct DeepSeek call.")
        return openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        ), model
    # Note: llama3.1-405b is implicitly handled by the OpenRouter check if key is set and mapping exists.
    # If key isn't set, this function currently has no specific handler for it.
    elif "gemini" in model:
        print(f"Using Google Gemini API (via OpenAI compatible endpoint) with {model}.")
        if "GEMINI_API_KEY" not in os.environ: raise ValueError("GEMINI_API_KEY missing for direct Gemini call.")
        # Assuming the base_url is correct for the user's specific proxy/setup
        return openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ), model
    else:
        raise ValueError(f"Model {model} not supported or appropriate API key for direct access is missing (and OPENROUTER_API_KEY not set).")