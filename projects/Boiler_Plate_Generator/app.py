from openai import OpenAI
from dotenv import load_dotenv
import os, json

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def system_manager(command):
    """Execute a shell command and return the result."""
    print(f"  ⚙️  Running: {command}")
    result = os.system(command)
    return f"Command executed with exit code: {result}"

available_tools = {
    "system_manager": {
        "fn": system_manager,
        "description": "Executes a single shell command to create folders or files for the boilerplate."
    }
}

system_prompt = """You are an AI assistant that generates production-ready boilerplate project structures.

The user will describe a project and its tech stack. You will create the full folder and file structure
on the user's system by calling shell commands ONE AT A TIME.

You have access to the following tool:
- system_manager: Executes a single shell command (e.g. mkdir, echo/copy to create files).

IMPORTANT RULES:
1. Do NOT ask follow-up questions. If info is missing, make reasonable assumptions.
2. You must respond with EXACTLY ONE JSON object per turn.
3. Each command in an "action" step must be a SINGLE shell command (one mkdir, one file write, etc).
4. On Windows use: mkdir for folders, and (echo content) > filename for files.
5. Create the project inside a folder in the current directory.
6. Every generated file should contain a helpful comment at the top explaining its purpose.
7. After ALL files and folders are created, output a final "result" step.

OUTPUT FORMAT — respond with exactly one JSON object per message:

Step 1 (plan): Announce what you will build.
  {"step": "plan", "content": "I will create the boilerplate for X with the following structure: ..."}

Steps 2..N (action): Execute commands one at a time.
  {"step": "action", "function": "system_manager", "input": "mkdir project_name"}
  {"step": "action", "function": "system_manager", "input": "mkdir project_name\\\\subfolder"}

  For creating files with content on Windows use:
  {"step": "action", "function": "system_manager", "input": "(echo # comment line) > project_name\\\\filename.py"}

  IMPORTANT: Output ONLY ONE action per response. Wait for the tool result before sending the next action.

Final step (result): Summarize what was created.
  {"step": "result", "content": "Project boilerplate created successfully! Here is the structure: ..."}
"""

user_query = input("🗣️: ")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

max_iterations = 50  # allow enough iterations for many files/folders
iteration = 0

while iteration < max_iterations:
    iteration += 1

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            messages=messages
        )
    except Exception as e:
        print(f"❌ API Error: {e}")
        break

    content = response.choices[0].message.content
    if not content:
        print("❌ No response from API")
        break

    # Try to parse JSON
    try:
        parsed_output = json.loads(content)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON from API: {content[:200]}")
        break

    step = parsed_output.get("step")

    messages.append({
        "role": "assistant",
        "content": content
    })

    if step == "plan":
        print(f"\n🧠 Plan: {parsed_output.get('content')}\n")
        # After the plan, tell the LLM to proceed with the first action
        messages.append({
            "role": "user",
            "content": "Good plan! Now proceed step by step. Execute the first command."
        })
        continue

    elif step == "action":
        tool_name = parsed_output.get("function")
        tool_input = parsed_output.get("input")

        if tool_name in available_tools:
            tool_output = available_tools[tool_name]["fn"](tool_input)

            messages.append({
                "role": "user",
                "content": f"Tool result: {tool_output}. Now execute the next command, or if all files and folders are created, output a result step."
            })
            continue
        else:
            print(f"❌ Tool '{tool_name}' not found")
            break

    elif step == "result":
        print(f"\n🤖 Done: {parsed_output.get('content')}\n")
        break

    else:
        print(f"❌ Unknown step: '{step}' — raw: {content[:200]}")
        break

else:
    print("❌ Max iterations reached. Possible loop detected.")
