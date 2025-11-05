from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv('API_KEY')

client = OpenAI(api_key=api_key)

tools = [
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a command, as a subprocess.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "description": 'Command to run, as a list of strings. Example: ["ls", "-la"]',
                    "items": {
                        "type": "string",
                    }
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

conversation = [
    {'role': 'system', 'content': 'You are a helpful agent.'},
    {'role': 'user', 'content': 'Run the commands "foo" and "bar".'},
]

counter = 0
while True:
    for item in conversation:
        print(item)
    print('===================================================')
    response = client.responses.create(
      model="gpt-5-nano",
      reasoning={'effort': 'medium'},
      input=conversation,
      store=True,
      tools=tools,
      parallel_tool_calls=True,
    )
    print(response.model_dump_json(indent=2))
    print('---------------------------------------------------------')
    done = True
    for item in response.output:
        if item.type == 'reasoning':
            conversation.append(item)
        elif item.type == 'function_call':
            conversation.append(item)
            conversation.append({
                'type': 'function_call_output',
                'call_id': item.call_id,
                'output': 'Error: command not found',
            })
            done = False
        elif item.type == 'message':
            conversation.append(item)
        else:
            raise Exception(f'Unknown item type {item.type}')
    counter += 1
    if done:
        break
    if counter > 4:
        raise Exception('Too many iterations')

print('')
print('============= Final conversation: =================')
for item in conversation:
    print(item)
print('===================================================')

