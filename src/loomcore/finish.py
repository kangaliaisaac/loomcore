"""The typed `finish(result)` tool: how the model says the task is done."""

from pydantic import JsonValue

from loomcore.events import JsonObject

NAME = "finish"

SCHEMA: JsonObject = {
    "name": NAME,
    "description": "End the run. Call it once, with the final result of the task.",
    "input_schema": {
        "type": "object",
        "properties": {"result": {"description": "The final result of the task."}},
        "required": ["result"],
    },
}


def result_of(args: JsonObject) -> JsonValue:
    return args.get("result")
