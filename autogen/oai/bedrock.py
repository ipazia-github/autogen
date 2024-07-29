import base64
import json
import logging
import re
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import boto3
import requests
from botocore.config import Config
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)

RETRY_NUMBER = 5


class BedrockClient:
    def __init__(self, **kwargs):
        # generate docstring for this method
        """Requires any valid combination of the standard set of parameters required for the creation of a boto3 client

        Args:
            aws_access_key_id (str): AWS access key ID
            aws_secret_access_key (str): AWS secret access key
            aws_session_token (str): AWS temporary session token
            region_name (str): Default region when creating new connections
            profile_name (str): The name of a profile to use. If not given, then the default profile is used.

        """
        self.region_name = kwargs.get("region_name", None)
        self.aws_access_key_id = kwargs.get("aws_access_key_id", None)
        self.aws_secret_access_key = kwargs.get("aws_secret_access_key", None)
        self.aws_session_token = kwargs.get("aws_session_token", None)
        self.profile_name = kwargs.get("profile_name", None)
        self.max_tokens = kwargs.get("max_tokens", 512)

        # Initialize Bedrock client
        bedrock_config = Config(
            region_name=self.region_name,
            signature_version="v4",
            retries={"max_attempts": RETRY_NUMBER, "mode": "standard"},
        )

        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            profile_name=self.profile_name,
        )

        self.bedrock_runtime = session.client(service_name="bedrock-runtime", config=bedrock_config)

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        return [choice.message.content for choice in response.choices]

    def create(self, params: Dict[str, Any]) -> ChatCompletion:
        model_id = params.get("model_id", None)
        has_tools = "tools" in params
        messages = oai_messages_to_bedrock_messages(params["messages"], has_tools)
        system_messages = extract_system_messages(params["messages"])
        tool_config = format_tools(params["tools"] if has_tools else [])

        request_args = {"messages": messages, "modelId": model_id, "system": system_messages}
        if len(tool_config["tools"]) > 0:
            request_args["toolConfig"] = tool_config

        try:
            response = self.bedrock_runtime.converse(
                **request_args,
                # messages=messages,
                # modelId=model_id,
                # system=system_messages,
                # toolConfig=tool_config,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Bedrock: {e}")

        if response is None:
            raise RuntimeError(f"Failed to get response from Bedrock after retrying {RETRY_NUMBER} times.")

        finish_reason = convert_stop_reason_to_finish_reason(response["stopReason"])
        response_message = response["output"]["message"]

        if finish_reason == "tool_calls":
            tool_calls = format_tool_calls(response_message["toolCalls"])
            text = ""
        else:
            tool_calls = None
            for content in response_message["content"]:
                if "text" in content:
                    text: str = content["text"]
                # NOTE: other type of output may be dealt with here

        message = ChatCompletionMessage(role="assistant", content=text, tool_calls=tool_calls)

        reponse_usage = response["usage"]
        usage = CompletionUsage(
            prompt_tokens=reponse_usage["inputTokens"],
            completion_tokens=reponse_usage["outputTokens"],
            total_tokens=reponse_usage["totalTokens"],
        )

        return ChatCompletion(
            id=response["ResponseMetadata"]["RequestId"],
            choices=[Choice(finish_reason=finish_reason, index=0, message=message)],
            created=int(time.time()),
            model=model_id,
            object="chat.completion",
            usage=usage,
        )

    def cost(self, response: ChatCompletion) -> float:
        """Calculate the cost of the response."""
        return calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens, response.model)

    @staticmethod
    def get_usage(response: ChatCompletion) -> Optional[CompletionUsage]:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": response.cost,
            "model": response.model,
        }


def extract_system_messages(messages: List[dict]) -> List:
    """Extract the system messages from the list of messages.

    Args:
        messages (list[dict]): List of messages.

    Returns:
        List[SystemMessage]: List of System messages.
    """
    for message in messages:
        if message.get("role") == "system":
            return [{"text": message.get("content")}]
    return []


def oai_messages_to_bedrock_messages(messages: List[Dict[str, Any]], has_tools: bool) -> List[Dict]:
    """
    Convert messages from OAI format to Bedrock format.
    We correct for any specific role orders and types, etc.
    AWS Bedrock requires messages to alternate between user and assistant roles. This function ensures that the messages
    are in the correct order and format for Bedrock by inserting "Please continue" messages as needed.
    This is the same method as the one in the Autogen Anthropic client
    """

    # Track whether we have tools passed in. If not,  tool use / result messages should be converted to text messages.
    # Bedrock requires a tools parameter with the tools listed, if there are other messages with tool use or tool results.
    # This can occur when we don't need tool calling, such as for group chat speaker selection

    # Convert messages to Bedrock compliant format
    messages = [x for x in messages if not x["role"] == "system"]
    processed_messages = []

    # Used to interweave user messages to ensure user/assistant alternating
    user_continue_message = {"content": [{"text": "Please continue."}], "role": "user"}
    assistant_continue_message = {
        "content": [{"text": "Please continue."}],
        "role": "assistant",
    }

    tool_use_messages = 0
    tool_result_messages = 0
    last_tool_use_index = -1
    last_tool_result_index = -1
    for message in messages:
        # New messages will be added here, manage role alternations
        expected_role = "user" if len(processed_messages) % 2 == 0 else "assistant"

        if "tool_calls" in message:
            # Map the tool call options to Bedrock's format
            tool_uses = []
            tool_names = []
            for tool_call in message["tool_calls"]:
                tool_uses.append(
                    {
                        "toolUse": {
                            "toolUseId": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                    }
                )
                if has_tools:
                    tool_use_messages += 1
                tool_names.append(tool_call["function"]["name"])

            if expected_role == "user":
                # Insert an extra user message as we will append an assistant message
                processed_messages.append(user_continue_message)

            if has_tools:
                processed_messages.append({"role": "assistant", "content": tool_uses})
                last_tool_use_index = len(processed_messages) - 1
            else:
                # Not using tools, so put in a plain text message
                processed_messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"text": f"Some internal function(s) that could be used: [{', '.join(tool_names)}]"}
                        ],
                    }
                )
        elif "tool_call_id" in message:
            if has_tools:
                # Map the tool usage call to tool_result for Bedrock
                tool_result = {
                    "toolResult": {
                        "toolUseId": message["tool_call_id"],
                        "content": [{"text": message["content"]}],
                    }
                }

                # If the previous message also had a tool_result, add it to that
                # Otherwise append a new message
                if last_tool_result_index == len(processed_messages) - 1:
                    processed_messages[-1]["content"].append(tool_result)
                else:
                    if expected_role == "assistant":
                        # Insert an extra assistant message as we will append a user message
                        processed_messages.append(assistant_continue_message)

                    processed_messages.append({"role": "user", "content": [tool_result]})
                    last_tool_result_index = len(processed_messages) - 1

                tool_result_messages += 1
            else:
                # Not using tools, so put in a plain text message
                processed_messages.append(
                    {
                        "role": "user",
                        "content": [{"text": f"Running the function returned: {message['content']}"}],
                    }
                )
        elif message["content"] == "":
            # Ignoring empty messages
            pass
        else:
            if expected_role != message["role"]:
                # Inserting the alternating continue message
                processed_messages.append(
                    user_continue_message if expected_role == "user" else assistant_continue_message
                )

            processed_messages.append(
                {
                    "role": message["role"],
                    "content": parse_content_parts(message=message),
                }
            )

    # We'll replace the last tool_use if there's no tool_result (occurs if we finish the conversation before running the function)
    if has_tools and tool_use_messages != tool_result_messages:
        processed_messages[last_tool_use_index] = assistant_continue_message

    # name is not a valid field on messages
    for message in processed_messages:
        if "name" in message:
            message.pop("name", None)

    # Note: When using reflection_with_llm we may end up with an "assistant" message as the last message and that may cause a blank response
    # So, if the last role is not user, add a 'user' continue message at the end
    if processed_messages[-1]["role"] != "user":
        processed_messages.append(user_continue_message)

    return processed_messages


def parse_content_parts(
    message: Dict[str, Any],
) -> List[dict]:
    content: str | List[Dict[str, Any]] = message.get("content")
    if isinstance(content, str):
        return [
            {
                "text": content,
            }
        ]
    content_parts = []
    for part in content:
        part_content: Dict = part.get("content")
        if "text" in part_content:
            content_parts.append(
                {
                    "text": part_content.get("text"),
                }
            )
        elif "image_url" in part_content:
            image_data, content_type = parse_image(part.get("image_url").get("url"))
            content_parts.append(
                {
                    "image": {
                        "format": content_type[6:],  # image/
                        "source": {"bytes": image_data},
                    },
                }
            )
        else:
            # Ignore..
            continue
    return content_parts


def parse_image(image_url: str) -> Tuple[bytes, str]:
    """Try to get the raw data from an image url.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
    returns a tuple of (Image Data, Content Type)
    """
    pattern = r"^data:(image/[a-z]*);base64,\s*"
    content_type = re.search(pattern, image_url)
    # if already base64 encoded.
    # Only supports 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'
    if content_type:
        image_data = re.sub(pattern, "", image_url)
        return base64.b64decode(image_data), content_type.group(1)

    # Send a request to the image URL
    response = requests.get(image_url)
    # Check if the request was successful
    if response.status_code == 200:

        content_type = response.headers.get("Content-Type")
        if not content_type.startswith("image"):
            content_type = "image/jpeg"
        # Get the image content
        image_content = response.content
        return image_content, content_type
    else:
        raise RuntimeError("Unable to access the image url")


def format_tools(tools: List[Dict[str, Any]]) -> Dict[Literal["tools"], List[Dict[str, Any]]]:
    converted_schema = {"tools": []}

    for tool in tools:
        if tool["type"] == "function":
            function = tool["function"]
            converted_tool = {
                "toolSpec": {
                    "name": function["name"],
                    "description": function["description"],
                    "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
                }
            }

            for prop_name, prop_details in function["parameters"]["properties"].items():
                converted_tool["toolSpec"]["inputSchema"]["json"]["properties"][prop_name] = {
                    "type": prop_details["type"],
                    "description": prop_details.get("description", ""),
                }
                if "enum" in prop_details:
                    converted_tool["toolSpec"]["inputSchema"]["json"]["properties"][prop_name]["enum"] = prop_details[
                        "enum"
                    ]
                if "default" in prop_details:
                    converted_tool["toolSpec"]["inputSchema"]["json"]["properties"][prop_name]["default"] = (
                        prop_details["default"]
                    )

            if "required" in function["parameters"]:
                converted_tool["toolSpec"]["inputSchema"]["json"]["required"] = function["parameters"]["required"]

            converted_schema["tools"].append(converted_tool)

    return converted_schema


def format_tool_calls(content):
    tool_calls = []
    for tool_request in content:
        if "toolUse" in tool_request:
            tool = tool_request["toolUse"]

            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool["toolUseId"],
                    function={
                        "name": tool["name"],
                        "arguments": json.dumps(tool["input"]),
                    },
                    type="function",
                )
            )
    return tool_calls


def convert_stop_reason_to_finish_reason(
    stop_reason: str,
) -> Literal["stop", "length", "tool_calls", "content_filter"]:
    """
    Below is a list of finish reason according to OpenAI doc:

    - stop: if the model hit a natural stop point or a provided stop sequence,
    - length: if the maximum number of tokens specified in the request was reached,
    - content_filter: if content was omitted due to a flag from our content filters,
    - tool_calls: if the model called a tool
    """
    if stop_reason:
        finish_reason_mapping = {
            "tool_use": "tool_calls",
            "finished": "stop",
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "complete": "stop",
            "content_filtered": "content_filter",
        }
        return finish_reason_mapping.get(stop_reason.lower(), stop_reason.lower())
    warnings.warn(f"Unsopported stop reason: {stop_reason}", UserWarning)
    return None


# NOTE: these needs to be changed with the results of the pricing API, stored in a json file (or obtained via boto3?) and segmented by region
PRICES_PER_K_TOKENS = {
    "meta.llama3-8b-instruct-v1:0": (0.0003, 0.0006),
    "meta.llama3-70b-instruct-v1:0": (0.00265, 0.0035),
    "mistral.mistral-7b-instruct-v0:2": (0.00015, 0.0002),
    "mistral.mixtral-8x7b-instruct-v0:1": (0.00045, 0.0007),
    "mistral.mistral-large-2402-v1:0": (0.004, 0.012),
    "mistral.mistral-small-2402-v1:0": (0.001, 0.003),
}


def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """Calculate the cost of the completion using the Bedrock pricing."""

    if model_id in PRICES_PER_K_TOKENS:
        input_cost_per_k, output_cost_per_k = PRICES_PER_K_TOKENS[model_id]
        input_cost = (input_tokens / 1000) * input_cost_per_k
        output_cost = (output_tokens / 1000) * output_cost_per_k
        return input_cost + output_cost
    else:
        warnings.warn(f"Cost calculation not available for model {model_id}", UserWarning)
        logger.warning(
            f'Cannot get the costs for {model_id}. The cost will be 0. In your config_list, add field {{"price" : [prompt_price_per_1k, completion_token_price_per_1k]}} for customized pricing.'
        )
        return 0
