import json, time
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Union
import warnings
import logging

import boto3
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
        self.model_id = kwargs.get("model_id", None)
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
        system_messages = extract_system_messages(params["messages"])
        messages = [x for x in params["messages"] if not x["role"] == "system"]
        messages = format_messages(messages)

        try:
            response = self.bedrock_runtime.converse(
                messages=messages,
                modelId=self.model_id,
                system=system_messages,
                # toolConfig=tool_config,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Bedrock: {e}")

        if response is None:
            raise RuntimeError(f"Failed to get response from Bedrock after retrying {RETRY_NUMBER} times.")

        finish_reason = convert_stop_reason_to_finish_reason(response["stopReason"])
        response_message = response["output"]["message"]
        role = response_message["role"]

        for content in response_message["content"]:
            if "text" in content:
                text: str = content["text"]
                message = ChatCompletionMessage(content=text, role=role, tool_calls=None)
            # NOTE: other type of output may be dealt with here

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
            model=self.model_id,
            object="chat.completion",
            usage=usage,
        )

    # def _format_tools(self, tools):
    #     converted_schema = {"tools": []}

    #     for tool in tools:
    #         if tool["type"] == "function":
    #             function = tool["function"]
    #             converted_tool = {
    #                 "toolSpec": {
    #                     "name": function["name"],
    #                     "description": function["description"],
    #                     "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
    #                 }
    #             }

    #             for prop_name, prop_details in function["parameters"]["properties"].items():
    #                 converted_tool["toolSpec"]["inputSchema"]["json"]["properties"][prop_name] = {
    #                     "type": prop_details["type"],
    #                     "description": prop_details.get("description", ""),
    #                 }
    #                 if "enum" in prop_details:
    #                     converted_tool["toolSpec"]["inputSchema"]["json"]["properties"][prop_name]["enum"] = (
    #                         prop_details["enum"]
    #                     )
    #                 if "default" in prop_details:
    #                     converted_tool["toolSpec"]["inputSchema"]["json"]["properties"][prop_name]["default"] = (
    #                         prop_details["default"]
    #                     )

    #             if "required" in function["parameters"]:
    #                 converted_tool["toolSpec"]["inputSchema"]["json"]["required"] = function["parameters"]["required"]

    #             converted_schema["tools"].append(converted_tool)

    #     return converted_schema

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


def format_messages(messages: List[Dict]) -> List[Dict]:
    formatted_messages = []
    for message in messages:
        content = message.get("content", "")
        role = message.get("role")

        # Bedrock system messages are passed as request parameter
        if role == "system":
            continue

        if type(content) == str:
            if len(content.strip()) == 0:
                content = ""
            content = [{"text": content}]

        formatted_messages.append({"role": role, "content": content})
    return formatted_messages


# NOTE: these needs to be changed with the results of the pricing API, stored in a json file (or obtained via boto3?) and segmented by region
PRICES_PER_K_TOKENS = {
    "meta.llama3-8b-instruct-v1:0": (0.0003, 0.0006),
    "meta.llama3-70b-instruct-v1:0": (0.00265, 0.0035),
    "mistral.mistral-7b-instruct-v0:2": (0.00015, 0.0002),
    "mistral.mixtral-8x7b-instruct-v0:1": (0.00045, 0.0007),
    "mistral.mistral-large-2402-v1:0": (0.004, 0.012),
    "mistral.mistral-small-2402-v1:0": (0.001, 0.003),
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


def convert_stop_reason_to_finish_reason(
    stop_reason: str,
) -> Literal["stop", "length", "tool_calls", "content_filter", "function_call"]:
    """Convert Bedrock stop_reason to OpenAI finish_reason

    Returns:
        _type_: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    """
    # possible bedrock stop_reason values: end_turn | tool_use | max_tokens | stop_sequence | guardrail_intervened | content_filtered
    # possible openai finish_reason values: stop", "length", "tool_calls", "content_filter"

    if stop_reason == "end_turn" or stop_reason == "stop_sequence":
        return "stop"
    elif stop_reason == "tool_use":
        return "tool_calls"
    elif stop_reason == "max_tokens":
        return "length"
    elif stop_reason == "content_filtered":
        return "content_filter"
    else:
        warnings.warn(f"Unsopported stop reason: {stop_reason}", UserWarning)
        return "stop"


def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """Calculate the cost of the completion using the Bedrock pricing."""

    if model_id in PRICES_PER_K_TOKENS:
        input_cost_per_k, output_cost_per_k = PRICES_PER_K_TOKENS[model_id]
        input_cost = (input_tokens / 1000) * input_cost_per_k
        output_cost = (output_tokens / 1000) * output_cost_per_k
        return input_cost + output_cost
    else:
        # NOTE: this should never happen because no request guardail can be configured
        warnings.warn(f"Cost calculation not available for model {model_id}", UserWarning)
        logger.warning(
            f'Cannot get the costs for {model_id}. The cost will be 0. In your config_list, add field {{"price" : [prompt_price_per_1k, completion_token_price_per_1k]}} for customized pricing.'
        )
        return 0
