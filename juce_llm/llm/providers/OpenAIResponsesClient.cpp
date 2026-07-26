namespace llm {
namespace openAIResponsesDetail {
juce::String resultContent(const ToolResult& result) {
    if (result.content.isString())
        return result.content.toString();
    if (!result.content.isVoid())
        return juce::JSON::toString(result.content);
    return result.error;
}

ToolCall parseCall(const juce::var& value) {
    ToolCall call;
    call.id = value["call_id"].toString();
    call.name = value["name"].toString();
    call.rawArguments = value["arguments"].toString();
    if (call.rawArguments.isEmpty())
        call.rawArguments = "{}";
    call.arguments = juce::JSON::parse(call.rawArguments);
    if (call.arguments.isVoid())
        call.error = "Malformed tool arguments: " + call.rawArguments.substring(0, 200);
    return call;
}
}  // namespace openAIResponsesDetail

juce::String OpenAIResponsesClient::buildRequestBody(const Request& request) const {
    auto* payload = new juce::DynamicObject();
    payload->setProperty("model", config_.model);
    payload->setProperty("instructions", request.systemPrompt);

    juce::Array<juce::var> pendingInput;
    auto lastAssistant = request.messages.rend();
    for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it) {
        if (it->role == "assistant") {
            lastAssistant = it;
            break;
        }
    }
    if (lastAssistant != request.messages.rend()) {
        for (auto it = lastAssistant.base(); it != request.messages.end(); ++it) {
            if (it->role != "tool")
                continue;
            for (const auto& result : it->toolResults) {
                auto* output = new juce::DynamicObject();
                output->setProperty("type", "function_call_output");
                output->setProperty("call_id", result.callId);
                output->setProperty("output", openAIResponsesDetail::resultContent(result));
                pendingInput.add(juce::var(output));
            }
        }
    }

    if (!pendingInput.isEmpty()) {
        if (request.userMessage.isNotEmpty()) {
            auto* user = new juce::DynamicObject();
            user->setProperty("role", "user");
            user->setProperty("content", request.userMessage);
            pendingInput.add(juce::var(user));
        }
        payload->setProperty("input", pendingInput);
    } else {
        payload->setProperty("input", request.userMessage);
    }

    // Stateful multi-turn: chain off the prior response so the server retains
    // context (including reasoning items) instead of resending the history.
    // Only userMessage is sent as the new input. First turn leaves this empty.
    if (request.previousResponseId.isNotEmpty())
        payload->setProperty("previous_response_id", request.previousResponseId);

    if (!config_.noTemperature)
        payload->setProperty("temperature", (double)request.temperature);

    if (config_.reasoningEffort.isNotEmpty()) {
        auto* reasoning = new juce::DynamicObject();
        reasoning->setProperty("effort", config_.reasoningEffort);
        payload->setProperty("reasoning", juce::var(reasoning));
    }

    juce::Array<juce::var> tools;

    // CFG grammar-constrained output via custom tool. This remains a custom
    // output channel and is intentionally distinct from executable functions.
    if (request.grammar.isNotEmpty()) {
        auto toolName = request.grammarToolName.isNotEmpty() ? request.grammarToolName
                                                             : juce::String("grammar_tool");
        auto toolDesc = request.grammarToolDescription.isNotEmpty() ? request.grammarToolDescription
                                                                    : request.systemPrompt;

        auto* tool = new juce::DynamicObject();
        tool->setProperty("type", "custom");
        tool->setProperty("name", toolName);
        tool->setProperty("description", toolDesc);

        auto* format = new juce::DynamicObject();
        format->setProperty("type", "grammar");
        format->setProperty("syntax", "lark");
        format->setProperty("definition", request.grammar);
        tool->setProperty("format", juce::var(format));

        tools.add(juce::var(tool));
        payload->setProperty("parallel_tool_calls", false);
    }
    // Structured output via JSON schema. The Responses API flattens the
    // json_schema fields into a single `format` object under `text`:
    //   text.format.{type: "json_schema", name, strict, schema}
    // — not the Chat Completions shape (text.{type, json_schema: {...}}),
    // which the endpoint rejects with "Unknown parameter: text.type".
    else if (!request.schema.isVoid()) {
        auto* format = new juce::DynamicObject();
        format->setProperty("type", "json_schema");
        format->setProperty("name", "response");
        format->setProperty("strict", true);
        format->setProperty("schema", request.schema);

        auto* text = new juce::DynamicObject();
        text->setProperty("format", juce::var(format));

        payload->setProperty("text", juce::var(text));
    }

    for (const auto& definition : request.tools) {
        auto* tool = new juce::DynamicObject();
        tool->setProperty("type", "function");
        tool->setProperty("name", definition.name);
        tool->setProperty("description", definition.description);
        tool->setProperty("parameters", definition.inputSchema);
        tool->setProperty("strict", true);
        tools.add(juce::var(tool));
    }
    if (!tools.isEmpty())
        payload->setProperty("tools", tools);

    if (!request.tools.empty()) {
        switch (request.toolChoice.mode) {
            case ToolChoiceMode::None:
                payload->setProperty("tool_choice", "none");
                break;
            case ToolChoiceMode::Required:
                payload->setProperty("tool_choice", "required");
                break;
            case ToolChoiceMode::Specific: {
                auto* choice = new juce::DynamicObject();
                choice->setProperty("type", "function");
                choice->setProperty("name", request.toolChoice.toolName);
                payload->setProperty("tool_choice", juce::var(choice));
                break;
            }
            case ToolChoiceMode::Auto:
                payload->setProperty("tool_choice", "auto");
                break;
        }
    }

    // Max output tokens — per-request override or provider config
    int maxTok = request.maxTokens > 0 ? request.maxTokens : config_.maxTokens;
    if (maxTok > 0)
        payload->setProperty("max_output_tokens", maxTok);

    // Prompt caching — bucket by app+agent, retain for 24h
    if (config_.userAgent.isNotEmpty()) {
        payload->setProperty("prompt_cache_key", config_.userAgent);
        payload->setProperty("prompt_cache_retention", "24h");
    }

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String OpenAIResponsesClient::getEndpointUrl() const {
    return config_.baseUrl + "/responses";
}

juce::StringPairArray OpenAIResponsesClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("Authorization", "Bearer " + config_.apiKey);
    headers.set("Content-Type", "application/json");
    return headers;
}

Response OpenAIResponsesClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    // Capture the response id + usage up front so every return path carries
    // them (the output loop has several early returns).
    response.id = json["id"].toString();
    if (auto usage = json["usage"]; usage.isObject()) {
        response.inputTokens = static_cast<int>(usage["input_tokens"]);
        response.outputTokens = static_cast<int>(usage["output_tokens"]);
        response.totalTokens = static_cast<int>(usage["total_tokens"]);
    }

    if (auto* output = json["output"].getArray()) {
        for (const auto& item : *output) {
            auto type = item["type"].toString();

            // CFG grammar tool response — extract from custom_tool_call
            if (type == "custom_tool_call") {
                auto input = item["input"].toString().trim();
                if (input.isNotEmpty()) {
                    response.text += input;
                }
            }

            if (type == "function_call")
                response.toolCalls.push_back(openAIResponsesDetail::parseCall(item));

            // Standard text response — output[].content[].text
            if (type == "message") {
                if (auto* content = item["content"].getArray()) {
                    for (const auto& c : *content) {
                        if (c["type"].toString() == "output_text") {
                            response.text += c["text"].toString();
                        }
                    }
                }
            }
        }
    }

    response.text = response.text.trim();
    response.success = response.text.isNotEmpty() || !response.toolCalls.empty();
    if (!response.success)
        response.error = "Failed to parse response: " + jsonString.substring(0, 200);
    return response;
}

/** Responses API SSE chunks look like:
      data: {"type":"response.output_text.delta","delta":"token","item_id":"..."}
      data: {"type":"response.completed", ...}
    The base LLMClient class only reads "data: " lines, so we just need to pull
    the "delta" field out of the JSON when the event type matches. */
juce::String OpenAIResponsesClient::parseStreamChunk(const juce::String& dataLine) const {
    auto json = juce::JSON::parse(dataLine);
    auto type = json["type"].toString();
    if (type == "response.output_text.delta")
        return json["delta"].toString();
    return {};
}

std::vector<StreamDelta> OpenAIResponsesClient::parseStreamDeltas(
    const juce::String& dataLine) const {
    std::vector<StreamDelta> result;
    auto json = juce::JSON::parse(dataLine);
    const auto type = json["type"].toString();

    if (type == "response.output_text.delta") {
        StreamDelta delta;
        delta.type = StreamDeltaType::Text;
        delta.text = json["delta"].toString();
        result.push_back(std::move(delta));
    } else if (type == "response.output_item.added" &&
               json["item"]["type"].toString() == "function_call") {
        StreamDelta delta;
        delta.type = StreamDeltaType::ToolCallStart;
        delta.toolCallIndex = static_cast<int>(json["output_index"]);
        delta.callId = json["item"]["call_id"].toString();
        delta.toolName = json["item"]["name"].toString();
        result.push_back(std::move(delta));
    } else if (type == "response.function_call_arguments.delta") {
        StreamDelta delta;
        delta.type = StreamDeltaType::ToolCallArguments;
        delta.toolCallIndex = static_cast<int>(json["output_index"]);
        delta.argumentsDelta = json["delta"].toString();
        result.push_back(std::move(delta));
    } else if (type == "response.function_call_arguments.done") {
        StreamDelta delta;
        delta.type = StreamDeltaType::ToolCallEnd;
        delta.toolCallIndex = static_cast<int>(json["output_index"]);
        result.push_back(std::move(delta));
    }
    return result;
}

}  // namespace llm
