namespace llm {
namespace anthropicDetail {
juce::String resultContent(const ToolResult& result) {
    if (result.content.isString())
        return result.content.toString();
    if (!result.content.isVoid())
        return juce::JSON::toString(result.content);
    return result.error;
}
}  // namespace anthropicDetail

juce::String AnthropicClient::buildRequestBody(const Request& request) const {
    auto messagesArray = juce::Array<juce::var>();

    // Prior turns first (empty = single-shot, identical to before).
    for (const auto& m : request.messages) {
        auto* turn = new juce::DynamicObject();
        turn->setProperty("role", m.role == "tool" ? "user" : m.role);

        if (m.role == "tool") {
            juce::Array<juce::var> blocks;
            for (const auto& result : m.toolResults) {
                auto* block = new juce::DynamicObject();
                block->setProperty("type", "tool_result");
                block->setProperty("tool_use_id", result.callId);
                block->setProperty("content", anthropicDetail::resultContent(result));
                block->setProperty("is_error", result.isError || result.error.isNotEmpty());
                blocks.add(juce::var(block));
            }
            turn->setProperty("content", blocks);
        } else if (m.role == "assistant" && !m.toolCalls.empty()) {
            juce::Array<juce::var> blocks;
            if (m.content.isNotEmpty()) {
                auto* text = new juce::DynamicObject();
                text->setProperty("type", "text");
                text->setProperty("text", m.content);
                blocks.add(juce::var(text));
            }
            for (const auto& call : m.toolCalls) {
                auto* block = new juce::DynamicObject();
                block->setProperty("type", "tool_use");
                block->setProperty("id", call.id);
                block->setProperty("name", call.name);
                block->setProperty("input", call.arguments);
                blocks.add(juce::var(block));
            }
            turn->setProperty("content", blocks);
        } else {
            turn->setProperty("content", m.content);
        }
        messagesArray.add(juce::var(turn));
    }

    if (request.userMessage.isNotEmpty()) {
        auto* userMsg = new juce::DynamicObject();
        userMsg->setProperty("role", "user");
        userMsg->setProperty("content", request.userMessage);
        messagesArray.add(juce::var(userMsg));
    }

    auto* payload = new juce::DynamicObject();
    payload->setProperty("model", config_.model);
    int maxTok = request.maxTokens > 0 ? request.maxTokens
                                       : (config_.maxTokens > 0 ? config_.maxTokens : 4096);
    payload->setProperty("max_tokens", maxTok);
    // Some newer models (e.g. Claude Opus 4.8) deprecated the temperature
    // parameter and reject the request if it's present.
    if (!config_.noTemperature)
        payload->setProperty("temperature", (double)request.temperature);
    payload->setProperty("messages", messagesArray);

    // System prompt with prompt caching — cache the system prompt block
    // since it's identical across calls for each agent
    if (request.systemPrompt.isNotEmpty()) {
        auto* sysBlock = new juce::DynamicObject();
        sysBlock->setProperty("type", "text");
        sysBlock->setProperty("text", request.systemPrompt);

        auto* cacheControl = new juce::DynamicObject();
        cacheControl->setProperty("type", "ephemeral");
        sysBlock->setProperty("cache_control", juce::var(cacheControl));

        juce::Array<juce::var> systemArray;
        systemArray.add(juce::var(sysBlock));
        payload->setProperty("system", systemArray);
    }

    // Structured output. Anthropic honours `output_config.format` with a JSON
    // schema (GA, no beta header) — the Anthropic analogue of the OpenAI
    // `response_format` path. Set it whenever the caller supplied a schema so
    // the model is constrained to schema-valid JSON, returned as its text
    // block (so parseResponseBody needs no special handling). Unlike OpenAI's
    // json_schema, no `name` field is required here.
    if (!request.schema.isVoid()) {
        auto* format = new juce::DynamicObject();
        format->setProperty("type", "json_schema");
        format->setProperty("schema", request.schema);

        auto* outputConfig = new juce::DynamicObject();
        outputConfig->setProperty("format", juce::var(format));
        payload->setProperty("output_config", juce::var(outputConfig));
    }

    if (!request.tools.empty()) {
        juce::Array<juce::var> tools;
        for (const auto& definition : request.tools) {
            auto* tool = new juce::DynamicObject();
            tool->setProperty("name", definition.name);
            tool->setProperty("description", definition.description);
            tool->setProperty("input_schema", definition.inputSchema);
            tools.add(juce::var(tool));
        }
        payload->setProperty("tools", tools);

        auto* choice = new juce::DynamicObject();
        switch (request.toolChoice.mode) {
            case ToolChoiceMode::None:
                choice->setProperty("type", "none");
                break;
            case ToolChoiceMode::Required:
                choice->setProperty("type", "any");
                break;
            case ToolChoiceMode::Specific:
                choice->setProperty("type", "tool");
                choice->setProperty("name", request.toolChoice.toolName);
                break;
            case ToolChoiceMode::Auto:
                choice->setProperty("type", "auto");
                break;
        }
        payload->setProperty("tool_choice", juce::var(choice));
    }

    // NOTE: `output_config.effort` is an OpenAI-ism that Anthropic rejects, so
    // reasoningEffort is deliberately ignored here. Only `output_config.format`
    // (above) is emitted. Extended thinking uses a separate `thinking` block on
    // models that support it.

    // App identification for abuse tracking
    if (config_.userAgent.isNotEmpty()) {
        auto* metadata = new juce::DynamicObject();
        metadata->setProperty("user_id", config_.userAgent);
        payload->setProperty("metadata", juce::var(metadata));
    }

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String AnthropicClient::getEndpointUrl() const {
    return config_.baseUrl + "/messages";
}

juce::StringPairArray AnthropicClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("x-api-key", config_.apiKey);
    headers.set("anthropic-version", "2023-06-01");
    headers.set("Content-Type", "application/json");
    return headers;
}

Response AnthropicClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    if (auto* content = json["content"].getArray()) {
        for (const auto& block : *content) {
            const auto type = block["type"].toString();
            if (type == "text")
                response.text += block["text"].toString();
            else if (type == "tool_use") {
                ToolCall call;
                call.id = block["id"].toString();
                call.name = block["name"].toString();
                call.arguments = block["input"];
                call.rawArguments = juce::JSON::toString(call.arguments);
                response.toolCalls.push_back(std::move(call));
            }
        }
    }
    response.text = response.text.trim();
    response.success = response.text.isNotEmpty() || !response.toolCalls.empty();

    if (auto usage = json["usage"]; usage.isObject()) {
        response.inputTokens = static_cast<int>(usage["input_tokens"]);
        response.outputTokens = static_cast<int>(usage["output_tokens"]);
        // Anthropic reports the two separately; total is their sum.
        response.totalTokens = response.inputTokens + response.outputTokens;
    }

    if (!response.success)
        response.error = "Failed to parse response: " + jsonString.substring(0, 200);

    return response;
}

// Anthropic SSE format:
//   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"token"}}
juce::String AnthropicClient::parseStreamChunk(const juce::String& dataLine) const {
    auto json = juce::JSON::parse(dataLine);
    auto type = json["type"].toString();
    if (type == "content_block_delta")
        return json["delta"]["text"].toString();
    return {};
}

std::vector<StreamDelta> AnthropicClient::parseStreamDeltas(const juce::String& dataLine) const {
    std::vector<StreamDelta> result;
    auto json = juce::JSON::parse(dataLine);
    const auto type = json["type"].toString();
    const int index = static_cast<int>(json["index"]);

    if (type == "content_block_start" && json["content_block"]["type"].toString() == "tool_use") {
        StreamDelta delta;
        delta.type = StreamDeltaType::ToolCallStart;
        delta.toolCallIndex = index;
        delta.callId = json["content_block"]["id"].toString();
        delta.toolName = json["content_block"]["name"].toString();
        result.push_back(std::move(delta));
    } else if (type == "content_block_delta") {
        const auto deltaType = json["delta"]["type"].toString();
        StreamDelta delta;
        delta.toolCallIndex = index;
        if (deltaType == "text_delta") {
            delta.type = StreamDeltaType::Text;
            delta.text = json["delta"]["text"].toString();
            result.push_back(std::move(delta));
        } else if (deltaType == "input_json_delta") {
            delta.type = StreamDeltaType::ToolCallArguments;
            delta.argumentsDelta = json["delta"]["partial_json"].toString();
            result.push_back(std::move(delta));
        }
    }
    return result;
}

}  // namespace llm
