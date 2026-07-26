namespace llm {
namespace openAIChatDetail {
juce::String resultContent(const ToolResult& result) {
    if (result.content.isString())
        return result.content.toString();
    if (!result.content.isVoid())
        return juce::JSON::toString(result.content);
    return result.error;
}

ToolCall parseCall(const juce::var& value) {
    ToolCall call;
    call.id = value["id"].toString();
    call.name = value["function"]["name"].toString();
    call.rawArguments = value["function"]["arguments"].toString();
    if (call.rawArguments.isEmpty())
        call.rawArguments = "{}";
    call.arguments = juce::JSON::parse(call.rawArguments);
    if (call.arguments.isVoid())
        call.error = "Malformed tool arguments: " + call.rawArguments.substring(0, 200);
    return call;
}
}  // namespace openAIChatDetail

juce::String OpenAIChatClient::buildRequestBody(const Request& request) const {
    auto messagesArray = juce::Array<juce::var>();

    auto* sysMsg = new juce::DynamicObject();
    sysMsg->setProperty("role", "system");
    sysMsg->setProperty("content", request.systemPrompt);
    messagesArray.add(juce::var(sysMsg));

    // Prior turns between the system prompt and the current user turn.
    for (const auto& m : request.messages) {
        if (m.role == "tool") {
            for (const auto& result : m.toolResults) {
                auto* turn = new juce::DynamicObject();
                turn->setProperty("role", "tool");
                turn->setProperty("tool_call_id", result.callId);
                turn->setProperty("content", openAIChatDetail::resultContent(result));
                messagesArray.add(juce::var(turn));
            }
            continue;
        }

        auto* turn = new juce::DynamicObject();
        turn->setProperty("role", m.role);
        turn->setProperty("content", m.content);
        if (m.role == "assistant" && !m.toolCalls.empty()) {
            juce::Array<juce::var> calls;
            for (const auto& call : m.toolCalls) {
                auto* function = new juce::DynamicObject();
                function->setProperty("name", call.name);
                function->setProperty("arguments", call.rawArguments.isNotEmpty()
                                                       ? call.rawArguments
                                                       : juce::JSON::toString(call.arguments));
                auto* toolCall = new juce::DynamicObject();
                toolCall->setProperty("id", call.id);
                toolCall->setProperty("type", "function");
                toolCall->setProperty("function", juce::var(function));
                calls.add(juce::var(toolCall));
            }
            turn->setProperty("tool_calls", calls);
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
    payload->setProperty("messages", messagesArray);
    if (!config_.noTemperature)
        payload->setProperty("temperature", (double)request.temperature);

    // Reasoning effort for GPT-5 / o-series (top-level field in Chat Completions)
    if (config_.reasoningEffort.isNotEmpty())
        payload->setProperty("reasoning_effort", config_.reasoningEffort);

    // Prompt caching — bucket by app+agent, retain for 24h
    if (config_.userAgent.isNotEmpty()) {
        payload->setProperty("prompt_cache_key", config_.userAgent);
        payload->setProperty("prompt_cache_retention", "24h");
    }

    // GBNF grammar for llama-server (per-config or per-request)
    auto grammar = config_.grammar.isNotEmpty() ? config_.grammar : request.grammar;
    if (grammar.isNotEmpty())
        payload->setProperty("grammar", grammar);

    // Max output tokens — per-request override or provider config
    // Use max_completion_tokens (required by newer OpenAI models like GPT-4o, o-series)
    int maxTok = request.maxTokens > 0 ? request.maxTokens : config_.maxTokens;
    if (maxTok > 0)
        payload->setProperty("max_completion_tokens", maxTok);

    // Structured output via JSON schema
    if (!request.schema.isVoid()) {
        auto* schemaWrapper = new juce::DynamicObject();
        schemaWrapper->setProperty("name", "response");
        schemaWrapper->setProperty("strict", true);
        schemaWrapper->setProperty("schema", request.schema);

        auto* responseFormat = new juce::DynamicObject();
        responseFormat->setProperty("type", "json_schema");
        responseFormat->setProperty("json_schema", juce::var(schemaWrapper));

        payload->setProperty("response_format", juce::var(responseFormat));
    }

    if (!request.tools.empty()) {
        juce::Array<juce::var> tools;
        for (const auto& definition : request.tools) {
            auto* function = new juce::DynamicObject();
            function->setProperty("name", definition.name);
            function->setProperty("description", definition.description);
            function->setProperty("parameters", definition.inputSchema);

            auto* tool = new juce::DynamicObject();
            tool->setProperty("type", "function");
            tool->setProperty("function", juce::var(function));
            tools.add(juce::var(tool));
        }
        payload->setProperty("tools", tools);

        switch (request.toolChoice.mode) {
            case ToolChoiceMode::None:
                payload->setProperty("tool_choice", "none");
                break;
            case ToolChoiceMode::Required:
                payload->setProperty("tool_choice", "required");
                break;
            case ToolChoiceMode::Specific: {
                auto* function = new juce::DynamicObject();
                function->setProperty("name", request.toolChoice.toolName);
                auto* choice = new juce::DynamicObject();
                choice->setProperty("type", "function");
                choice->setProperty("function", juce::var(function));
                payload->setProperty("tool_choice", juce::var(choice));
                break;
            }
            case ToolChoiceMode::Auto:
                payload->setProperty("tool_choice", "auto");
                break;
        }
    }

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String OpenAIChatClient::getEndpointUrl() const {
    return config_.baseUrl + "/chat/completions";
}

juce::StringPairArray OpenAIChatClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("Authorization", "Bearer " + config_.apiKey);
    headers.set("Content-Type", "application/json");

    // OpenRouter-specific headers for app identification
    if (config_.baseUrl.contains("openrouter.ai")) {
        if (config_.userAgent.isNotEmpty())
            headers.set("X-Title", config_.userAgent);
        if (config_.appUrl.isNotEmpty())
            headers.set("HTTP-Referer", config_.appUrl);
    }

    return headers;
}

Response OpenAIChatClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    if (auto* choices = json["choices"].getArray()) {
        if (choices->size() > 0) {
            const auto message = (*choices)[0]["message"];
            response.text = message["content"].toString().trim();
            if (auto* calls = message["tool_calls"].getArray())
                for (const auto& call : *calls)
                    response.toolCalls.push_back(openAIChatDetail::parseCall(call));
            response.success = response.text.isNotEmpty() || !response.toolCalls.empty();
        }
    }

    if (auto usage = json["usage"]; usage.isObject()) {
        response.inputTokens = static_cast<int>(usage["prompt_tokens"]);
        response.outputTokens = static_cast<int>(usage["completion_tokens"]);
        response.totalTokens = static_cast<int>(usage["total_tokens"]);
    }

    if (!response.success)
        response.error = "Failed to parse response: " + jsonString.substring(0, 200);

    return response;
}

std::vector<StreamDelta> OpenAIChatClient::parseStreamDeltas(const juce::String& dataLine) const {
    std::vector<StreamDelta> result;
    auto json = juce::JSON::parse(dataLine);
    auto* choices = json["choices"].getArray();
    if (choices == nullptr || choices->isEmpty())
        return result;

    const auto delta = (*choices)[0]["delta"];
    auto text = delta["content"].toString();
    if (text.isNotEmpty()) {
        StreamDelta textDelta;
        textDelta.type = StreamDeltaType::Text;
        textDelta.text = text;
        result.push_back(std::move(textDelta));
    }

    if (auto* calls = delta["tool_calls"].getArray()) {
        for (const auto& call : *calls) {
            const int index = static_cast<int>(call["index"]);
            const auto id = call["id"].toString();
            const auto name = call["function"]["name"].toString();
            if (id.isNotEmpty() || name.isNotEmpty()) {
                StreamDelta start;
                start.type = StreamDeltaType::ToolCallStart;
                start.toolCallIndex = index;
                start.callId = id;
                start.toolName = name;
                result.push_back(std::move(start));
            }
            const auto arguments = call["function"]["arguments"].toString();
            if (arguments.isNotEmpty()) {
                StreamDelta args;
                args.type = StreamDeltaType::ToolCallArguments;
                args.toolCallIndex = index;
                args.argumentsDelta = arguments;
                result.push_back(std::move(args));
            }
        }
    }
    return result;
}

}  // namespace llm
