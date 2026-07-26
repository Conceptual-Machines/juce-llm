namespace llm {
namespace geminiDetail {
juce::var resultResponse(const ToolResult& result) {
    if (result.content.isObject())
        return result.content;

    auto* response = new juce::DynamicObject();
    if (result.isError || result.error.isNotEmpty())
        response->setProperty("error",
                              result.error.isNotEmpty() ? juce::var(result.error) : result.content);
    else
        response->setProperty("result", result.content);
    return juce::var(response);
}
}  // namespace geminiDetail

juce::String GeminiClient::buildRequestBody(const Request& request) const {
    // System instruction
    auto* sysPart = new juce::DynamicObject();
    sysPart->setProperty("text", request.systemPrompt);

    auto sysPartsArray = juce::Array<juce::var>();
    sysPartsArray.add(juce::var(sysPart));

    auto* sysInstruction = new juce::DynamicObject();
    sysInstruction->setProperty("parts", sysPartsArray);

    auto contentsArray = juce::Array<juce::var>();

    // Prior turns first; Gemini names the assistant role "model".
    for (const auto& m : request.messages) {
        auto partsArray = juce::Array<juce::var>();

        if (m.content.isNotEmpty()) {
            auto* part = new juce::DynamicObject();
            part->setProperty("text", m.content);
            partsArray.add(juce::var(part));
        }
        for (const auto& call : m.toolCalls) {
            auto* functionCall = new juce::DynamicObject();
            functionCall->setProperty("id", call.id);
            functionCall->setProperty("name", call.name);
            functionCall->setProperty("args", call.arguments);
            auto* part = new juce::DynamicObject();
            part->setProperty("functionCall", juce::var(functionCall));
            if (!call.providerData["thoughtSignature"].isVoid())
                part->setProperty("thoughtSignature", call.providerData["thoughtSignature"]);
            partsArray.add(juce::var(part));
        }
        for (const auto& result : m.toolResults) {
            auto* functionResponse = new juce::DynamicObject();
            functionResponse->setProperty("id", result.callId);
            functionResponse->setProperty("name", result.name);
            functionResponse->setProperty("response", geminiDetail::resultResponse(result));
            auto* part = new juce::DynamicObject();
            part->setProperty("functionResponse", juce::var(functionResponse));
            partsArray.add(juce::var(part));
        }

        if (partsArray.isEmpty())
            continue;

        auto* content = new juce::DynamicObject();
        content->setProperty("role", m.role == "assistant" ? "model" : "user");
        content->setProperty("parts", partsArray);
        contentsArray.add(juce::var(content));
    }

    // Current user turn.
    if (request.userMessage.isNotEmpty()) {
        auto* userPart = new juce::DynamicObject();
        userPart->setProperty("text", request.userMessage);

        auto userPartsArray = juce::Array<juce::var>();
        userPartsArray.add(juce::var(userPart));

        auto* userContent = new juce::DynamicObject();
        userContent->setProperty("role", "user");
        userContent->setProperty("parts", userPartsArray);
        contentsArray.add(juce::var(userContent));
    }

    // Generation config
    auto* genConfig = new juce::DynamicObject();
    genConfig->setProperty("temperature", (double)request.temperature);
    int maxTok = request.maxTokens > 0 ? request.maxTokens : config_.maxTokens;
    if (maxTok > 0)
        genConfig->setProperty("maxOutputTokens", maxTok);

    // Thinking config for Gemini 2.5 models
    if (config_.reasoningEffort.isNotEmpty()) {
        auto* thinkingConfig = new juce::DynamicObject();
        thinkingConfig->setProperty("thinkingBudget", config_.reasoningEffort == "low"    ? 1024
                                                      : config_.reasoningEffort == "high" ? 16384
                                                                                          : 4096);
        genConfig->setProperty("thinkingConfig", juce::var(thinkingConfig));
    }

    // Structured output via JSON schema
    if (!request.schema.isVoid()) {
        genConfig->setProperty("responseMimeType", "application/json");
        genConfig->setProperty("responseSchema", request.schema);
    }

    // Payload
    auto* payload = new juce::DynamicObject();
    payload->setProperty("system_instruction", juce::var(sysInstruction));
    payload->setProperty("contents", contentsArray);
    payload->setProperty("generationConfig", juce::var(genConfig));

    if (!request.tools.empty()) {
        juce::Array<juce::var> declarations;
        for (const auto& definition : request.tools) {
            auto* declaration = new juce::DynamicObject();
            declaration->setProperty("name", definition.name);
            declaration->setProperty("description", definition.description);
            declaration->setProperty("parameters", definition.inputSchema);
            declarations.add(juce::var(declaration));
        }
        auto* tool = new juce::DynamicObject();
        tool->setProperty("functionDeclarations", declarations);
        juce::Array<juce::var> tools;
        tools.add(juce::var(tool));
        payload->setProperty("tools", tools);

        auto* functionConfig = new juce::DynamicObject();
        switch (request.toolChoice.mode) {
            case ToolChoiceMode::None:
                functionConfig->setProperty("mode", "NONE");
                break;
            case ToolChoiceMode::Required:
                functionConfig->setProperty("mode", "ANY");
                break;
            case ToolChoiceMode::Specific: {
                functionConfig->setProperty("mode", "ANY");
                juce::Array<juce::var> names;
                names.add(request.toolChoice.toolName);
                functionConfig->setProperty("allowedFunctionNames", names);
                break;
            }
            case ToolChoiceMode::Auto:
                functionConfig->setProperty("mode", "AUTO");
                break;
        }
        auto* toolConfig = new juce::DynamicObject();
        toolConfig->setProperty("functionCallingConfig", juce::var(functionConfig));
        payload->setProperty("toolConfig", juce::var(toolConfig));
    }

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String GeminiClient::getEndpointUrl() const {
    return config_.baseUrl + "/v1beta/models/" + config_.model + ":generateContent";
}

juce::StringPairArray GeminiClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("Content-Type", "application/json");
    headers.set("x-goog-api-key", config_.apiKey);
    return headers;
}

Response GeminiClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    if (auto* candidates = json["candidates"].getArray()) {
        if (candidates->size() > 0) {
            if (auto* parts = (*candidates)[0]["content"]["parts"].getArray()) {
                const auto responseId = json["responseId"].toString();
                int callIndex = 0;
                for (const auto& part : *parts) {
                    if (!part["text"].isVoid())
                        response.text += part["text"].toString();
                    if (!part["functionCall"].isVoid()) {
                        const auto function = part["functionCall"];
                        ToolCall call;
                        call.id = function["id"].toString();
                        if (call.id.isEmpty())
                            call.id = (responseId.isNotEmpty() ? responseId : "gemini-call") + "-" +
                                      juce::String(callIndex);
                        call.name = function["name"].toString();
                        call.arguments = function["args"];
                        call.rawArguments = juce::JSON::toString(call.arguments);
                        if (!part["thoughtSignature"].isVoid()) {
                            auto* providerData = new juce::DynamicObject();
                            providerData->setProperty("thoughtSignature", part["thoughtSignature"]);
                            call.providerData = juce::var(providerData);
                        }
                        response.toolCalls.push_back(std::move(call));
                        ++callIndex;
                    }
                }
            }
        }
    }
    response.text = response.text.trim();
    response.success = response.text.isNotEmpty() || !response.toolCalls.empty();

    if (auto usage = json["usageMetadata"]; usage.isObject()) {
        response.inputTokens = static_cast<int>(usage["promptTokenCount"]);
        response.outputTokens = static_cast<int>(usage["candidatesTokenCount"]);
        response.totalTokens = static_cast<int>(usage["totalTokenCount"]);
    }

    if (!response.success)
        response.error = "Failed to parse response: " + jsonString.substring(0, 200);

    return response;
}

juce::String GeminiClient::getStreamingEndpointUrl() const {
    return config_.baseUrl + "/v1beta/models/" + config_.model + ":streamGenerateContent?alt=sse";
}

// Gemini streaming body is the same as non-streaming (no "stream":true needed)
juce::String GeminiClient::buildStreamingRequestBody(const Request& request) const {
    return buildRequestBody(request);
}

// Gemini SSE chunks use the same candidates/parts structure as non-streaming
juce::String GeminiClient::parseStreamChunk(const juce::String& dataLine) const {
    auto json = juce::JSON::parse(dataLine);
    if (auto* candidates = json["candidates"].getArray()) {
        if (candidates->size() > 0) {
            if (auto* parts = (*candidates)[0]["content"]["parts"].getArray()) {
                if (parts->size() > 0)
                    return (*parts)[0]["text"].toString();
            }
        }
    }
    return {};
}

std::vector<StreamDelta> GeminiClient::parseStreamDeltas(const juce::String& dataLine) const {
    std::vector<StreamDelta> result;
    auto json = juce::JSON::parse(dataLine);
    auto* candidates = json["candidates"].getArray();
    if (candidates == nullptr || candidates->isEmpty())
        return result;
    auto* parts = (*candidates)[0]["content"]["parts"].getArray();
    if (parts == nullptr)
        return result;

    const auto responseId = json["responseId"].toString();
    int callIndex = 0;
    for (const auto& part : *parts) {
        const auto text = part["text"].toString();
        if (text.isNotEmpty()) {
            StreamDelta delta;
            delta.type = StreamDeltaType::Text;
            delta.text = text;
            result.push_back(std::move(delta));
        }
        if (!part["functionCall"].isVoid()) {
            const auto function = part["functionCall"];
            StreamDelta start;
            start.type = StreamDeltaType::ToolCallStart;
            start.toolCallIndex = callIndex;
            start.callId = function["id"].toString();
            if (start.callId.isEmpty())
                start.callId = (responseId.isNotEmpty() ? responseId : "gemini-call") + "-" +
                               juce::String(callIndex);
            start.toolName = function["name"].toString();
            if (!part["thoughtSignature"].isVoid()) {
                auto* providerData = new juce::DynamicObject();
                providerData->setProperty("thoughtSignature", part["thoughtSignature"]);
                start.providerData = juce::var(providerData);
            }
            result.push_back(std::move(start));

            StreamDelta arguments;
            arguments.type = StreamDeltaType::ToolCallArguments;
            arguments.toolCallIndex = callIndex;
            arguments.argumentsDelta = juce::JSON::toString(function["args"]);
            result.push_back(std::move(arguments));
            ++callIndex;
        }
    }
    return result;
}

}  // namespace llm
