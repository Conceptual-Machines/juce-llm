namespace llm {

namespace {

juce::String stripCodexPrefix(const juce::String& model) {
    return model.startsWith("codex/") ? model.fromFirstOccurrenceOf("/", false, false) : model;
}

juce::String normalizeResponsesEndpoint(const juce::String& baseUrl) {
    return baseUrl.endsWith("/responses") ? baseUrl : baseUrl + "/responses";
}

}  // namespace

juce::String OpenAIResponsesClient::buildRequestBody(const Request& request) const {
    auto* payload = new juce::DynamicObject();
    payload->setProperty("model", config_.useCodexBackend ? stripCodexPrefix(config_.model)
                                                          : config_.model);

    if (config_.useCodexBackend) {
        if (request.systemPrompt.isNotEmpty())
            payload->setProperty("instructions", request.systemPrompt);

        auto* inputMessage = new juce::DynamicObject();
        inputMessage->setProperty("role", "user");

        auto* inputText = new juce::DynamicObject();
        inputText->setProperty("type", "input_text");
        inputText->setProperty("text", request.userMessage);

        juce::Array<juce::var> content;
        content.add(juce::var(inputText));
        inputMessage->setProperty("content", content);

        juce::Array<juce::var> input;
        input.add(juce::var(inputMessage));
        payload->setProperty("input", input);
        payload->setProperty("store", false);
    } else {
        payload->setProperty("instructions", request.systemPrompt);
        payload->setProperty("input", request.userMessage);
    }

    if (!config_.noTemperature)
        payload->setProperty("temperature", (double)request.temperature);

    if (config_.reasoningEffort.isNotEmpty()) {
        auto* reasoning = new juce::DynamicObject();
        reasoning->setProperty("effort", config_.reasoningEffort);
        payload->setProperty("reasoning", juce::var(reasoning));
    }

    // CFG grammar-constrained output via custom tool
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

        juce::Array<juce::var> tools;
        tools.add(juce::var(tool));
        payload->setProperty("tools", tools);
        payload->setProperty("parallel_tool_calls", false);
    }
    // Structured output via JSON schema
    else if (!request.schema.isVoid()) {
        auto* schemaWrapper = new juce::DynamicObject();
        schemaWrapper->setProperty("name", "response");
        schemaWrapper->setProperty("strict", true);
        schemaWrapper->setProperty("schema", request.schema);

        auto* text = new juce::DynamicObject();
        text->setProperty("type", "json_schema");
        text->setProperty("json_schema", juce::var(schemaWrapper));

        payload->setProperty("text", juce::var(text));
    }

    // Prompt caching is supported on the official Responses API, but Codex rejects it.
    if (!config_.useCodexBackend && config_.userAgent.isNotEmpty()) {
        payload->setProperty("prompt_cache_key", config_.userAgent);
        payload->setProperty("prompt_cache_retention", "24h");
    }

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String OpenAIResponsesClient::getEndpointUrl() const {
    return normalizeResponsesEndpoint(config_.baseUrl);
}

juce::StringPairArray OpenAIResponsesClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("Authorization", "Bearer " + config_.apiKey);
    headers.set("Content-Type", "application/json");
    if (config_.useCodexBackend) {
        headers.set("Accept", "text/event-stream");
        headers.set("OpenAI-Beta", "responses=experimental");
        headers.set("originator", "pi");
        if (config_.codexAccountId.isNotEmpty())
            headers.set("chatgpt-account-id", config_.codexAccountId);
    }
    return headers;
}

Response OpenAIResponsesClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    if (auto* output = json["output"].getArray()) {
        for (const auto& item : *output) {
            auto type = item["type"].toString();

            // CFG grammar tool response — extract from custom_tool_call
            if (type == "custom_tool_call") {
                auto input = item["input"].toString().trim();
                if (input.isNotEmpty()) {
                    response.text = input;
                    response.success = true;
                    return response;
                }
            }

            // Standard text response — output[].content[].text
            if (type == "message") {
                if (auto* content = item["content"].getArray()) {
                    for (const auto& c : *content) {
                        if (c["type"].toString() == "output_text") {
                            response.text = c["text"].toString().trim();
                            response.success = response.text.isNotEmpty();
                            return response;
                        }
                    }
                }
            }
        }
    }

    auto outputText = json["output_text"];
    if (outputText.isString() && outputText.toString().isNotEmpty()) {
        response.text = outputText.toString().trim();
        response.success = true;
        return response;
    }

    response.error = "Failed to parse response: " + jsonString.substring(0, 200);
    return response;
}

juce::String OpenAIResponsesClient::buildStreamingRequestBody(const Request& request) const {
    auto body = buildRequestBody(request);
    auto json = juce::JSON::parse(body);

    if (auto* obj = json.getDynamicObject()) {
        obj->setProperty("stream", true);
        if (config_.useCodexBackend)
            obj->setProperty("store", false);
        return juce::JSON::toString(json, true);
    }

    return LLMClient::buildStreamingRequestBody(request);
}

juce::String OpenAIResponsesClient::parseStreamChunk(const juce::String& dataLine) const {
    auto json = juce::JSON::parse(dataLine);

    if (config_.useCodexBackend) {
        auto type = json["type"].toString();
        if (type == "response.output_text.delta")
            return json["delta"].toString();
    }

    if (auto type = json["type"].toString(); type == "response.output_text.delta")
        return json["delta"].toString();

    return LLMClient::parseStreamChunk(dataLine);
}

}  // namespace llm
