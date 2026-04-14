namespace llm {

juce::String AnthropicClient::buildRequestBody(const Request& request) const {
    auto messagesArray = juce::Array<juce::var>();

    auto* userMsg = new juce::DynamicObject();
    userMsg->setProperty("role", "user");
    userMsg->setProperty("content", request.userMessage);
    messagesArray.add(juce::var(userMsg));

    auto* payload = new juce::DynamicObject();
    payload->setProperty("model", config_.model);
    int maxTok = request.maxTokens > 0 ? request.maxTokens
                                       : (config_.maxTokens > 0 ? config_.maxTokens : 4096);
    payload->setProperty("max_tokens", maxTok);
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

    // NOTE: Anthropic's Messages API does not accept an `effort` / `output_config`
    // field — that is an OpenAI-ism. `reasoningEffort` is deliberately ignored here.
    // (Extended thinking uses a separate `thinking` block on models that support it.)

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
        if (content->size() > 0) {
            response.text = (*content)[0]["text"].toString().trim();
            response.success = response.text.isNotEmpty();
        }
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

}  // namespace llm
