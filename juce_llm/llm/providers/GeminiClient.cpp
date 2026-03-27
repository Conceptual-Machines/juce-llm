namespace llm {

juce::String GeminiClient::buildRequestBody(const Request& request) const {
    // System instruction
    auto* sysPart = new juce::DynamicObject();
    sysPart->setProperty("text", request.systemPrompt);

    auto sysPartsArray = juce::Array<juce::var>();
    sysPartsArray.add(juce::var(sysPart));

    auto* sysInstruction = new juce::DynamicObject();
    sysInstruction->setProperty("parts", sysPartsArray);

    // User content
    auto* userPart = new juce::DynamicObject();
    userPart->setProperty("text", request.userMessage);

    auto userPartsArray = juce::Array<juce::var>();
    userPartsArray.add(juce::var(userPart));

    auto* userContent = new juce::DynamicObject();
    userContent->setProperty("parts", userPartsArray);

    auto contentsArray = juce::Array<juce::var>();
    contentsArray.add(juce::var(userContent));

    // Generation config
    auto* genConfig = new juce::DynamicObject();
    genConfig->setProperty("temperature", (double)request.temperature);

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

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String GeminiClient::getEndpointUrl() const {
    return config_.baseUrl + "/v1beta/models/" + config_.model +
           ":generateContent?key=" + config_.apiKey;
}

juce::StringPairArray GeminiClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("Content-Type", "application/json");
    return headers;
}

Response GeminiClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    if (auto* candidates = json["candidates"].getArray()) {
        if (candidates->size() > 0) {
            if (auto* parts = (*candidates)[0]["content"]["parts"].getArray()) {
                if (parts->size() > 0) {
                    response.text = (*parts)[0]["text"].toString().trim();
                    response.success = response.text.isNotEmpty();
                }
            }
        }
    }

    if (!response.success)
        response.error = "Failed to parse response: " + jsonString.substring(0, 200);

    return response;
}

}  // namespace llm
