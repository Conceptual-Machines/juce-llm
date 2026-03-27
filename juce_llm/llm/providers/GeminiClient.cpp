namespace llm
{

juce::var GeminiClient::buildPayload (const Request& request) const
{
    // System instruction
    auto* sysPart = new juce::DynamicObject();
    sysPart->setProperty ("text", request.systemPrompt);

    auto sysPartsArray = juce::Array<juce::var>();
    sysPartsArray.add (juce::var (sysPart));

    auto* sysInstruction = new juce::DynamicObject();
    sysInstruction->setProperty ("parts", sysPartsArray);

    // User content
    auto* userPart = new juce::DynamicObject();
    userPart->setProperty ("text", request.userMessage);

    auto userPartsArray = juce::Array<juce::var>();
    userPartsArray.add (juce::var (userPart));

    auto* userContent = new juce::DynamicObject();
    userContent->setProperty ("parts", userPartsArray);

    auto contentsArray = juce::Array<juce::var>();
    contentsArray.add (juce::var (userContent));

    // Generation config
    auto* genConfig = new juce::DynamicObject();
    genConfig->setProperty ("temperature", (double) request.temperature);

    // Payload
    auto* payload = new juce::DynamicObject();
    payload->setProperty ("system_instruction", juce::var (sysInstruction));
    payload->setProperty ("contents", contentsArray);
    payload->setProperty ("generationConfig", juce::var (genConfig));

    return juce::var (payload);
}

juce::URL GeminiClient::buildUrl() const
{
    return juce::URL (config_.baseUrl
        + "/v1beta/models/" + config_.model
        + ":generateContent?key=" + config_.apiKey);
}

juce::StringPairArray GeminiClient::buildHeaders() const
{
    juce::StringPairArray headers;
    headers.set ("Content-Type", "application/json");
    return headers;
}

juce::String GeminiClient::parseResponse (const juce::var& json) const
{
    if (auto* candidates = json["candidates"].getArray())
        if (candidates->size() > 0)
            if (auto* parts = (*candidates)[0]["content"]["parts"].getArray())
                if (parts->size() > 0)
                    return (*parts)[0]["text"].toString().trim();

    return {};
}

} // namespace llm
