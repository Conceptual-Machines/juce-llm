namespace llm
{

juce::var AnthropicClient::buildPayload (const Request& request) const
{
    auto messagesArray = juce::Array<juce::var>();

    auto* userMsg = new juce::DynamicObject();
    userMsg->setProperty ("role", "user");
    userMsg->setProperty ("content", request.userMessage);
    messagesArray.add (juce::var (userMsg));

    auto* payload = new juce::DynamicObject();
    payload->setProperty ("model", config_.model);
    payload->setProperty ("max_tokens", 256);
    payload->setProperty ("temperature", (double) request.temperature);
    payload->setProperty ("system", request.systemPrompt);
    payload->setProperty ("messages", messagesArray);

    return juce::var (payload);
}

juce::URL AnthropicClient::buildUrl() const
{
    return juce::URL (config_.baseUrl + "/messages");
}

juce::StringPairArray AnthropicClient::buildHeaders() const
{
    juce::StringPairArray headers;
    headers.set ("x-api-key", config_.apiKey);
    headers.set ("anthropic-version", "2023-06-01");
    headers.set ("Content-Type", "application/json");
    return headers;
}

juce::String AnthropicClient::parseResponse (const juce::var& json) const
{
    if (auto* content = json["content"].getArray())
        if (content->size() > 0)
            return (*content)[0]["text"].toString().trim();

    return {};
}

} // namespace llm
