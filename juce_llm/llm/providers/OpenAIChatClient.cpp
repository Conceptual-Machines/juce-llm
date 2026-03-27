namespace llm
{

juce::var OpenAIChatClient::buildPayload (const Request& request) const
{
    auto* messages = new juce::DynamicObject();
    auto messagesArray = juce::Array<juce::var>();

    auto* sysMsg = new juce::DynamicObject();
    sysMsg->setProperty ("role", "system");
    sysMsg->setProperty ("content", request.systemPrompt);
    messagesArray.add (juce::var (sysMsg));

    auto* userMsg = new juce::DynamicObject();
    userMsg->setProperty ("role", "user");
    userMsg->setProperty ("content", request.userMessage);
    messagesArray.add (juce::var (userMsg));

    auto* payload = new juce::DynamicObject();
    payload->setProperty ("model", config_.model);
    payload->setProperty ("messages", messagesArray);
    payload->setProperty ("temperature", (double) request.temperature);

    if (config_.grammar.isNotEmpty())
        payload->setProperty ("grammar", config_.grammar);

    return juce::var (payload);
}

juce::URL OpenAIChatClient::buildUrl() const
{
    return juce::URL (config_.baseUrl + "/chat/completions");
}

juce::StringPairArray OpenAIChatClient::buildHeaders() const
{
    juce::StringPairArray headers;
    headers.set ("Authorization", "Bearer " + config_.apiKey);
    headers.set ("Content-Type", "application/json");
    return headers;
}

juce::String OpenAIChatClient::parseResponse (const juce::var& json) const
{
    if (auto* choices = json["choices"].getArray())
        if (choices->size() > 0)
            return (*choices)[0]["message"]["content"].toString().trim();

    return {};
}

} // namespace llm
