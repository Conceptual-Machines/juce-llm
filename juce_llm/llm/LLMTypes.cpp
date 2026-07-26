namespace llm {
namespace {
juce::var makeObject() {
    return juce::var(new juce::DynamicObject());
}

void set(juce::var& object, const juce::Identifier& key, const juce::var& value) {
    object.getDynamicObject()->setProperty(key, value);
}

}  // namespace

juce::var ToolDefinition::toVar() const {
    auto v = makeObject();
    set(v, "name", name);
    set(v, "description", description);
    set(v, "inputSchema", inputSchema);
    if (!annotations.isVoid())
        set(v, "annotations", annotations);
    return v;
}

ToolDefinition ToolDefinition::fromVar(const juce::var& v) {
    return {v["name"].toString(), v["description"].toString(), v["inputSchema"], v["annotations"]};
}

juce::var ToolCall::toVar() const {
    auto v = makeObject();
    set(v, "id", id);
    set(v, "name", name);
    set(v, "arguments", arguments);
    set(v, "rawArguments", rawArguments);
    set(v, "error", error);
    if (!providerData.isVoid())
        set(v, "providerData", providerData);
    return v;
}

ToolCall ToolCall::fromVar(const juce::var& v) {
    ToolCall call;
    call.id = v["id"].toString();
    call.name = v["name"].toString();
    call.arguments = v["arguments"];
    call.rawArguments = v["rawArguments"].toString();
    call.error = v["error"].toString();
    call.providerData = v["providerData"];
    return call;
}

juce::var ToolResult::toVar() const {
    auto v = makeObject();
    set(v, "callId", callId);
    set(v, "name", name);
    set(v, "content", content);
    set(v, "isError", isError);
    set(v, "error", error);
    return v;
}

ToolResult ToolResult::fromVar(const juce::var& v) {
    ToolResult result;
    result.callId = v["callId"].toString();
    result.name = v["name"].toString();
    result.content = v["content"];
    result.isError = static_cast<bool>(v["isError"]);
    result.error = v["error"].toString();
    return result;
}

juce::var Message::toVar() const {
    auto v = makeObject();
    set(v, "role", role);
    set(v, "content", content);

    if (!toolCalls.empty()) {
        juce::Array<juce::var> calls;
        for (const auto& call : toolCalls)
            calls.add(call.toVar());
        set(v, "toolCalls", calls);
    }
    if (!toolResults.empty()) {
        juce::Array<juce::var> results;
        for (const auto& result : toolResults)
            results.add(result.toVar());
        set(v, "toolResults", results);
    }
    return v;
}

Message Message::fromVar(const juce::var& v) {
    Message message;
    message.role = v["role"].toString();
    message.content = v["content"].toString();
    if (auto* calls = v["toolCalls"].getArray())
        for (const auto& call : *calls)
            message.toolCalls.push_back(ToolCall::fromVar(call));
    if (auto* results = v["toolResults"].getArray())
        for (const auto& result : *results)
            message.toolResults.push_back(ToolResult::fromVar(result));
    return message;
}

void Conversation::addToolResult(ToolResult result) {
    if (result.name.isEmpty()) {
        for (auto message = messages.rbegin(); message != messages.rend(); ++message) {
            auto call = std::find_if(
                message->toolCalls.begin(), message->toolCalls.end(),
                [&result](const ToolCall& candidate) { return candidate.id == result.callId; });
            if (call != message->toolCalls.end()) {
                result.name = call->name;
                break;
            }
        }
    }
    if (!messages.empty() && messages.back().role == "tool") {
        messages.back().toolResults.push_back(std::move(result));
        return;
    }
    Message turn;
    turn.role = "tool";
    turn.toolResults.push_back(std::move(result));
    messages.push_back(std::move(turn));
}

juce::var Conversation::toVar() const {
    auto v = makeObject();
    juce::Array<juce::var> turns;
    for (const auto& message : messages)
        turns.add(message.toVar());
    set(v, "messages", turns);
    set(v, "lastResponseId", lastResponseId);
    return v;
}

Conversation Conversation::fromVar(const juce::var& v) {
    Conversation conversation;
    if (auto* messages = v["messages"].getArray())
        for (const auto& message : *messages)
            conversation.messages.push_back(Message::fromVar(message));
    conversation.lastResponseId = v["lastResponseId"].toString();
    return conversation;
}

void StreamAccumulator::add(const StreamDelta& delta) {
    if (delta.type == StreamDeltaType::Text) {
        text_ += delta.text;
        return;
    }

    auto index = std::find(toolCallIndices_.begin(), toolCallIndices_.end(), delta.toolCallIndex);
    if (index == toolCallIndices_.end()) {
        toolCallIndices_.push_back(delta.toolCallIndex);
        toolCalls_.emplace_back();
        index = std::prev(toolCallIndices_.end());
    }
    auto& call = toolCalls_[static_cast<size_t>(std::distance(toolCallIndices_.begin(), index))];
    if (delta.callId.isNotEmpty())
        call.id = delta.callId;
    if (delta.toolName.isNotEmpty())
        call.name = delta.toolName;
    if (delta.argumentsDelta.isNotEmpty())
        call.rawArguments += delta.argumentsDelta;
    if (!delta.providerData.isVoid())
        call.providerData = delta.providerData;
}

Response StreamAccumulator::finish() const {
    Response response;
    response.text = text_.trim();
    response.toolCalls = toolCalls_;

    for (auto& call : response.toolCalls) {
        if (call.rawArguments.isEmpty())
            call.rawArguments = "{}";
        auto parsed = juce::JSON::parse(call.rawArguments);
        if (parsed.isVoid()) {
            call.error = "Malformed tool arguments: " + call.rawArguments.substring(0, 200);
        } else {
            call.arguments = parsed;
        }
    }

    response.success = response.text.isNotEmpty() || !response.toolCalls.empty();
    return response;
}

}  // namespace llm
