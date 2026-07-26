#include <juce_core/juce_core.h>
#include <juce_llm/juce_llm.h>

//==============================================================================
/** Test helpers */
static int passed_ = 0;
static int failed_ = 0;
static int skipped_ = 0;

static void check(const juce::String& name, bool condition) {
    std::cout << "  " << name << " ... " << std::flush;
    if (condition) {
        passed_++;
        std::cout << "PASS" << std::endl;
    } else {
        failed_++;
        std::cout << "FAIL" << std::endl;
    }
}

static void skip(const juce::String& name, const juce::String& reason = "no API key") {
    skipped_++;
    std::cout << "  " << name << " ... SKIP (" << reason << ")" << std::endl;
}

static void printHeader(const juce::String& text) {
    std::cout << std::endl
              << std::string(60, '=') << std::endl
              << "  " << text << std::endl
              << std::string(60, '=') << std::endl;
}

//==============================================================================
// Offline tests — no API keys or network required
//==============================================================================

void testSchemaBuilder() {
    printHeader("Schema builder");

    // String schema
    auto s = llm::Schema::string();
    auto json = juce::JSON::parse(juce::JSON::toString(s));
    check("Schema::string()", json["type"].toString() == "string");

    // Number schema
    auto n = llm::Schema::number();
    json = juce::JSON::parse(juce::JSON::toString(n));
    check("Schema::number()", json["type"].toString() == "number");

    // Integer schema
    auto i = llm::Schema::integer();
    json = juce::JSON::parse(juce::JSON::toString(i));
    check("Schema::integer()", json["type"].toString() == "integer");

    // Boolean schema
    auto b = llm::Schema::boolean();
    json = juce::JSON::parse(juce::JSON::toString(b));
    check("Schema::boolean()", json["type"].toString() == "boolean");

    // Array schema
    auto a = llm::Schema::array(llm::Schema::string());
    json = juce::JSON::parse(juce::JSON::toString(a));
    check("Schema::array(string)",
          json["type"].toString() == "array" && json["items"]["type"].toString() == "string");

    // Object schema
    auto obj = llm::Schema::object({
        {"name", llm::Schema::string()},
        {"age", llm::Schema::integer()},
    });
    json = juce::JSON::parse(juce::JSON::toString(obj));
    check("Schema::object() type", json["type"].toString() == "object");
    check("Schema::object() properties",
          json["properties"]["name"]["type"].toString() == "string" &&
              json["properties"]["age"]["type"].toString() == "integer");
    check("Schema::object() required", json["required"].getArray()->size() == 2);
    check("Schema::object() additionalProperties", !(bool)json["additionalProperties"]);

    // Enum schema
    auto e = llm::Schema::oneOf({"major", "minor", "diminished"});
    json = juce::JSON::parse(juce::JSON::toString(e));
    check("Schema::oneOf()",
          json["type"].toString() == "string" && json["enum"].getArray()->size() == 3);
}

void testOpenAIChatPayload() {
    printHeader("OpenAI Chat payload");

    llm::ProviderConfig config;
    config.provider = llm::Provider::OpenAIChat;
    config.baseUrl = "https://api.openai.com/v1";
    config.apiKey = "test-key";
    config.model = "gpt-4o-mini";

    auto client = llm::LLMClientFactory::create(config);

    llm::Request request;
    request.systemPrompt = "You are helpful.";
    request.userMessage = "Hello";
    request.temperature = 0.5f;

    auto body = juce::JSON::parse(client->buildRequestBody(request));
    check("model", body["model"].toString() == "gpt-4o-mini");
    check("messages array", body["messages"].getArray()->size() == 2);
    check("system role", (*body["messages"].getArray())[0]["role"].toString() == "system");
    check("user role", (*body["messages"].getArray())[1]["role"].toString() == "user");
    check("temperature", (double)body["temperature"] > 0.4);

    check("endpoint", client->getEndpointUrl() == "https://api.openai.com/v1/chat/completions");

    auto headers = client->getHeaders();
    check("auth header", headers["Authorization"] == "Bearer test-key");
    check("content-type", headers["Content-Type"] == "application/json");

    // With schema
    request.schema = llm::Schema::object({{"answer", llm::Schema::string()}});
    body = juce::JSON::parse(client->buildRequestBody(request));
    check("response_format present", !body["response_format"].isVoid());
    check("response_format type", body["response_format"]["type"].toString() == "json_schema");
}

void testOpenAIChatResponseParsing() {
    printHeader("OpenAI Chat response parsing");

    llm::ProviderConfig config;
    config.provider = llm::Provider::OpenAIChat;
    config.baseUrl = "https://api.openai.com/v1";
    config.model = "test";

    auto client = llm::LLMClientFactory::create(config);

    // Valid response
    auto response =
        client->parseResponseBody(R"({"choices":[{"message":{"content":"Hello world"}}]})");
    check("valid response text", response.text == "Hello world");
    check("valid response success", response.success);

    // Empty response
    response = client->parseResponseBody(R"({"choices":[]})");
    check("empty choices fails", !response.success);
    check("empty choices has error", response.error.isNotEmpty());

    // Malformed JSON
    response = client->parseResponseBody("not json");
    check("malformed JSON fails", !response.success);
}

void testOpenAIResponsesPayload() {
    printHeader("OpenAI Responses payload");

    llm::ProviderConfig config;
    config.provider = llm::Provider::OpenAIResponses;
    config.baseUrl = "https://api.openai.com/v1";
    config.apiKey = "test-key";
    config.model = "gpt-5";
    config.noTemperature = true;
    config.reasoningEffort = "minimal";

    auto client = llm::LLMClientFactory::create(config);

    llm::Request request;
    request.systemPrompt = "Be concise.";
    request.userMessage = "Hi";

    auto body = juce::JSON::parse(client->buildRequestBody(request));
    check("model", body["model"].toString() == "gpt-5");
    check("instructions", body["instructions"].toString() == "Be concise.");
    check("input", body["input"].toString() == "Hi");
    check("no temperature", body["temperature"].isVoid());
    check("reasoning effort", body["reasoning"]["effort"].toString() == "minimal");

    check("endpoint", client->getEndpointUrl() == "https://api.openai.com/v1/responses");
}

void testOpenAIResponsesParsing() {
    printHeader("OpenAI Responses response parsing");

    llm::ProviderConfig config;
    config.provider = llm::Provider::OpenAIResponses;
    config.baseUrl = "https://api.openai.com/v1";
    config.model = "test";

    auto client = llm::LLMClientFactory::create(config);

    auto response = client->parseResponseBody(
        R"({"output":[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}]})");
    check("valid response text", response.text == "Hello");
    check("valid response success", response.success);
}

void testAnthropicPayload() {
    printHeader("Anthropic payload");

    llm::ProviderConfig config;
    config.provider = llm::Provider::Anthropic;
    config.baseUrl = "https://api.anthropic.com/v1";
    config.apiKey = "test-key";
    config.model = "claude-haiku-4-5-20251001";

    auto client = llm::LLMClientFactory::create(config);

    llm::Request request;
    request.systemPrompt = "Be helpful.";
    request.userMessage = "Hi";

    auto body = juce::JSON::parse(client->buildRequestBody(request));
    check("model", body["model"].toString() == "claude-haiku-4-5-20251001");
    check("system", body["system"].getArray() != nullptr && body["system"].getArray()->size() == 1);
    check("messages", body["messages"].getArray()->size() == 1);
    check("max_tokens", (int)body["max_tokens"] == 4096);

    check("endpoint", client->getEndpointUrl() == "https://api.anthropic.com/v1/messages");

    auto headers = client->getHeaders();
    check("x-api-key", headers["x-api-key"] == "test-key");
    check("anthropic-version", headers["anthropic-version"] == "2023-06-01");
}

void testAnthropicParsing() {
    printHeader("Anthropic response parsing");

    llm::ProviderConfig config;
    config.provider = llm::Provider::Anthropic;
    config.baseUrl = "https://api.anthropic.com/v1";
    config.model = "test";

    auto client = llm::LLMClientFactory::create(config);

    auto response = client->parseResponseBody(R"({"content":[{"type":"text","text":"Hello"}]})");
    check("valid response text", response.text == "Hello");
    check("valid response success", response.success);
}

void testGeminiPayload() {
    printHeader("Gemini payload");

    llm::ProviderConfig config;
    config.provider = llm::Provider::Gemini;
    config.baseUrl = "https://generativelanguage.googleapis.com";
    config.apiKey = "test-key";
    config.model = "gemini-2.5-flash";

    auto client = llm::LLMClientFactory::create(config);

    llm::Request request;
    request.systemPrompt = "Be helpful.";
    request.userMessage = "Hi";

    auto body = juce::JSON::parse(client->buildRequestBody(request));
    check("system_instruction", body["system_instruction"]["parts"].getArray()->size() == 1);
    check("contents", body["contents"].getArray()->size() == 1);
    check("generationConfig", !body["generationConfig"].isVoid());

    check("endpoint contains model", client->getEndpointUrl().contains("gemini-2.5-flash"));
    check("api key in header", client->getHeaders()["x-goog-api-key"] == "test-key");

    // With schema
    request.schema = llm::Schema::object({{"answer", llm::Schema::string()}});
    body = juce::JSON::parse(client->buildRequestBody(request));
    check("responseMimeType",
          body["generationConfig"]["responseMimeType"].toString() == "application/json");
    check("responseSchema present", !body["generationConfig"]["responseSchema"].isVoid());
}

void testGeminiParsing() {
    printHeader("Gemini response parsing");

    llm::ProviderConfig config;
    config.provider = llm::Provider::Gemini;
    config.baseUrl = "https://generativelanguage.googleapis.com";
    config.model = "test";

    auto client = llm::LLMClientFactory::create(config);

    auto response =
        client->parseResponseBody(R"({"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]})");
    check("valid response text", response.text == "Hello");
    check("valid response success", response.success);
}

void testFactory() {
    printHeader("Client factory");

    auto makeConfig = [](llm::Provider p) {
        llm::ProviderConfig c;
        c.provider = p;
        c.baseUrl = "http://test";
        c.model = "test";
        return c;
    };

    check("creates OpenAIChat",
          llm::LLMClientFactory::create(makeConfig(llm::Provider::OpenAIChat))->getName() ==
              "OpenAI Chat");
    check("creates OpenAIResponses",
          llm::LLMClientFactory::create(makeConfig(llm::Provider::OpenAIResponses))->getName() ==
              "OpenAI Responses");
    check("creates Anthropic",
          llm::LLMClientFactory::create(makeConfig(llm::Provider::Anthropic))->getName() ==
              "Anthropic");
    check("creates Gemini",
          llm::LLMClientFactory::create(makeConfig(llm::Provider::Gemini))->getName() == "Gemini");
}

llm::ProviderConfig toolTestConfig(llm::Provider provider) {
    llm::ProviderConfig config;
    config.provider = provider;
    config.baseUrl = provider == llm::Provider::Gemini ? "https://generativelanguage.googleapis.com"
                                                       : "https://example.test/v1";
    config.model = "test";
    return config;
}

llm::ToolDefinition toolTestDefinition() {
    llm::ToolDefinition tool;
    tool.name = "get_weather";
    tool.description = "Get weather for a city";
    tool.inputSchema = llm::Schema::object({{"city", llm::Schema::string()}});
    return tool;
}

void testToolCalls() {
    printHeader("Provider-neutral tool calls");

    auto definition = toolTestDefinition();
    definition.annotations = juce::JSON::parse(R"({"readOnlyHint":true})");
    auto restoredDefinition = llm::ToolDefinition::fromVar(definition.toVar());
    check("tool definition annotations round-trip",
          static_cast<bool>(restoredDefinition.annotations["readOnlyHint"]));

    llm::ToolCall call;
    call.id = "call-1";
    call.name = "get_weather";
    call.rawArguments = R"({"city":"London"})";
    call.arguments = juce::JSON::parse(call.rawArguments);

    llm::Conversation conversation;
    llm::Message assistant("assistant", "Checking");
    assistant.toolCalls.push_back(call);
    conversation.messages.push_back(std::move(assistant));
    conversation.addToolResult({"call-1", {}, juce::JSON::parse(R"({"temperature":18})")});
    conversation.lastResponseId = "resp-1";

    auto restored = llm::Conversation::fromVar(conversation.toVar());
    check("conversation call round-trip",
          restored.messages[0].toolCalls[0].arguments["city"].toString() == "London");
    check("conversation resolves result name",
          restored.messages[1].toolResults[0].name == "get_weather");
    check("conversation response id round-trip", restored.lastResponseId == "resp-1");

    {
        auto client = llm::LLMClientFactory::create(toolTestConfig(llm::Provider::OpenAIChat));
        llm::Request request;
        request.tools = {toolTestDefinition()};
        request.toolChoice = {llm::ToolChoiceMode::Specific, "get_weather"};
        request.messages = restored.messages;
        auto body = juce::JSON::parse(client->buildRequestBody(request));
        check("OpenAI Chat tool definition",
              body["tools"][0]["function"]["name"].toString() == "get_weather");
        check("OpenAI Chat result mapping",
              body["messages"][2]["tool_call_id"].toString() == "call-1");

        auto response = client->parseResponseBody(
            R"({"choices":[{"message":{"content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\"}"}}]}}]})");
        check("OpenAI Chat parses call", response.success && response.toolCalls.size() == 1);

        llm::StreamAccumulator stream;
        for (
            const auto& chunk :
            {juce::String(
                 R"({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]})"),
             juce::String(
                 R"({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"London\"}"}}]}}]})")})
            for (const auto& delta : client->parseStreamDeltas(chunk))
                stream.add(delta);
        response = stream.finish();
        check("OpenAI Chat assembles streamed arguments",
              response.toolCalls[0].arguments["city"].toString() == "London");
    }

    {
        auto client = llm::LLMClientFactory::create(toolTestConfig(llm::Provider::OpenAIResponses));
        llm::Request request;
        request.grammar = "start: /.+/";
        request.tools = {toolTestDefinition()};
        auto body = juce::JSON::parse(client->buildRequestBody(request));
        check("Responses keeps CFG custom tool distinct",
              body["tools"][0]["type"].toString() == "custom" &&
                  body["tools"][1]["type"].toString() == "function");

        auto response = client->parseResponseBody(
            R"({"id":"resp-1","output":[{"type":"custom_tool_call","input":"dsl text"},{"type":"function_call","call_id":"call-1","name":"get_weather","arguments":"{\"city\":\"London\"}"}]})");
        check("Responses parses text and executable call",
              response.text == "dsl text" && response.toolCalls.size() == 1);
    }

    {
        auto client = llm::LLMClientFactory::create(toolTestConfig(llm::Provider::Anthropic));
        llm::Request request;
        request.tools = {toolTestDefinition()};
        request.messages = restored.messages;
        auto body = juce::JSON::parse(client->buildRequestBody(request));
        check("Anthropic tool definition",
              body["tools"][0]["input_schema"]["type"].toString() == "object");
        check("Anthropic result block",
              body["messages"][1]["content"][0]["type"].toString() == "tool_result");

        auto response = client->parseResponseBody(
            R"({"content":[{"type":"text","text":"Checking"},{"type":"tool_use","id":"call-1","name":"get_weather","input":{"city":"London"}}]})");
        check("Anthropic parses mixed blocks",
              response.text == "Checking" && response.toolCalls.size() == 1);
    }

    {
        auto client = llm::LLMClientFactory::create(toolTestConfig(llm::Provider::Gemini));
        llm::Request request;
        request.tools = {toolTestDefinition()};
        request.messages = restored.messages;
        auto body = juce::JSON::parse(client->buildRequestBody(request));
        check("Gemini function declaration",
              body["tools"][0]["functionDeclarations"][0]["name"].toString() == "get_weather");
        check("Gemini function response id",
              body["contents"][1]["parts"][0]["functionResponse"]["id"].toString() == "call-1");

        auto response = client->parseResponseBody(
            R"({"responseId":"gem-resp","candidates":[{"content":{"parts":[{"functionCall":{"id":"call-1","name":"get_weather","args":{"city":"London"}},"thoughtSignature":"opaque"}]}}]})");
        check("Gemini parses call id", response.toolCalls[0].id == "call-1");
        check("Gemini preserves thought signature",
              response.toolCalls[0].providerData["thoughtSignature"].toString() == "opaque");
    }

    llm::StreamAccumulator malformed;
    malformed.add({llm::StreamDeltaType::ToolCallStart, 0, {}, "bad", "broken", {}, {}});
    malformed.add({llm::StreamDeltaType::ToolCallArguments, 0, {}, {}, {}, "{bad", {}});
    auto malformedResponse = malformed.finish();
    check("malformed arguments are structured failure",
          malformedResponse.success && !malformedResponse.toolCalls[0].isValid() &&
              malformedResponse.text.isEmpty());
}

//==============================================================================
// Live tests — require API keys
//==============================================================================

void runLiveTest(const juce::String& name, const llm::ProviderConfig& config) {
    std::cout << "  " << name << " ... " << std::flush;

    auto client = llm::LLMClientFactory::create(config);

    llm::Request request;
    request.systemPrompt = "Respond with exactly one word: HELLO";
    request.userMessage = "Say hello.";

    auto response = client->sendRequest(request);

    if (response.success && response.text.containsIgnoreCase("hello")) {
        passed_++;
        std::cout << "PASS (" << response.wallSeconds << "s)" << std::endl;
    } else {
        failed_++;
        std::cout << "FAIL";
        if (response.error.isNotEmpty())
            std::cout << " — " << response.error.substring(0, 100);
        else
            std::cout << " — got: " << response.text.substring(0, 50);
        std::cout << std::endl;
    }
}

void testLiveProviders() {
    printHeader("Live provider tests");

    // llama-server
    {
        llm::ProviderConfig config;
        config.provider = llm::Provider::OpenAIChat;
        config.baseUrl = "http://localhost:8080/v1";
        config.model = "local";

        auto client = llm::LLMClientFactory::create(config);
        llm::Request request;
        request.systemPrompt = "Respond with exactly one word: HELLO";
        request.userMessage = "Say hello.";

        std::cout << "  llama-server (local) ... " << std::flush;
        auto response = client->sendRequest(request);

        if (response.error.contains("Failed to connect")) {
            skip("llama-server (local)", "server not running");
        } else if (response.success && response.text.containsIgnoreCase("hello")) {
            passed_++;
            std::cout << "PASS (" << response.wallSeconds << "s)" << std::endl;
        } else {
            failed_++;
            std::cout << "FAIL" << std::endl;
        }
    }

    // OpenAI Chat
    auto openaiKey = juce::SystemStats::getEnvironmentVariable("OPENAI_API_KEY", "");
    if (openaiKey.isNotEmpty()) {
        llm::ProviderConfig config;
        config.provider = llm::Provider::OpenAIChat;
        config.baseUrl = "https://api.openai.com/v1";
        config.apiKey = openaiKey;
        config.model = "gpt-4o-mini";
        runLiveTest("OpenAI Chat (gpt-4o-mini)", config);

        config.provider = llm::Provider::OpenAIResponses;
        config.model = "gpt-5";
        config.noTemperature = true;
        config.reasoningEffort = "minimal";
        runLiveTest("OpenAI Responses (gpt-5)", config);
    } else {
        skip("OpenAI Chat");
        skip("OpenAI Responses");
    }

    // Anthropic
    auto anthropicKey = juce::SystemStats::getEnvironmentVariable("ANTHROPIC_API_KEY", "");
    if (anthropicKey.isNotEmpty()) {
        llm::ProviderConfig config;
        config.provider = llm::Provider::Anthropic;
        config.baseUrl = "https://api.anthropic.com/v1";
        config.apiKey = anthropicKey;
        config.model = "claude-haiku-4-5-20251001";
        runLiveTest("Anthropic (claude-haiku)", config);
    } else {
        skip("Anthropic");
    }

    // Gemini
    auto geminiKey = juce::SystemStats::getEnvironmentVariable("GEMINI_API_KEY", "");
    if (geminiKey.isNotEmpty()) {
        llm::ProviderConfig config;
        config.provider = llm::Provider::Gemini;
        config.baseUrl = "https://generativelanguage.googleapis.com";
        config.apiKey = geminiKey;
        config.model = "gemini-2.5-flash";
        runLiveTest("Gemini (2.5-flash)", config);
    } else {
        skip("Gemini");
    }

    // OpenRouter
    auto orKey = juce::SystemStats::getEnvironmentVariable("OPENROUTER_API_KEY", "");
    if (orKey.isNotEmpty()) {
        llm::ProviderConfig config;
        config.provider = llm::Provider::OpenAIChat;
        config.baseUrl = "https://openrouter.ai/api/v1";
        config.apiKey = orKey;
        config.model = "meta-llama/llama-3.3-70b-instruct";
        runLiveTest("OpenRouter (llama-3.3-70b)", config);
    } else {
        skip("OpenRouter");
    }
}

//==============================================================================
int main() {
    // Offline tests (always run)
    testSchemaBuilder();
    testFactory();
    testOpenAIChatPayload();
    testOpenAIChatResponseParsing();
    testOpenAIResponsesPayload();
    testOpenAIResponsesParsing();
    testAnthropicPayload();
    testAnthropicParsing();
    testGeminiPayload();
    testGeminiParsing();
    testToolCalls();

    // Live tests (skip if no keys)
    testLiveProviders();

    printHeader("Results: " + juce::String(passed_) + " passed, " + juce::String(failed_) +
                " failed, " + juce::String(skipped_) + " skipped");

    return failed_ > 0 ? 1 : 0;
}
