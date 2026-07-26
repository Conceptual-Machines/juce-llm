#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace llm {

//==============================================================================
/** An executable provider-neutral function definition. `annotations` is an
    optional object for hints such as readOnlyHint/idempotentHint. Providers
    that do not support annotations ignore it. */
struct ToolDefinition {
    ToolDefinition() = default;
    ToolDefinition(juce::String nameIn, juce::String descriptionIn, juce::var inputSchemaIn,
                   juce::var annotationsIn = {})
        : name(std::move(nameIn)),
          description(std::move(descriptionIn)),
          inputSchema(std::move(inputSchemaIn)),
          annotations(std::move(annotationsIn)) {}

    juce::String name;
    juce::String description;
    juce::var inputSchema;
    juce::var annotations;

    juce::var toVar() const;
    static ToolDefinition fromVar(const juce::var&);
};

/** One function call requested by an assistant response. `rawArguments`
    preserves the exact streamed/provider payload. If it is malformed JSON,
    `arguments` remains void and `error` describes the structured failure. */
struct ToolCall {
    juce::String id;
    juce::String name;
    juce::var arguments;
    juce::String rawArguments;
    juce::String error;
    juce::var providerData;

    bool isValid() const {
        return error.isEmpty();
    }

    juce::var toVar() const;
    static ToolCall fromVar(const juce::var&);
};

/** Result returned to the model for a prior tool call. `content` may be a
    string, object, array, number, or boolean. */
struct ToolResult {
    ToolResult() = default;
    ToolResult(juce::String callIdIn, juce::String nameIn, juce::var contentIn,
               bool isErrorIn = false, juce::String errorIn = {})
        : callId(std::move(callIdIn)),
          name(std::move(nameIn)),
          content(std::move(contentIn)),
          isError(isErrorIn),
          error(std::move(errorIn)) {}

    juce::String callId;
    juce::String name;
    juce::var content;
    bool isError = false;
    juce::String error;

    juce::var toVar() const;
    static ToolResult fromVar(const juce::var&);
};

enum class ToolChoiceMode { Auto, None, Required, Specific };

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    juce::String toolName;
};

//==============================================================================
/** A prior conversation turn. Assistant turns may contain text and/or tool
    calls; tool turns contain one or more results. */
struct Message {
    Message() = default;
    Message(juce::String roleIn, juce::String contentIn)
        : role(std::move(roleIn)), content(std::move(contentIn)) {}

    juce::String role;
    juce::String content;
    std::vector<ToolCall> toolCalls;
    std::vector<ToolResult> toolResults;

    juce::var toVar() const;
    static Message fromVar(const juce::var&);
};

//==============================================================================
enum class Provider {
    OpenAIChat,       // Chat Completions — DeepSeek, OpenRouter, local llama-server
    OpenAIResponses,  // Responses API — GPT-5+
    Anthropic,        // Messages API — Claude models
    Gemini            // generateContent — Gemini models
};

//==============================================================================
struct ProviderConfig {
    Provider provider;
    juce::String baseUrl;
    juce::String apiKey;
    juce::String model;

    int maxTokens = 0;  // max output tokens (0 = provider default — don't send a cap)

    // Provider-specific options
    bool noTemperature = false;    // GPT-5 doesn't support temperature
    juce::String reasoningEffort;  // "none", "low", "medium", "high", "xhigh"
    juce::String grammar;          // GBNF grammar for llama-server
    int connectionTimeoutMs = 0;   // 0 = system default; useful for local server checks

    // Application identity — used for User-Agent and provider-specific headers
    juce::String userAgent;  // e.g. "MAGDA/0.3.0"
    juce::String appUrl;     // e.g. "https://magda.dev" (for OpenRouter HTTP-Referer)
};

//==============================================================================
struct Request {
    juce::String systemPrompt;
    juce::String userMessage;
    float temperature = 0.1f;

    /** Optional JSON schema for structured output.
        Built via Schema::object(), Schema::array(), etc.
        When set, providers will use their native structured output mechanism. */
    juce::var schema;

    /** Optional CFG grammar for constrained output (Lark format for OpenAI Responses API,
        GBNF for llama-server). When set, the provider will constrain the model output to match
        the grammar. For OpenAI Responses, this uses a custom tool with grammar format. */
    juce::String grammar;

    /** Tool name used for CFG grammar-constrained output (OpenAI Responses API).
        Defaults to "grammar_tool" if not set. */
    juce::String grammarToolName;

    /** Tool description for CFG grammar output. If empty, systemPrompt is used. */
    juce::String grammarToolDescription;

    /** Override max output tokens for this request (0 = use provider config default). */
    int maxTokens = 0;

    /** When true, sendStreamingRequest will add provider-specific streaming flags. */
    bool stream = false;

    /** Executable function tools. These are independent of `grammar`, which
        remains constrained model output and is never surfaced as a ToolCall. */
    std::vector<ToolDefinition> tools;
    ToolChoice toolChoice;

    /** Prior conversation turns for stateless providers (Anthropic / OpenAI Chat /
        Gemini). Emitted before `userMessage`, which is the current user turn.
        OpenAI Responses chains via `previousResponseId` and reads only pending
        tool-result turns after the last assistant response. */
    std::vector<Message> messages;

    /** For the OpenAI Responses API only: the `id` of the previous response.
        When set, the server retains prior context (including reasoning items),
        so only `userMessage` is sent as the new input. Ignored by stateless
        providers. */
    juce::String previousResponseId;
};

//==============================================================================
struct Response {
    juce::String text;
    std::vector<ToolCall> toolCalls;
    double wallSeconds = 0.0;
    bool success = false;
    juce::String error;

    /** Provider response id. Used by the OpenAI Responses API to chain the next
        turn (passed back as `previousResponseId`). Empty for providers that
        don't expose one or where it isn't parsed. */
    juce::String id;

    /** Token usage reported by the provider, -1 when it wasn't reported (or
        parsing failed). totalTokens is whatever the provider returned, or
        input+output when it only reports the two. Useful for context-budget
        and cost accounting. */
    int inputTokens = -1;
    int outputTokens = -1;
    int totalTokens = -1;
};

//==============================================================================
/** A running multi-turn conversation. Holds the turn history (for stateless
    providers) and the last response id (for the stateful Responses API). The
    client picks whichever the active provider needs; callers just keep one of
    these and pass it to LLMClient::continueConversation. Serialise via
    toVar / fromVar to persist it across UI rebuilds. */
struct Conversation {
    std::vector<Message> messages;  // user / assistant / tool turns
    juce::String lastResponseId;    // OpenAI Responses chaining

    void clear() {
        messages.clear();
        lastResponseId = {};
    }

    void addToolResult(ToolResult result);
    juce::var toVar() const;
    static Conversation fromVar(const juce::var&);
};

//==============================================================================
using ResponseCallback = std::function<void(Response)>;

/** Called for each token/chunk during streaming. Return false to cancel. */
using StreamCallback = std::function<bool(const juce::String& token)>;

enum class StreamDeltaType { Text, ToolCallStart, ToolCallArguments, ToolCallEnd };

/** Provider-neutral streaming event. `toolCallIndex` correlates interleaved
    parallel calls; argument fragments are concatenated in arrival order. */
struct StreamDelta {
    StreamDeltaType type = StreamDeltaType::Text;
    int toolCallIndex = 0;
    juce::String text;
    juce::String callId;
    juce::String toolName;
    juce::String argumentsDelta;
    juce::var providerData;
};

using StreamDeltaCallback = std::function<bool(const StreamDelta&)>;

/** Stateful assembler used by the HTTP transport and exposed for deterministic
    provider-stream fixture tests. */
class StreamAccumulator {
  public:
    void add(const StreamDelta&);
    Response finish() const;

  private:
    juce::String text_;
    std::vector<int> toolCallIndices_;
    std::vector<ToolCall> toolCalls_;
};

}  // namespace llm
