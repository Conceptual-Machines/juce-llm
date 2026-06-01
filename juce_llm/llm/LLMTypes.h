#pragma once

#include <vector>

namespace llm {

//==============================================================================
/** A single prior conversation turn. role is "user" or "assistant". */
struct Message {
    juce::String role;
    juce::String content;
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

    /** Prior conversation turns for stateless providers (Anthropic / OpenAI Chat /
        Gemini). Emitted before `userMessage`, which is the current user turn.
        Empty = single-shot, identical to the original behaviour. Ignored by the
        OpenAI Responses provider, which chains via `previousResponseId` instead. */
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
    double wallSeconds = 0.0;
    bool success = false;
    juce::String error;

    /** Provider response id. Used by the OpenAI Responses API to chain the next
        turn (passed back as `previousResponseId`). Empty for providers that
        don't expose one or where it isn't parsed. */
    juce::String id;
};

//==============================================================================
/** A running multi-turn conversation. Holds the turn history (for stateless
    providers) and the last response id (for the stateful Responses API). The
    client picks whichever the active provider needs; callers just keep one of
    these and pass it to LLMClient::continueConversation. Serialise via
    toVar / fromVar to persist it across UI rebuilds. */
struct Conversation {
    std::vector<Message> messages;  // alternating user / assistant turns
    juce::String lastResponseId;    // OpenAI Responses chaining

    void clear() {
        messages.clear();
        lastResponseId = {};
    }

    juce::var toVar() const {
        auto* obj = new juce::DynamicObject();
        juce::Array<juce::var> turns;
        for (const auto& m : messages) {
            auto* t = new juce::DynamicObject();
            t->setProperty("role", m.role);
            t->setProperty("content", m.content);
            turns.add(juce::var(t));
        }
        obj->setProperty("messages", turns);
        obj->setProperty("lastResponseId", lastResponseId);
        return juce::var(obj);
    }

    static Conversation fromVar(const juce::var& v) {
        Conversation c;
        if (auto* obj = v.getDynamicObject()) {
            if (auto* turns = obj->getProperty("messages").getArray())
                for (const auto& t : *turns)
                    c.messages.push_back({t["role"].toString(), t["content"].toString()});
            c.lastResponseId = obj->getProperty("lastResponseId").toString();
        }
        return c;
    }
};

//==============================================================================
using ResponseCallback = std::function<void(Response)>;

/** Called for each token/chunk during streaming. Return false to cancel. */
using StreamCallback = std::function<bool(const juce::String& token)>;

}  // namespace llm
