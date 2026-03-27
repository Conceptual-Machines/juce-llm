#include <juce_core/juce_core.h>
#include <juce_llm/juce_llm.h>

//==============================================================================
/** Simple test runner for juce_llm providers.
    Set environment variables before running:
      OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
*/
class TestRunner {
  public:
    void runAll() {
        printHeader("juce_llm test suite");

        testLlamaServer();
        testOpenAIChat();
        testOpenAIResponses();
        testAnthropic();
        testGemini();
        testOpenRouter();

        printHeader("Results: " + juce::String(passed_) + "/" + juce::String(total_) + " passed");
    }

  private:
    void testLlamaServer() {
        // Try to connect — skip if server isn't running
        llm::ProviderConfig config;
        config.provider = llm::Provider::OpenAIChat;
        config.baseUrl = "http://localhost:8080/v1";
        config.model = "local";

        auto client = llm::LLMClientFactory::create(config);

        llm::Request request;
        request.systemPrompt = "Respond with exactly one word: HELLO";
        request.userMessage = "Say hello.";

        total_++;
        std::cout << "  llama-server (local) ... " << std::flush;

        auto response = client->sendRequest(request);

        if (response.error.contains("Failed to connect")) {
            total_--;
            std::cout << "SKIP (server not running)" << std::endl;
            return;
        }

        if (response.success && response.text.containsIgnoreCase("hello")) {
            passed_++;
            std::cout << "PASS (" << response.wallSeconds << "s)" << std::endl;
        } else {
            std::cout << "FAIL";
            if (response.error.isNotEmpty())
                std::cout << " — " << response.error.substring(0, 100);
            else
                std::cout << " — got: " << response.text.substring(0, 50);
            std::cout << std::endl;
        }
    }

    void testOpenAIChat() {
        auto key = juce::SystemStats::getEnvironmentVariable("OPENAI_API_KEY", "");
        if (key.isEmpty()) {
            skip("OpenAI Chat");
            return;
        }

        llm::ProviderConfig config;
        config.provider = llm::Provider::OpenAIChat;
        config.baseUrl = "https://api.openai.com/v1";
        config.apiKey = key;
        config.model = "gpt-4o-mini";

        runTest("OpenAI Chat (gpt-4o-mini)", config);
    }

    void testOpenAIResponses() {
        auto key = juce::SystemStats::getEnvironmentVariable("OPENAI_API_KEY", "");
        if (key.isEmpty()) {
            skip("OpenAI Responses");
            return;
        }

        llm::ProviderConfig config;
        config.provider = llm::Provider::OpenAIResponses;
        config.baseUrl = "https://api.openai.com/v1";
        config.apiKey = key;
        config.model = "gpt-5";
        config.noTemperature = true;
        config.reasoningEffort = "minimal";

        runTest("OpenAI Responses (gpt-5)", config);
    }

    void testAnthropic() {
        auto key = juce::SystemStats::getEnvironmentVariable("ANTHROPIC_API_KEY", "");
        if (key.isEmpty()) {
            skip("Anthropic");
            return;
        }

        llm::ProviderConfig config;
        config.provider = llm::Provider::Anthropic;
        config.baseUrl = "https://api.anthropic.com/v1";
        config.apiKey = key;
        config.model = "claude-haiku-4-5-20251001";

        runTest("Anthropic (claude-haiku)", config);
    }

    void testGemini() {
        auto key = juce::SystemStats::getEnvironmentVariable("GEMINI_API_KEY", "");
        if (key.isEmpty()) {
            skip("Gemini");
            return;
        }

        llm::ProviderConfig config;
        config.provider = llm::Provider::Gemini;
        config.baseUrl = "https://generativelanguage.googleapis.com";
        config.apiKey = key;
        config.model = "gemini-2.5-flash";

        runTest("Gemini (2.5-flash)", config);
    }

    void testOpenRouter() {
        auto key = juce::SystemStats::getEnvironmentVariable("OPENROUTER_API_KEY", "");
        if (key.isEmpty()) {
            skip("OpenRouter");
            return;
        }

        llm::ProviderConfig config;
        config.provider = llm::Provider::OpenAIChat;
        config.baseUrl = "https://openrouter.ai/api/v1";
        config.apiKey = key;
        config.model = "meta-llama/llama-3.3-70b-instruct";

        runTest("OpenRouter (llama-3.3-70b)", config);
    }

    //==============================================================================
    void runTest(const juce::String& name, const llm::ProviderConfig& config) {
        total_++;
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
            std::cout << "FAIL";
            if (response.error.isNotEmpty())
                std::cout << " — " << response.error.substring(0, 100);
            else
                std::cout << " — got: " << response.text.substring(0, 50);
            std::cout << std::endl;
        }
    }

    void skip(const juce::String& name) {
        std::cout << "  " << name << " ... SKIP (no API key)" << std::endl;
    }

    void printHeader(const juce::String& text) {
        std::cout << std::endl
                  << std::string(60, '=') << std::endl
                  << "  " << text << std::endl
                  << std::string(60, '=') << std::endl;
    }

    int passed_ = 0;
    int total_ = 0;
};

//==============================================================================
int main() {
    TestRunner runner;
    runner.runAll();

    return 0;
}
