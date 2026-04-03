# Contributing to juce-llm

Thanks for your interest in juce-llm! This document covers how to contribute effectively.

## Before You Start

**Open an issue first.** Describe what you want to change and why. Wait for approval before writing code. PRs for features that have been discussed and declined will be closed immediately.

## What We Accept

- Bug fixes with a clear reproduction case
- Documentation improvements
- Performance improvements
- Support for new LLM providers **only if pre-approved via an issue**

## What We Don't Accept

- Unsolicited provider integrations — we decide which providers to support
- PRs that add external service dependencies without prior discussion
- Large refactors without prior discussion

## Development

juce-llm is a JUCE module. It's developed as a submodule of [magda-core](https://github.com/Conceptual-Machines/magda-core) and tested through that project's build system.

### Building (via magda-core)

```bash
git clone --recursive https://github.com/Conceptual-Machines/magda-core.git
cd magda-core
make debug
```

## PR Guidelines

1. **One concern per PR.** Don't mix fixes with features.
2. **Keep it small.** Large PRs take longer to review.
3. **Match the existing style.** Follow the conventions in the existing code.
4. **Write a clear description.** Explain what changed and why.
5. **Don't break the build.** Test through magda-core before pushing.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
