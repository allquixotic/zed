---
title: Use a Gateway - Zed
description: Configure OpenRouter, Vercel AI Gateway, Amazon Bedrock, and other gateway or cloud model platforms in Zed.
---

# Use a Gateway

Use a gateway when you route model requests through a platform such as OpenRouter, Vercel AI Gateway, Amazon Bedrock, or another OpenAI-compatible service.

| Gateway                   | Zed AI features | External Agents | Terminal Threads | Notes                                        |
| ------------------------- | --------------- | --------------- | ---------------- | -------------------------------------------- |
| OpenRouter                | Yes             | Separate config | Separate config  | Uses OpenRouter API access                   |
| Vercel AI Gateway         | Yes             | Separate config | Separate config  | Uses Vercel AI Gateway API access            |
| Amazon Bedrock            | Yes             | Separate config | Separate config  | Uses AWS credentials or Bedrock bearer token |
| OpenAI-compatible gateway | Yes             | Separate config | Separate config  | Configure base URL, model, and key           |

## OpenRouter {#openrouter}

Use OpenRouter when you want to route Zed AI features through OpenRouter.

1. Visit [OpenRouter](https://openrouter.ai) and create an account.
2. Generate an API key from your [OpenRouter keys page](https://openrouter.ai/keys).
3. Open **Settings → AI → LLM Providers** with {#action agent::OpenSettings} and find the OpenRouter row.
4. Enter your OpenRouter API key.

Zed also reads `OPENROUTER_API_KEY` from the local Zed process environment.

When using OpenRouter as your assistant provider, explicitly select a model in your settings:

```json [settings]
{
  "agent": {
    "default_model": {
      "provider": "openrouter",
      "model": "openrouter/auto"
    }
  }
}
```

The `openrouter/auto` model routes requests to an available model selected by OpenRouter. You can also specify any model available through OpenRouter's API.

### OpenRouter Custom Models {#openrouter-custom-models}

You can add custom models to the OpenRouter provider in settings:

```json [settings]
{
  "language_models": {
    "open_router": {
      "api_url": "https://openrouter.ai/api/v1",
      "available_models": [
        {
          "name": "google/gemini-2.0-flash-thinking-exp",
          "display_name": "Gemini 2.0 Flash (Thinking)",
          "max_tokens": 200000,
          "max_output_tokens": 8192,
          "supports_tools": true,
          "supports_images": true,
          "mode": {
            "type": "thinking",
            "budget_tokens": 8000
          }
        }
      ]
    }
  }
}
```

Custom model entries support fields such as `name`, `display_name`, `max_tokens`, `max_output_tokens`, `max_completion_tokens`, `supports_tools`, `supports_images`, and `mode`.

### OpenRouter Provider Routing {#openrouter-provider-routing}

You can control how OpenRouter routes a custom model request among upstream providers with the `provider` object on each model entry.

Supported fields include `order`, `allow_fallbacks`, `require_parameters`, `data_collection`, `only`, `ignore`, `quantizations`, and `sort`.

```json [settings]
{
  "language_models": {
    "open_router": {
      "available_models": [
        {
          "name": "openrouter/auto",
          "display_name": "Auto Router",
          "max_tokens": 2000000,
          "supports_tools": true,
          "provider": {
            "order": ["anthropic", "openai"],
            "allow_fallbacks": true,
            "require_parameters": true,
            "data_collection": "allow"
          }
        }
      ]
    }
  }
}
```

## Vercel AI Gateway {#vercel-ai-gateway}

Use Vercel AI Gateway when you want to route Zed AI features through Vercel.

1. Create an API key from your Vercel AI Gateway keys page.
2. Open **Settings → AI → LLM Providers** with {#action agent::OpenSettings} and find the Vercel AI Gateway row.
3. Enter your Vercel AI Gateway API key.

Zed also reads `VERCEL_AI_GATEWAY_API_KEY` from the local Zed process environment.

You can set a custom endpoint for Vercel AI Gateway in settings:

```json [settings]
{
  "language_models": {
    "vercel_ai_gateway": {
      "api_url": "https://ai-gateway.vercel.sh/v1"
    }
  }
}
```

## Amazon Bedrock {#amazon-bedrock}

Use Amazon Bedrock when you want model access through AWS.

Bedrock supports models that support streaming tool use. See [Amazon Bedrock's Tool Use documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html).

Your AWS credentials need these permissions:

- `bedrock:InvokeModelWithResponseStream`
- `bedrock:InvokeModel`
- `bedrock:ListFoundationModels`
- `bedrock:ListInferenceProfiles`

The two list permissions let Zed populate the model picker with the models and inference profiles available to your AWS account. If model discovery is not permitted, Zed keeps its built-in and configured models available and shows the discovery error in Agent Settings.

Bedrock supports Zed-prefixed AWS environment variables so Zed does not override or consume your normal AWS credentials:

- `ZED_ACCESS_KEY_ID`
- `ZED_SECRET_ACCESS_KEY`
- `ZED_SESSION_TOKEN`
- `ZED_AWS_PROFILE`
- `ZED_AWS_REGION`
- `ZED_AWS_ENDPOINT`
- `ZED_BEDROCK_BEARER_TOKEN`

### Bedrock Authentication {#bedrock-authentication}

Open Agent Settings with {#action agent::OpenSettings} and select Amazon Bedrock. You can choose:

- **Automatic (AWS credential chain)** to use the standard AWS SDK credential chain, including environment, container, and instance credentials.
- **AWS Profile** to select any profile loaded from your local AWS config and credentials files, including IAM Identity Center (SSO) profiles.
- **Static credentials or API key** to store an access key pair, temporary IAM credentials, or a Bedrock API key in the system keychain.

Select an AWS Region in the same view. Zed uses that Region for Bedrock Runtime and Mantle model discovery, shows the status of both catalogs, and lets you retry a failed discovery. `ZED_AWS_REGION` overrides the selector until the environment variable is unset and Zed is restarted.

The controls write the same settings available in `settings.json`. Configure any CLI/IAM or SSO profile with `named_profile`:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "authentication_method": "named_profile",
      "region": "your-aws-region",
      "profile": "your-profile-name"
    }
  }
}
```

Use `"authentication_method": "default"` for the automatic credential chain or `"api_key"` for credentials stored in the keychain. Existing `"sso"` settings remain compatible and are normalized to `"named_profile"`.

For a Bedrock API key, choose **Static credentials or API key**, then enter the key in the Bedrock API Key field. The equivalent settings are:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "authentication_method": "api_key",
      "region": "your-aws-region"
    }
  }
}
```

The API key itself is stored in the system keychain, not in `settings.json`. AWS profiles remain in the normal AWS config files; Zed stores only the selected profile name.

### Bedrock Cross-Region Inference {#bedrock-cross-region-inference}

Zed uses [Cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) for Bedrock on a best-effort basis.

By default, Zed uses regional inference profiles. To opt into global profiles, add `allow_global`:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "authentication_method": "named_profile",
      "region": "your-aws-region",
      "profile": "your-profile-name",
      "allow_global": true
    }
  }
}
```

Only some models support global inference profiles. See the AWS Bedrock supported models documentation for the current list.

### Bedrock Guardrails {#bedrock-guardrails}

Some AWS environments require a guardrail on every Bedrock API call. Add `guardrail_identifier` to apply a guardrail to all Bedrock requests:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "guardrail_identifier": "arn:aws:bedrock:us-east-1:123456789012:guardrail/abc123",
      "guardrail_version": "DRAFT"
    }
  }
}
```

### Bedrock Mantle Models {#bedrock-mantle-models}

Some models, such as the GPT-5.6 family (Sol, Terra, and Luna), GPT-5.5, GPT-5.4, and Grok 4.3, aren't available through Bedrock's Converse API and are only reachable through `bedrock-mantle`. Mantle exposes OpenAI Responses and Chat Completions APIs as well as Anthropic's Messages API. Zed routes each discovered model through its supported protocol automatically, including current Mantle Claude models, and shows the result in the normal model picker without requiring a model ID or ARN.

Mantle models require IAM permissions for the `bedrock-mantle` endpoint (for example via the [`AmazonBedrockMantleInferenceAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonBedrockMantleInferenceAccess.html) managed policy) in addition to whatever permissions your existing Bedrock credentials already have. Model discovery uses `bedrock-mantle:ListModels`. Bedrock API keys require `bedrock-mantle:CallWithBearerToken` for Mantle requests.

`bedrock-mantle` is only available in [some AWS Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html#regions). When the selected Region does not support Mantle, Zed disables only the Mantle catalog and keeps Bedrock Runtime models available. If either catalog request fails, built-in and explicitly configured models remain usable.

Claude Mythos 5 and Claude Fable 5 require the AWS account's data retention mode to be set to `provider_data_share` before they can be used. Review [Amazon Bedrock's data retention documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html) before opting in; AWS documents that prompts and completions for these models are shared with Anthropic and retained for up to 30 days for trust and safety purposes.

#### Custom Bedrock Mantle Models {#bedrock-mantle-custom-models}

You can add custom models served through `bedrock-mantle` with `mantle_available_models`:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "mantle_available_models": [
        {
          "name": "openai.gpt-oss-120b",
          "display_name": "GPT-OSS 120B",
          "max_tokens": 128000,
          "protocol": "chat_completions",
          "supports_tools": true,
          "supports_images": false,
          "supports_thinking": true
        }
      ]
    }
  }
}
```

`protocol` selects the API used for the model and must be `chat_completions`, `responses`, or `anthropic_messages`. Set `supports_thinking` to `true` for custom Mantle models that accept reasoning effort parameters; Zed then exposes `low`, `medium`, `high`, and `xhigh` in the thinking effort picker. Disabling thinking sends `none` for OpenAI-compatible protocols and omits thinking parameters for Anthropic Messages.

## OpenAI-Compatible Gateways {#openai-compatible}

If your gateway exposes an OpenAI-compatible API, configure it with [Use API Access](./use-api-access.md#openai-compatible).
