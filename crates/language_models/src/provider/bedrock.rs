use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use async_lock::OnceCell;
use aws_config::stalled_stream_protection::StalledStreamProtectionConfig;
use aws_config::{BehaviorVersion, Region};
use aws_credential_types::provider::ProvideCredentials;
use aws_credential_types::{Credentials, Token};
use aws_http_client::AwsHttpClient;
use aws_runtime::env_config::file::EnvConfigFiles;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use aws_types::os_shim_internal::{Env as AwsEnv, Fs as AwsFs};
use bedrock::BedrockSystemContentBlock;
use bedrock::bedrock_client::Client as BedrockClient;
use bedrock::bedrock_client::config::timeout::TimeoutConfig;
use bedrock::bedrock_client::types::{
    CachePointBlock, CachePointType, ContentBlockDelta, ContentBlockStart, ConverseStreamOutput,
    ReasoningContentBlockDelta, StopReason,
};
use bedrock::{
    BedrockAnyToolChoice, BedrockAutoToolChoice, BedrockBlob, BedrockError, BedrockImageBlock,
    BedrockImageFormat, BedrockImageSource, BedrockInnerContent, BedrockMessage, BedrockModelMode,
    BedrockStreamingResponse, BedrockThinkingBlock, BedrockThinkingTextBlock, BedrockTool,
    BedrockToolChoice, BedrockToolConfig, BedrockToolInputSchema, BedrockToolResultBlock,
    BedrockToolResultContentBlock, BedrockToolResultStatus, BedrockToolSpec, BedrockToolUseBlock,
    ConverseModel, MantleModel, MantleProtocol, value_to_aws_document,
};
use collections::{BTreeMap, HashMap, HashSet};
use credentials_provider::CredentialsProvider;
use fs::Fs;
use futures::{
    AsyncBufReadExt, AsyncReadExt, FutureExt, Stream, StreamExt, future::BoxFuture, io::BufReader,
    stream::BoxStream,
};
use gpui::{
    App, AppContext, AsyncApp, Context, Entity, FocusHandle, Subscription, Task, TaskExt, Window,
    actions,
};
use gpui_tokio::Tokio;
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest, RequestBuilderExt,
    http::{HeaderValue, header::AUTHORIZATION},
};
use language_model::{
    AuthenticateError, EnvVar, IconOrSvg, InlineDescription, LanguageModel,
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelEffortLevel,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, LanguageModelToolResultContent, LanguageModelToolSchemaFormat,
    LanguageModelToolUse, MessageContent, ProviderSettingsView, RateLimiter, Role,
    SubPageProviderSettings, TokenUsage, env_var,
};
use open_ai::responses::Request as OpenAiResponseRequest;
use open_ai::responses::{ResponseOutputItem, StreamEvent as OpenAiResponseStreamEvent};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use settings::{
    BedrockAvailableModel as AvailableModel, BedrockMantleAvailableModel as MantleAvailableModel,
    Settings, SettingsStore, update_settings_file,
};
use std::sync::LazyLock;
use std::time::SystemTime;
use strum::{EnumIter, IntoEnumIterator, IntoStaticStr};
use ui::{
    ButtonLink, ConfiguredApiCard, ContextMenu, ContextMenuEntry, Divider, DropdownMenu,
    DropdownStyle, IconPosition, List, ListBulletItem, prelude::*,
};
use ui_input::InputField;
use util::ResultExt;

use crate::AllLanguageModelSettings;
use crate::provider::anthropic::{AnthropicEventMapper, AnthropicPromptCacheMode, into_anthropic};
use crate::provider::open_ai::{
    ChatCompletionMaxTokensParameter, OpenAiEventMapper, OpenAiResponseEventMapper, into_open_ai,
    into_open_ai_response,
};
use language_model::util::{fix_streamed_json, parse_tool_arguments};
use open_ai::{ReasoningEffort, RequestError, ResponseStreamEvent};

actions!(bedrock, [Tab, TabPrev]);

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("amazon-bedrock");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Amazon Bedrock");
pub(crate) const RESERVED_HEADER_NAMES: &[&str] = &[
    "host",
    "x-amz-date",
    "x-amz-security-token",
    "x-amz-content-sha256",
    "amz-sdk-invocation-id",
    "amz-sdk-request",
    "anthropic-version",
];

/// Credentials stored in the keychain for static authentication.
/// Region is handled separately since it's orthogonal to auth method.
#[derive(Default, Clone, Deserialize, Serialize, PartialEq, Debug)]
pub struct BedrockCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub bearer_token: Option<String>,
}

/// Resolved authentication configuration for Bedrock.
/// Settings take priority over UX-provided credentials.
#[derive(Clone, Debug, PartialEq)]
pub enum BedrockAuth {
    /// Use default AWS credential provider chain (IMDSv2, PodIdentity, env vars, etc.)
    Automatic,
    /// Use AWS named profile from ~/.aws/credentials or ~/.aws/config
    NamedProfile { profile_name: String },
    /// Use IAM credentials (access key + secret + optional session token)
    IamCredentials {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    /// Use Bedrock API Key (bearer token authentication)
    ApiKey { api_key: String },
}

impl BedrockCredentials {
    /// Convert stored credentials to the appropriate auth variant.
    /// Prefers API key if present, otherwise uses IAM credentials.
    fn into_auth(self) -> Option<BedrockAuth> {
        if let Some(api_key) = self.bearer_token.filter(|t| !t.is_empty()) {
            Some(BedrockAuth::ApiKey { api_key })
        } else if !self.access_key_id.is_empty() && !self.secret_access_key.is_empty() {
            Some(BedrockAuth::IamCredentials {
                access_key_id: self.access_key_id,
                secret_access_key: self.secret_access_key,
                session_token: self.session_token.filter(|t| !t.is_empty()),
            })
        } else {
            None
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct AmazonBedrockSettings {
    pub available_models: Vec<AvailableModel>,
    pub mantle_available_models: Vec<MantleAvailableModel>,
    pub custom_headers: CustomHeaders,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub profile_name: Option<String>,
    pub role_arn: Option<String>,
    pub authentication_method: Option<BedrockAuthMethod>,
    pub allow_global: Option<bool>,
    pub guardrail_identifier: Option<String>,
    pub guardrail_version: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, EnumIter, IntoStaticStr, JsonSchema)]
pub enum BedrockAuthMethod {
    #[serde(rename = "named_profile", alias = "sso")]
    NamedProfile,
    #[serde(rename = "api_key")]
    ApiKey,
    /// IMDSv2, PodIdentity, env vars, etc.
    #[serde(rename = "default")]
    Automatic,
}

impl From<settings::BedrockAuthMethodContent> for BedrockAuthMethod {
    fn from(value: settings::BedrockAuthMethodContent) -> Self {
        match value {
            settings::BedrockAuthMethodContent::Automatic => BedrockAuthMethod::Automatic,
            settings::BedrockAuthMethodContent::NamedProfile => BedrockAuthMethod::NamedProfile,
            settings::BedrockAuthMethodContent::ApiKey => BedrockAuthMethod::ApiKey,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AwsProfile {
    pub name: String,
}

pub async fn load_aws_profiles() -> Result<Vec<AwsProfile>> {
    let profile_set = aws_config::profile::load(
        &AwsFs::real(),
        &AwsEnv::real(),
        &EnvConfigFiles::default(),
        None,
    )
    .await
    .context("loading AWS profiles")?;
    Ok(aws_profiles_from_set(&profile_set))
}

#[cfg(test)]
async fn load_aws_profiles_from(
    filesystem: &AwsFs,
    environment: &AwsEnv,
    files: &EnvConfigFiles,
) -> Result<Vec<AwsProfile>> {
    let profile_set = aws_config::profile::load(filesystem, environment, files, None)
        .await
        .context("loading AWS profiles")?;
    Ok(aws_profiles_from_set(&profile_set))
}

fn aws_profiles_from_set(profile_set: &aws_config::profile::ProfileSet) -> Vec<AwsProfile> {
    let mut profiles = profile_set
        .profiles()
        .map(|name| AwsProfile {
            name: name.to_string(),
        })
        .collect::<Vec<_>>();
    profiles.sort_by(|left, right| {
        (left.name != "default", left.name.as_str())
            .cmp(&(right.name != "default", right.name.as_str()))
    });
    profiles
}

#[derive(Clone)]
struct AwsSessionCache {
    cell: Arc<OnceCell<AwsSession>>,
}

impl Default for AwsSessionCache {
    fn default() -> Self {
        Self {
            cell: Arc::new(OnceCell::new()),
        }
    }
}

impl AwsSessionCache {
    fn invalidate(&mut self) {
        self.cell = Arc::new(OnceCell::new());
    }
}

#[derive(Clone)]
struct AwsSessionConfiguration {
    auth: Option<BedrockAuth>,
    endpoint: Option<String>,
    region: String,
}

struct AwsSession {
    sdk_config: aws_types::SdkConfig,
    control_client: aws_sdk_bedrock::Client,
    runtime_client: BedrockClient,
    api_key: Option<String>,
}

impl AwsSession {
    async fn resolve_mantle_auth(&self) -> Result<MantleAuth> {
        if let Some(api_key) = &self.api_key {
            return Ok(MantleAuth::ApiKey {
                api_key: api_key.clone(),
            });
        }

        let credentials_provider = self
            .sdk_config
            .credentials_provider()
            .context("no AWS credentials provider is configured")?;
        let credentials = credentials_provider
            .provide_credentials()
            .await
            .context("failed to resolve AWS credentials")?;
        Ok(MantleAuth::SigV4 { credentials })
    }
}

async fn build_aws_session(
    configuration: AwsSessionConfiguration,
    http_client: AwsHttpClient,
) -> Result<AwsSession> {
    let api_key = match &configuration.auth {
        Some(BedrockAuth::ApiKey { api_key }) => Some(api_key.clone()),
        _ => None,
    };
    let mut config_builder = aws_config::defaults(BehaviorVersion::latest())
        .stalled_stream_protection(StalledStreamProtectionConfig::disabled())
        .http_client(http_client)
        .region(Region::new(configuration.region))
        .timeout_config(TimeoutConfig::disabled());

    if let Some(endpoint_url) = configuration.endpoint.filter(|url| !url.is_empty()) {
        config_builder = config_builder.endpoint_url(endpoint_url);
    }

    match configuration.auth {
        Some(BedrockAuth::Automatic) | None => {}
        Some(BedrockAuth::NamedProfile { profile_name }) => {
            if !profile_name.is_empty() {
                config_builder = config_builder.profile_name(profile_name);
            }
        }
        Some(BedrockAuth::IamCredentials {
            access_key_id,
            secret_access_key,
            session_token,
        }) => {
            config_builder = config_builder.credentials_provider(Credentials::new(
                access_key_id,
                secret_access_key,
                session_token,
                None,
                "zed-bedrock-provider",
            ));
        }
        Some(BedrockAuth::ApiKey { api_key }) => {
            config_builder = config_builder
                .auth_scheme_preference(["httpBearerAuth".into()])
                .token_provider(Token::new(api_key, None));
        }
    }

    let sdk_config = config_builder.load().await;
    let control_client = aws_sdk_bedrock::Client::new(&sdk_config);
    let runtime_client = BedrockClient::new(&sdk_config);
    Ok(AwsSession {
        sdk_config,
        control_client,
        runtime_client,
        api_key,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BedrockEndpoint {
    Runtime,
    Mantle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BedrockTargetKind {
    FoundationModel,
    SystemInferenceProfile,
    ApplicationInferenceProfile,
    MantleModel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BedrockTargetProtocol {
    Converse,
    OpenAiChatCompletions,
    OpenAiResponses,
    AnthropicMessages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BedrockTargetCapabilities {
    pub max_token_count: u64,
    pub max_output_tokens: u64,
    pub supports_tools: bool,
    pub supports_images: bool,
    pub supports_thinking: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BedrockTargetDescriptor {
    pub endpoint: BedrockEndpoint,
    pub kind: BedrockTargetKind,
    pub protocol: BedrockTargetProtocol,
    pub id: String,
    pub invocation_id: String,
    pub arn: Option<String>,
    pub base_model_id: String,
    pub display_name: String,
    pub capabilities: BedrockTargetCapabilities,
}

impl BedrockTargetDescriptor {
    fn into_converse_model(self) -> ConverseModel {
        let base_model = ConverseModel::from_request_id(&self.base_model_id).unwrap_or_else(|| {
            ConverseModel::Custom {
                name: self.base_model_id,
                max_tokens: self.capabilities.max_token_count,
                display_name: None,
                max_output_tokens: Some(self.capabilities.max_output_tokens),
                default_temperature: None,
                cache_configuration: None,
            }
        });
        ConverseModel::Discovered {
            id: self.id,
            invocation_id: self.invocation_id,
            display_name: self.display_name,
            base_model: Box::new(base_model),
        }
    }

    fn into_mantle_model(self) -> MantleModel {
        MantleModel::Discovered {
            id: self.id,
            request_id: self.invocation_id,
            display_name: self.display_name,
            protocol: match self.protocol {
                BedrockTargetProtocol::OpenAiChatCompletions => MantleProtocol::ChatCompletions,
                BedrockTargetProtocol::OpenAiResponses => MantleProtocol::Responses,
                BedrockTargetProtocol::AnthropicMessages => MantleProtocol::AnthropicMessages,
                BedrockTargetProtocol::Converse => MantleProtocol::ChatCompletions,
            },
            max_tokens: self.capabilities.max_token_count,
            max_output_tokens: self.capabilities.max_output_tokens,
            supports_tools: self.capabilities.supports_tools,
            supports_images: self.capabilities.supports_images,
            supports_thinking: self.capabilities.supports_thinking,
        }
    }
}

fn conservative_target_capabilities() -> BedrockTargetCapabilities {
    BedrockTargetCapabilities {
        max_token_count: 128_000,
        max_output_tokens: 4_096,
        supports_tools: false,
        supports_images: false,
        supports_thinking: false,
    }
}

fn converse_target_capabilities(model_id: &str) -> BedrockTargetCapabilities {
    ConverseModel::from_request_id(model_id).map_or_else(
        conservative_target_capabilities,
        |model| BedrockTargetCapabilities {
            max_token_count: model.max_token_count(),
            max_output_tokens: model.max_output_tokens(),
            supports_tools: model.supports_tool_use(),
            supports_images: model.supports_images(),
            supports_thinking: model.supports_thinking(),
        },
    )
}

fn mantle_target_capabilities(model: &MantleModel) -> BedrockTargetCapabilities {
    BedrockTargetCapabilities {
        max_token_count: model.max_token_count(),
        max_output_tokens: model.max_output_tokens(),
        supports_tools: model.supports_tools(),
        supports_images: model.supports_images(),
        supports_thinking: model.supports_thinking(),
    }
}

#[derive(Clone, Debug)]
struct RuntimeFoundationModel {
    model_arn: String,
    model_id: String,
    model_name: Option<String>,
    output_modalities: Vec<String>,
    response_streaming_supported: Option<bool>,
    inference_types_supported: Vec<String>,
    lifecycle_status: Option<String>,
}

#[derive(Clone, Debug)]
struct RuntimeInferenceProfile {
    name: String,
    arn: String,
    model_arns: Vec<String>,
    id: String,
    status: String,
    kind: String,
}

#[derive(Clone, Debug)]
struct RuntimeInferenceProfilePage {
    profiles: Vec<RuntimeInferenceProfile>,
    next_token: Option<String>,
}

trait RuntimeCatalogSource: Send + Sync {
    fn list_foundation_models(&self) -> BoxFuture<'_, Result<Vec<RuntimeFoundationModel>>>;
    fn list_inference_profiles(
        &self,
        next_token: Option<String>,
    ) -> BoxFuture<'_, Result<RuntimeInferenceProfilePage>>;
}

struct AwsRuntimeCatalogSource {
    client: aws_sdk_bedrock::Client,
}

impl RuntimeCatalogSource for AwsRuntimeCatalogSource {
    fn list_foundation_models(&self) -> BoxFuture<'_, Result<Vec<RuntimeFoundationModel>>> {
        async move {
            let output = self
                .client
                .list_foundation_models()
                .send()
                .await
                .context("listing Bedrock foundation models")?;
            Ok(output
                .model_summaries()
                .iter()
                .map(|model| RuntimeFoundationModel {
                    model_arn: model.model_arn().to_string(),
                    model_id: model.model_id().to_string(),
                    model_name: model.model_name().map(str::to_string),
                    output_modalities: model
                        .output_modalities()
                        .iter()
                        .map(|modality| modality.as_str().to_string())
                        .collect(),
                    response_streaming_supported: model.response_streaming_supported(),
                    inference_types_supported: model
                        .inference_types_supported()
                        .iter()
                        .map(|inference_type| inference_type.as_str().to_string())
                        .collect(),
                    lifecycle_status: model
                        .model_lifecycle()
                        .map(|lifecycle| lifecycle.status().as_str().to_string()),
                })
                .collect())
        }
        .boxed()
    }

    fn list_inference_profiles(
        &self,
        next_token: Option<String>,
    ) -> BoxFuture<'_, Result<RuntimeInferenceProfilePage>> {
        async move {
            let mut request = self.client.list_inference_profiles().max_results(100);
            if let Some(next_token) = next_token {
                request = request.next_token(next_token);
            }
            let output = request
                .send()
                .await
                .context("listing Bedrock inference profiles")?;
            Ok(RuntimeInferenceProfilePage {
                profiles: output
                    .inference_profile_summaries()
                    .iter()
                    .map(|profile| RuntimeInferenceProfile {
                        name: profile.inference_profile_name().to_string(),
                        arn: profile.inference_profile_arn().to_string(),
                        model_arns: profile
                            .models()
                            .iter()
                            .filter_map(|model| model.model_arn().map(str::to_string))
                            .collect(),
                        id: profile.inference_profile_id().to_string(),
                        status: profile.status().as_str().to_string(),
                        kind: profile.r#type().as_str().to_string(),
                    })
                    .collect(),
                next_token: output.next_token().map(str::to_string),
            })
        }
        .boxed()
    }
}

async fn discover_runtime_targets(
    source: &dyn RuntimeCatalogSource,
) -> Result<Vec<BedrockTargetDescriptor>> {
    let foundation_models = source.list_foundation_models();
    let inference_profiles = async {
        let mut profiles = Vec::new();
        let mut next_token: Option<String> = None;
        let mut seen_tokens = BTreeMap::default();
        loop {
            if let Some(token) = next_token.as_ref() {
                if seen_tokens.insert(token.clone(), ()).is_some() {
                    return Err(anyhow!(
                        "Bedrock returned a repeated inference-profile page token"
                    ));
                }
            }
            let page = source.list_inference_profiles(next_token).await?;
            profiles.extend(page.profiles);
            next_token = page.next_token;
            if next_token.is_none() {
                break;
            }
        }
        Ok::<_, anyhow::Error>(profiles)
    };
    let (foundation_models, inference_profiles) =
        futures::try_join!(foundation_models, inference_profiles)?;

    let mut targets = BTreeMap::default();
    for model in foundation_models {
        if model
            .lifecycle_status
            .as_deref()
            .is_some_and(|status| status != "ACTIVE")
            || !model
                .output_modalities
                .iter()
                .any(|modality| modality == "TEXT")
            || model.response_streaming_supported == Some(false)
            || !model
                .inference_types_supported
                .iter()
                .any(|inference_type| inference_type == "ON_DEMAND")
        {
            continue;
        }
        let id = format!("bedrock-runtime:foundation:{}", model.model_id);
        let capabilities = converse_target_capabilities(&model.model_id);
        targets.insert(
            id.clone(),
            BedrockTargetDescriptor {
                endpoint: BedrockEndpoint::Runtime,
                kind: BedrockTargetKind::FoundationModel,
                protocol: BedrockTargetProtocol::Converse,
                id,
                invocation_id: model.model_id.clone(),
                arn: Some(model.model_arn),
                base_model_id: model.model_id.clone(),
                display_name: model.model_name.unwrap_or(model.model_id),
                capabilities,
            },
        );
    }

    for profile in inference_profiles {
        if profile.status != "ACTIVE" {
            continue;
        }
        let Some(base_model_id) = profile.model_arns.iter().find_map(|arn| {
            arn.split_once("foundation-model/")
                .map(|(_, model_id)| model_id.to_string())
        }) else {
            continue;
        };
        let kind = match profile.kind.as_str() {
            "SYSTEM_DEFINED" => BedrockTargetKind::SystemInferenceProfile,
            "APPLICATION" => BedrockTargetKind::ApplicationInferenceProfile,
            _ => continue,
        };
        let id = format!("bedrock-runtime:profile:{}", profile.id);
        let capabilities = converse_target_capabilities(&base_model_id);
        targets.insert(
            id.clone(),
            BedrockTargetDescriptor {
                endpoint: BedrockEndpoint::Runtime,
                kind,
                protocol: BedrockTargetProtocol::Converse,
                id,
                invocation_id: profile.id,
                arn: Some(profile.arn),
                base_model_id,
                display_name: profile.name,
                capabilities,
            },
        );
    }

    let mut targets = targets.into_values().collect::<Vec<_>>();
    targets.sort_by(|left, right| {
        (
            left.kind,
            left.base_model_id.as_str(),
            left.display_name.as_str(),
        )
            .cmp(&(
                right.kind,
                right.base_model_id.as_str(),
                right.display_name.as_str(),
            ))
    });

    let mut assigned_stable_ids = HashSet::default();
    for target in &mut targets {
        let Some(model) = ConverseModel::from_request_id(&target.base_model_id) else {
            continue;
        };
        let stable_id = model.id().to_string();
        if assigned_stable_ids.insert(stable_id.clone()) {
            target.id = stable_id;
        }
    }

    Ok(targets)
}

#[derive(Deserialize)]
struct MantleModelsResponse {
    data: Vec<MantleModelEntry>,
}

#[derive(Deserialize)]
struct MantleModelEntry {
    id: String,
}

fn mantle_target_protocol(model: &MantleModel) -> BedrockTargetProtocol {
    match model.protocol() {
        MantleProtocol::ChatCompletions => BedrockTargetProtocol::OpenAiChatCompletions,
        MantleProtocol::Responses => BedrockTargetProtocol::OpenAiResponses,
        MantleProtocol::AnthropicMessages => BedrockTargetProtocol::AnthropicMessages,
    }
}

fn conservative_mantle_protocol(model_id: &str) -> BedrockTargetProtocol {
    if model_id.starts_with("anthropic.") {
        BedrockTargetProtocol::AnthropicMessages
    } else if model_id.starts_with("openai.") || model_id.starts_with("xai.") {
        BedrockTargetProtocol::OpenAiResponses
    } else {
        BedrockTargetProtocol::OpenAiChatCompletions
    }
}

fn mantle_targets_from_model_ids(
    model_ids: impl IntoIterator<Item = String>,
) -> Vec<BedrockTargetDescriptor> {
    let mut targets = BTreeMap::default();
    for model_id in model_ids {
        let known_model = MantleModel::from_request_id(&model_id);
        let (id, display_name, protocol, capabilities) = if let Some(model) = known_model {
            (
                model.id().to_string(),
                model.display_name().to_string(),
                mantle_target_protocol(&model),
                mantle_target_capabilities(&model),
            )
        } else {
            (
                format!("bedrock-mantle/{model_id}"),
                model_id.clone(),
                conservative_mantle_protocol(&model_id),
                conservative_target_capabilities(),
            )
        };
        targets.insert(
            id.clone(),
            BedrockTargetDescriptor {
                endpoint: BedrockEndpoint::Mantle,
                kind: BedrockTargetKind::MantleModel,
                protocol,
                id,
                invocation_id: model_id.clone(),
                arn: None,
                base_model_id: model_id,
                display_name,
                capabilities,
            },
        );
    }
    targets.into_values().collect()
}

fn parse_mantle_model_catalog(body: &str) -> Result<Vec<BedrockTargetDescriptor>> {
    let response: MantleModelsResponse =
        serde_json::from_str(body).context("parsing Bedrock Mantle model catalog")?;
    Ok(mantle_targets_from_model_ids(
        response.data.into_iter().map(|model| model.id),
    ))
}

async fn discover_mantle_targets(
    client: &dyn HttpClient,
    region: &str,
    auth: &MantleAuth,
    extra_headers: &CustomHeaders,
) -> Result<Vec<BedrockTargetDescriptor>> {
    let url = mantle_endpoint_url(region, "v1/models");
    let request =
        build_mantle_http_request(Method::GET, &url, &[], region, auth, extra_headers, &[])?;
    let mut response = client
        .send(request)
        .await
        .context("listing Bedrock Mantle models")?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Bedrock Mantle model discovery returned {status}");
    }
    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .context("reading Bedrock Mantle model catalog")?;
    parse_mantle_model_catalog(&body)
}

fn mantle_protocol_from_settings(value: settings::BedrockMantleProtocolContent) -> MantleProtocol {
    match value {
        settings::BedrockMantleProtocolContent::ChatCompletions => MantleProtocol::ChatCompletions,
        settings::BedrockMantleProtocolContent::Responses => MantleProtocol::Responses,
        settings::BedrockMantleProtocolContent::AnthropicMessages => {
            MantleProtocol::AnthropicMessages
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelMode {
    #[default]
    Default,
    Thinking {
        /// The maximum number of tokens to use for reasoning. Must be lower than the model's `max_output_tokens`.
        budget_tokens: Option<u64>,
    },
    AdaptiveThinking {
        effort: bedrock::BedrockAdaptiveThinkingEffort,
    },
}

impl From<ModelMode> for BedrockModelMode {
    fn from(value: ModelMode) -> Self {
        match value {
            ModelMode::Default => BedrockModelMode::Default,
            ModelMode::Thinking { budget_tokens } => BedrockModelMode::Thinking { budget_tokens },
            ModelMode::AdaptiveThinking { effort } => BedrockModelMode::AdaptiveThinking { effort },
        }
    }
}

impl From<BedrockModelMode> for ModelMode {
    fn from(value: BedrockModelMode) -> Self {
        match value {
            BedrockModelMode::Default => ModelMode::Default,
            BedrockModelMode::Thinking { budget_tokens } => ModelMode::Thinking { budget_tokens },
            BedrockModelMode::AdaptiveThinking { effort } => ModelMode::AdaptiveThinking { effort },
        }
    }
}

/// The URL of the base AWS service.
///
/// Right now we're just using this as the key to store the AWS credentials
/// under in the keychain.
const AMAZON_AWS_URL: &str = "https://amazonaws.com";

// These environment variables all use a `ZED_` prefix because we don't want to overwrite the user's AWS credentials.
static ZED_BEDROCK_ACCESS_KEY_ID_VAR: LazyLock<EnvVar> = env_var!("ZED_ACCESS_KEY_ID");
static ZED_BEDROCK_SECRET_ACCESS_KEY_VAR: LazyLock<EnvVar> = env_var!("ZED_SECRET_ACCESS_KEY");
static ZED_BEDROCK_SESSION_TOKEN_VAR: LazyLock<EnvVar> = env_var!("ZED_SESSION_TOKEN");
static ZED_AWS_PROFILE_VAR: LazyLock<EnvVar> = env_var!("ZED_AWS_PROFILE");
static ZED_BEDROCK_REGION_VAR: LazyLock<EnvVar> = env_var!("ZED_AWS_REGION");
static ZED_AWS_ENDPOINT_VAR: LazyLock<EnvVar> = env_var!("ZED_AWS_ENDPOINT");
static ZED_BEDROCK_BEARER_TOKEN_VAR: LazyLock<EnvVar> = env_var!("ZED_BEDROCK_BEARER_TOKEN");

/// AWS Regions where the `bedrock-mantle` endpoint is available.
/// See <https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html#regions>.
const MANTLE_SUPPORTED_REGIONS: &[&str] = &[
    "us-east-2",
    "us-east-1",
    "us-west-2",
    "ap-southeast-3",
    "ap-south-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-south-1",
    "eu-north-1",
    "sa-east-1",
    "us-gov-west-1",
];

fn mantle_endpoint_url(region: &str, path: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws/{path}")
}

const ANTHROPIC_MANTLE_HEADERS: &[(&str, &str)] = &[("anthropic-version", "2023-06-01")];

enum MantleAuth {
    ApiKey { api_key: String },
    SigV4 { credentials: Credentials },
}

impl MantleAuth {
    fn apply(&self, request: &mut HttpRequest<AsyncBody>, body: &[u8], region: &str) -> Result<()> {
        match self {
            MantleAuth::ApiKey { api_key } => {
                let value = HeaderValue::from_str(&format!("Bearer {}", api_key.trim()))
                    .context("building Mantle bearer token authorization header")?;
                request.headers_mut().insert(AUTHORIZATION, value);
            }
            MantleAuth::SigV4 { credentials } => {
                sign_mantle_request_sigv4(request, body, credentials, region)?;
            }
        }

        Ok(())
    }
}

fn sign_mantle_request_sigv4(
    request: &mut HttpRequest<AsyncBody>,
    body: &[u8],
    credentials: &Credentials,
    region: &str,
) -> Result<()> {
    sign_mantle_request_sigv4_at(request, body, credentials, region, SystemTime::now())
}

fn sign_mantle_request_sigv4_at(
    request: &mut HttpRequest<AsyncBody>,
    body: &[u8],
    credentials: &Credentials,
    region: &str,
    time: SystemTime,
) -> Result<()> {
    if !request
        .headers()
        .contains_key(http_client::http::header::HOST)
        && let Some(authority) = request.uri().authority()
    {
        let host = HeaderValue::from_str(authority.as_str())
            .context("invalid host header derived from Mantle request URI")?;
        request
            .headers_mut()
            .insert(http_client::http::header::HOST, host);
    }

    let identity = credentials.clone().into();
    let signing_params: aws_sigv4::http_request::SigningParams = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name("bedrock-mantle")
        .time(time)
        .settings(SigningSettings::default())
        .build()
        .context("building Mantle SigV4 signing params")?
        .into();

    let method = request.method().as_str();
    let uri = request.uri().to_string();
    let headers = request
        .headers()
        .iter()
        .map(|(name, value)| {
            value
                .to_str()
                .map(|value| (name.as_str(), value))
                .with_context(|| format!("header {name} is not valid UTF-8 and cannot be signed"))
        })
        .collect::<Result<Vec<_>>>()?;

    let signable_request =
        SignableRequest::new(method, uri, headers.into_iter(), SignableBody::Bytes(body))
            .context("constructing Mantle SigV4 request")?;

    let (instructions, _signature) = sign(signable_request, &signing_params)
        .context("signing Mantle request with SigV4")?
        .into_parts();
    instructions.apply_to_request_http1x(request);

    Ok(())
}

pub struct State {
    /// The resolved authentication method. Settings take priority over UX credentials.
    auth: Option<BedrockAuth>,
    /// Raw settings from settings.json
    settings: Option<AmazonBedrockSettings>,
    /// Whether credentials came from environment variables (only relevant for static credentials)
    credentials_from_env: bool,
    configuration_revision: u64,
    credentials_provider: Arc<dyn CredentialsProvider>,
    aws_http_client: AwsHttpClient,
    plain_http_client: Arc<dyn HttpClient>,
    aws_session_cache: AwsSessionCache,
    runtime_targets: Vec<BedrockTargetDescriptor>,
    runtime_discovery_error: Option<String>,
    mantle_targets: Vec<BedrockTargetDescriptor>,
    mantle_discovery_error: Option<String>,
    mantle_endpoint_available: Option<bool>,
    discovery_in_progress: bool,
    discovery_task: Task<()>,
    settings_authentication_task: Task<()>,
    _subscription: Subscription,
}

impl State {
    fn set_auth(&mut self, auth: Option<BedrockAuth>) {
        if self.auth != auth {
            self.auth = auth;
            self.aws_session_cache.invalidate();
            self.reset_discovery();
        }
    }

    fn reset_discovery(&mut self) {
        self.runtime_targets.clear();
        self.runtime_discovery_error = None;
        self.mantle_targets.clear();
        self.mantle_discovery_error = None;
        self.mantle_endpoint_available = None;
        self.discovery_in_progress = false;
        self.discovery_task = Task::ready(());
    }

    fn refresh_targets(&mut self, cx: &mut Context<Self>) {
        if !self.is_authenticated() {
            return;
        }

        let session_cache = self.aws_session_cache.clone();
        let configuration = self.aws_session_configuration();
        let aws_http_client = self.aws_http_client.clone();
        let plain_http_client = self.plain_http_client.clone();
        let region = configuration.region.clone();
        let mantle_endpoint_available = MANTLE_SUPPORTED_REGIONS.contains(&region.as_str());
        let extra_headers = self
            .settings
            .as_ref()
            .map(|settings| settings.custom_headers.clone())
            .unwrap_or_default();
        self.runtime_discovery_error = None;
        self.mantle_discovery_error = None;
        self.mantle_endpoint_available = Some(mantle_endpoint_available);
        self.discovery_in_progress = true;
        let discovery_task = spawn_bedrock_discovery(cx, async move {
            let session = session_cache
                .cell
                .get_or_try_init(|| build_aws_session(configuration, aws_http_client))
                .await;
            match session {
                Ok(session) => {
                    let runtime_source = AwsRuntimeCatalogSource {
                        client: session.control_client.clone(),
                    };
                    let runtime = discover_runtime_targets(&runtime_source);
                    let mantle = async {
                        if !mantle_endpoint_available {
                            return Ok(Vec::new());
                        }
                        let auth = session.resolve_mantle_auth().await?;
                        discover_mantle_targets(
                            plain_http_client.as_ref(),
                            &region,
                            &auth,
                            &extra_headers,
                        )
                        .await
                    };
                    futures::join!(runtime, mantle)
                }
                Err(error) => {
                    let error = format!("initializing Bedrock session: {error}");
                    (Err(anyhow!(error.clone())), Err(anyhow!(error)))
                }
            }
        });
        self.discovery_task = cx.spawn(async move |this, cx| {
            let (runtime_result, mantle_result) =
                contain_bedrock_discovery_task_failure(discovery_task.await);

            if let Err(error) = this.update(cx, |this, cx| {
                match runtime_result {
                    Ok(targets) => {
                        this.runtime_targets = targets;
                        this.runtime_discovery_error = None;
                    }
                    Err(error) => {
                        log::warn!("Bedrock model discovery failed: {error:#}");
                        this.runtime_discovery_error = Some(format!("{error:#}"));
                    }
                }
                match mantle_result {
                    Ok(targets) => {
                        this.mantle_targets = targets;
                        this.mantle_discovery_error = None;
                    }
                    Err(error) => {
                        log::warn!("Bedrock Mantle model discovery failed: {error:#}");
                        this.mantle_discovery_error = Some(format!("{error:#}"));
                    }
                }
                this.discovery_in_progress = false;
                cx.notify();
            }) {
                log::debug!("Bedrock provider was dropped during model discovery: {error:#}");
            }
        });
        cx.notify();
    }

    fn aws_session_configuration(&self) -> AwsSessionConfiguration {
        AwsSessionConfiguration {
            auth: self.auth.clone(),
            endpoint: self
                .settings
                .as_ref()
                .and_then(|settings| settings.endpoint.clone()),
            region: self.get_region(),
        }
    }

    fn reset_auth(&self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        cx.spawn(async move |this, cx| {
            credentials_provider
                .delete_credentials(AMAZON_AWS_URL, cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.set_auth(None);
                this.credentials_from_env = false;
                cx.notify();
            })
        })
    }

    fn set_static_credentials(
        &mut self,
        credentials: BedrockCredentials,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let auth = credentials.clone().into_auth();
        let credentials_provider = self.credentials_provider.clone();
        cx.spawn(async move |this, cx| {
            credentials_provider
                .write_credentials(
                    AMAZON_AWS_URL,
                    "Bearer",
                    &serde_json::to_vec(&credentials)?,
                    cx,
                )
                .await?;
            this.update(cx, |this, cx| {
                this.set_auth(auth);
                this.credentials_from_env = false;
                this.refresh_targets(cx);
                cx.notify();
            })
        })
    }

    fn is_authenticated(&self) -> bool {
        self.auth.is_some()
    }

    /// Resolve authentication. Settings take priority over UX-provided credentials.
    fn authenticate_with_invalidation(
        &mut self,
        invalidate_session: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            if invalidate_session {
                self.aws_session_cache.invalidate();
                self.reset_discovery();
            }
            return Task::ready(Ok(()));
        }
        let configuration_revision = self.configuration_revision;

        // Step 1: Check if settings specify an auth method (enterprise control)
        if let Some(settings) = &self.settings {
            if let Some(method) = &settings.authentication_method {
                let profile_name = settings
                    .profile_name
                    .clone()
                    .unwrap_or_else(|| "default".to_string());

                let auth = match method {
                    BedrockAuthMethod::Automatic => BedrockAuth::Automatic,
                    BedrockAuthMethod::NamedProfile => BedrockAuth::NamedProfile { profile_name },
                    BedrockAuthMethod::ApiKey => {
                        // ApiKey method means "use static credentials from keychain/env"
                        // Fall through to load them below
                        return self.load_static_credentials(
                            invalidate_session,
                            configuration_revision,
                            cx,
                        );
                    }
                };

                return cx.spawn(async move |this, cx| {
                    this.update(cx, |this, cx| {
                        if this.configuration_revision != configuration_revision {
                            return;
                        }
                        if invalidate_session {
                            this.set_auth(Some(auth));
                        } else {
                            this.auth = Some(auth);
                        }
                        this.credentials_from_env = false;
                        cx.notify();
                    })?;
                    Ok(())
                });
            }
        }

        if let Some(profile_name) = ZED_AWS_PROFILE_VAR
            .value
            .as_deref()
            .filter(|profile_name| !profile_name.is_empty())
            .map(str::to_string)
        {
            return cx.spawn(async move |this, cx| {
                this.update(cx, |this, cx| {
                    if this.configuration_revision != configuration_revision {
                        return;
                    }
                    let auth = BedrockAuth::NamedProfile { profile_name };
                    if invalidate_session {
                        this.set_auth(Some(auth));
                    } else {
                        this.auth = Some(auth);
                    }
                    this.credentials_from_env = false;
                    cx.notify();
                })?;
                Ok(())
            });
        }

        // Step 2: No settings auth method - try to load static credentials
        self.load_static_credentials(invalidate_session, configuration_revision, cx)
    }

    /// Load static credentials from environment variables or keychain.
    fn load_static_credentials(
        &self,
        invalidate_session: bool,
        configuration_revision: u64,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        cx.spawn(async move |this, cx| {
            // Try environment variables first
            let (auth, from_env) = if let Some(bearer_token) = &ZED_BEDROCK_BEARER_TOKEN_VAR.value {
                if !bearer_token.is_empty() {
                    (
                        Some(BedrockAuth::ApiKey {
                            api_key: bearer_token.to_string(),
                        }),
                        true,
                    )
                } else {
                    (None, false)
                }
            } else if let Some(access_key_id) = &ZED_BEDROCK_ACCESS_KEY_ID_VAR.value {
                if let Some(secret_access_key) = &ZED_BEDROCK_SECRET_ACCESS_KEY_VAR.value {
                    if !access_key_id.is_empty() && !secret_access_key.is_empty() {
                        let session_token = ZED_BEDROCK_SESSION_TOKEN_VAR
                            .value
                            .as_deref()
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string());
                        (
                            Some(BedrockAuth::IamCredentials {
                                access_key_id: access_key_id.to_string(),
                                secret_access_key: secret_access_key.to_string(),
                                session_token,
                            }),
                            true,
                        )
                    } else {
                        (None, false)
                    }
                } else {
                    (None, false)
                }
            } else {
                (None, false)
            };

            // If we got auth from env vars, use it
            if let Some(auth) = auth {
                this.update(cx, |this, cx| {
                    if this.configuration_revision != configuration_revision {
                        return;
                    }
                    if invalidate_session {
                        this.set_auth(Some(auth));
                    } else {
                        this.auth = Some(auth);
                    }
                    this.credentials_from_env = from_env;
                    cx.notify();
                })?;
                return Ok(());
            }

            // Try keychain
            let (_, credentials_bytes) = credentials_provider
                .read_credentials(AMAZON_AWS_URL, cx)
                .await?
                .ok_or(AuthenticateError::CredentialsNotFound)?;

            let credentials_str = String::from_utf8(credentials_bytes)
                .with_context(|| format!("invalid {PROVIDER_NAME} credentials"))?;

            let credentials: BedrockCredentials =
                serde_json::from_str(&credentials_str).context("failed to parse credentials")?;

            let auth = credentials
                .into_auth()
                .ok_or(AuthenticateError::CredentialsNotFound)?;

            this.update(cx, |this, cx| {
                if this.configuration_revision != configuration_revision {
                    return;
                }
                if invalidate_session {
                    this.set_auth(Some(auth));
                } else {
                    this.auth = Some(auth);
                }
                this.credentials_from_env = false;
                cx.notify();
            })?;

            Ok(())
        })
    }

    fn authenticate_and_refresh(
        &mut self,
        invalidate_session: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), AuthenticateError>> {
        let configuration_revision = self.configuration_revision;
        let authenticate = self.authenticate_with_invalidation(invalidate_session, cx);
        cx.spawn(async move |this, cx| {
            authenticate.await?;
            this.update(cx, |this, cx| {
                if this.configuration_revision == configuration_revision {
                    this.refresh_targets(cx);
                }
            })?;
            Ok(())
        })
    }

    /// Get the resolved region. Checks env var, then settings, then defaults to us-east-1.
    fn get_region(&self) -> String {
        // Priority: env var > settings > default
        if let Some(region) = ZED_BEDROCK_REGION_VAR.value.as_deref() {
            if !region.is_empty() {
                return region.to_string();
            }
        }

        self.settings
            .as_ref()
            .and_then(|s| s.region.clone())
            .unwrap_or_else(|| "us-east-1".to_string())
    }

    fn get_allow_global(&self) -> bool {
        self.settings
            .as_ref()
            .and_then(|s| s.allow_global)
            .unwrap_or(false)
    }

    fn get_guardrail_config(&self) -> (Option<String>, Option<String>) {
        self.settings.as_ref().map_or((None, None), |s| {
            (s.guardrail_identifier.clone(), s.guardrail_version.clone())
        })
    }
}

fn spawn_bedrock_discovery<C, F, R>(cx: &C, future: F) -> Task<Result<R>>
where
    C: AppContext,
    F: Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    Tokio::spawn_result(cx, async move { Ok(future.await) })
}

type BedrockDiscoveryResult = Result<Vec<BedrockTargetDescriptor>>;
type BedrockDiscoveryResults = (BedrockDiscoveryResult, BedrockDiscoveryResult);

fn contain_bedrock_discovery_task_failure(
    task_result: Result<BedrockDiscoveryResults>,
) -> BedrockDiscoveryResults {
    match task_result {
        Ok(results) => results,
        Err(error) => {
            let error = format!("running Bedrock discovery: {error:#}");
            (Err(anyhow!(error.clone())), Err(anyhow!(error)))
        }
    }
}

pub struct BedrockLanguageModelProvider {
    http_client: AwsHttpClient,
    plain_http_client: Arc<dyn HttpClient>,
    handle: tokio::runtime::Handle,
    state: Entity<State>,
}

impl BedrockLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let aws_http_client = AwsHttpClient::new(http_client.clone());
        let state = cx.new(|cx| State {
            auth: None,
            settings: Some(AllLanguageModelSettings::get_global(cx).bedrock.clone()),
            credentials_from_env: false,
            configuration_revision: 0,
            credentials_provider,
            aws_http_client: aws_http_client.clone(),
            plain_http_client: http_client.clone(),
            aws_session_cache: AwsSessionCache::default(),
            runtime_targets: Vec::new(),
            runtime_discovery_error: None,
            mantle_targets: Vec::new(),
            mantle_discovery_error: None,
            mantle_endpoint_available: None,
            discovery_in_progress: false,
            discovery_task: Task::ready(()),
            settings_authentication_task: Task::ready(()),
            _subscription: cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let settings = AllLanguageModelSettings::get_global(cx).bedrock.clone();
                if this.settings.as_ref() != Some(&settings) {
                    this.settings = Some(settings);
                    this.auth = None;
                    this.credentials_from_env = false;
                    this.configuration_revision = this.configuration_revision.wrapping_add(1);
                    this.aws_session_cache.invalidate();
                    this.reset_discovery();
                    let authenticate = this.authenticate_and_refresh(false, cx);
                    this.settings_authentication_task =
                        cx.spawn(async move |_, _| match authenticate.await {
                            Ok(()) | Err(AuthenticateError::CredentialsNotFound) => {}
                            Err(error) => {
                                log::warn!(
                                    "Failed to apply updated Amazon Bedrock settings: {error}"
                                );
                            }
                        });
                }
                cx.notify();
            }),
        });

        Self {
            http_client: aws_http_client,
            plain_http_client: http_client,
            handle: Tokio::handle(cx),
            state,
        }
    }

    fn create_language_model(&self, model: bedrock::ConverseModel) -> Arc<dyn LanguageModel> {
        Arc::new(BedrockModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            http_client: self.http_client.clone(),
            handle: self.handle.clone(),
            state: self.state.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn create_mantle_language_model(&self, model: bedrock::MantleModel) -> Arc<dyn LanguageModel> {
        Arc::new(BedrockMantleModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            http_client: self.plain_http_client.clone(),
            aws_http_client: self.http_client.clone(),
            state: self.state.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn discovered_runtime_model(
        &self,
        base_model_id: &str,
        cx: &App,
    ) -> Option<Arc<dyn LanguageModel>> {
        let state = self.state.read(cx);
        let target = state
            .runtime_targets
            .iter()
            .find(|target| target.base_model_id == base_model_id)
            .cloned()?;
        Some(self.create_language_model(target.into_converse_model()))
    }
}

impl LanguageModelProvider for BedrockLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiBedrock)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let model = bedrock::ConverseModel::default();
        self.discovered_runtime_model(model.request_id(), cx)
            .or_else(|| Some(self.create_language_model(model)))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let region = self.state.read(cx).get_region();
        let model = bedrock::ConverseModel::default_fast(region.as_str());
        self.discovered_runtime_model(model.request_id(), cx)
            .or_else(|| Some(self.create_language_model(model)))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let bedrock_settings = &AllLanguageModelSettings::get_global(cx).bedrock;
        let mut models = BTreeMap::default();

        let state = self.state.read(cx);
        let runtime_targets = &state.runtime_targets;
        if runtime_targets.is_empty() {
            for model in bedrock::ConverseModel::iter() {
                if !matches!(
                    model,
                    bedrock::ConverseModel::Custom { .. }
                        | bedrock::ConverseModel::Discovered { .. }
                ) {
                    models.insert(model.id().to_string(), model);
                }
            }
        } else {
            for target in runtime_targets.iter().cloned() {
                let model = target.into_converse_model();
                models.insert(model.id().to_string(), model);
            }
        }

        // Override with available models from settings
        for model in bedrock_settings.available_models.iter() {
            models.insert(
                model.name.clone(),
                bedrock::ConverseModel::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                    default_temperature: model.default_temperature,
                    cache_configuration: model.cache_configuration.as_ref().map(|config| {
                        bedrock::BedrockModelCacheConfiguration {
                            max_cache_anchors: config.max_cache_anchors,
                            min_total_token: config.min_total_token,
                        }
                    }),
                },
            );
        }

        let mut models: Vec<Arc<dyn LanguageModel>> = models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect();

        let mut mantle_models = BTreeMap::default();

        if state.mantle_targets.is_empty() {
            for model in bedrock::MantleModel::iter().filter(|model| {
                !matches!(
                    model,
                    bedrock::MantleModel::Custom { .. } | bedrock::MantleModel::Discovered { .. }
                )
            }) {
                if state.mantle_endpoint_available != Some(false) {
                    mantle_models.insert(model.id().to_string(), model);
                }
            }
        } else {
            for target in state.mantle_targets.iter().cloned() {
                let model = target.into_mantle_model();
                mantle_models.insert(model.id().to_string(), model);
            }
        }

        // Override with available Mantle models from settings
        for model in bedrock_settings.mantle_available_models.iter() {
            mantle_models.insert(
                model.name.clone(),
                bedrock::MantleModel::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                    protocol: mantle_protocol_from_settings(model.protocol),
                    supports_tools: model.supports_tools.unwrap_or(false),
                    supports_images: model.supports_images.unwrap_or(false),
                    supports_thinking: model.supports_thinking.unwrap_or(false),
                },
            );
        }

        models.extend(
            mantle_models
                .into_values()
                .map(|model| self.create_mantle_language_model(model)),
        );

        models
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state
            .update(cx, |state, cx| state.authenticate_and_refresh(true, cx))
    }

    fn settings_view(&self, _cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.clone();
        Some(ProviderSettingsView::SubPage(
            SubPageProviderSettings::new(move |window, cx| {
                cx.new(|cx| ConfigurationView::new(state.clone(), window, cx))
                    .into()
            })
            .description(InlineDescription::Text(
                "To use Zed's agent with Bedrock, set a custom authentication strategy in your settings or use static credentials. Mantle-only models (e.g. GPT-5.5, GPT-5.4, Grok 4.3) additionally require IAM permissions for the `bedrock-mantle` endpoint.".into(),
            )),
        ))
    }
}

impl LanguageModelProviderState for BedrockLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

struct BedrockModel {
    id: LanguageModelId,
    model: ConverseModel,
    http_client: AwsHttpClient,
    handle: tokio::runtime::Handle,
    state: Entity<State>,
    request_limiter: RateLimiter,
}

impl BedrockModel {
    fn get_or_init_client(&self, cx: &AsyncApp) -> Result<BedrockClient> {
        let (session_cache, configuration) = cx.read_entity(&self.state, |state, _cx| {
            (
                state.aws_session_cache.clone(),
                state.aws_session_configuration(),
            )
        });
        let session = session_cache
            .cell
            .get_or_try_init_blocking(|| {
                self.handle
                    .block_on(build_aws_session(configuration, self.http_client.clone()))
            })
            .context("initializing Bedrock session")?;
        Ok(session.runtime_client.clone())
    }

    fn stream_completion(
        &self,
        request: bedrock::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<BoxStream<'static, Result<BedrockStreamingResponse, anyhow::Error>>, BedrockError>,
    > {
        let runtime_client = match self.get_or_init_client(cx) {
            Ok(client) => client,
            Err(error) => {
                return futures::future::ready(Err(BedrockError::Other(error))).boxed();
            }
        };
        let extra_headers = self.state.read_with(cx, |_, cx| {
            AllLanguageModelSettings::get_global(cx)
                .bedrock
                .custom_headers
                .clone()
        });

        let task = Tokio::spawn(
            cx,
            bedrock::stream_completion(runtime_client, request, extra_headers),
        );
        async move { task.await.map_err(|e| BedrockError::Other(e.into()))? }.boxed()
    }
}

impl LanguageModel for BedrockModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        self.model.supports_tool_use()
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_thinking(&self) -> bool {
        self.model.supports_thinking()
    }

    fn refusal_fallback_model_id(&self) -> Option<&'static str> {
        if self
            .model
            .id()
            .starts_with(anthropic::FABLE_MODEL_ID_PREFIX)
        {
            Some(anthropic::FABLE_FALLBACK_MODEL_ID)
        } else {
            None
        }
    }

    fn supported_effort_levels(&self) -> Vec<language_model::LanguageModelEffortLevel> {
        if self.model.supports_adaptive_thinking() {
            vec![
                language_model::LanguageModelEffortLevel {
                    name: "Low".into(),
                    value: "low".into(),
                    is_default: false,
                },
                language_model::LanguageModelEffortLevel {
                    name: "Medium".into(),
                    value: "medium".into(),
                    is_default: false,
                },
                language_model::LanguageModelEffortLevel {
                    name: "High".into(),
                    value: "high".into(),
                    is_default: true,
                },
                language_model::LanguageModelEffortLevel {
                    name: "XHigh".into(),
                    value: "xhigh".into(),
                    is_default: false,
                },
                language_model::LanguageModelEffortLevel {
                    name: "Max".into(),
                    value: "max".into(),
                    is_default: false,
                },
            ]
            .into_iter()
            .filter(|effort_level| {
                effort_level.value != "xhigh" || self.model.supports_xhigh_adaptive_thinking()
            })
            .collect()
        } else {
            Vec::new()
        }
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto | LanguageModelToolChoice::Any => {
                self.model.supports_tool_use()
            }
            // Add support for None - we'll filter tool calls at response
            LanguageModelToolChoice::None => self.model.supports_tool_use(),
        }
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn telemetry_id(&self) -> String {
        format!("bedrock/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        Some(self.model.max_output_tokens())
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        if request.contains_custom_tool_input() {
            return async move {
                Err(anyhow::anyhow!("Bedrock does not support custom tools").into())
            }
            .boxed();
        }

        let (region, allow_global, guardrail_identifier, guardrail_version) =
            cx.read_entity(&self.state, |state, _cx| {
                let (gid, gv) = state.get_guardrail_config();
                (state.get_region(), state.get_allow_global(), gid, gv)
            });

        let model_id = match self.model.cross_region_inference_id(&region, allow_global) {
            Ok(s) => s,
            Err(e) => {
                return async move { Err(e.into()) }.boxed();
            }
        };

        let deny_tool_calls = request.tool_choice == Some(LanguageModelToolChoice::None);

        let request = match into_bedrock(
            request,
            model_id,
            self.model.default_temperature(),
            self.model.max_output_tokens(),
            self.model.thinking_mode(),
            self.model.supports_caching(),
            self.model.supports_tool_use(),
            guardrail_identifier,
            guardrail_version,
        ) {
            Ok(request) => request,
            Err(err) => return futures::future::ready(Err(err.into())).boxed(),
        };

        let request = self.stream_completion(request, cx);
        let display_name = self.model.display_name().to_string();
        let future = self.request_limiter.stream(async move {
            let response = request.await.map_err(|err| match err {
                BedrockError::Validation(ref msg) => {
                    if msg.contains("model identifier is invalid") {
                        LanguageModelCompletionError::Other(anyhow!(
                            "{display_name} is not available in {region}. \
                                 Try switching to a region where this model is supported."
                        ))
                    } else {
                        LanguageModelCompletionError::BadRequestFormat {
                            provider: PROVIDER_NAME,
                            message: msg.clone(),
                        }
                    }
                }
                BedrockError::RateLimited => LanguageModelCompletionError::RateLimitExceeded {
                    provider: PROVIDER_NAME,
                    retry_after: None,
                },
                BedrockError::ServiceUnavailable => {
                    LanguageModelCompletionError::ServerOverloaded {
                        provider: PROVIDER_NAME,
                        retry_after: None,
                    }
                }
                BedrockError::AccessDenied(msg) => LanguageModelCompletionError::PermissionError {
                    provider: PROVIDER_NAME,
                    message: msg,
                },
                BedrockError::InternalServer(msg) => {
                    LanguageModelCompletionError::ApiInternalServerError {
                        provider: PROVIDER_NAME,
                        message: msg,
                    }
                }
                other => LanguageModelCompletionError::Other(anyhow!(other)),
            })?;
            let events = map_to_language_model_completion_events(response);

            if deny_tool_calls {
                Ok(deny_tool_use_events(events).boxed())
            } else {
                Ok(events.boxed())
            }
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

const MANTLE_SELECTABLE_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::XHigh,
];

fn mantle_default_reasoning_effort(model: &MantleModel) -> Option<ReasoningEffort> {
    model.supports_thinking().then_some(ReasoningEffort::Medium)
}

fn mantle_selected_reasoning_effort(
    request: &LanguageModelRequest,
    model: &MantleModel,
) -> Option<ReasoningEffort> {
    if !model.supports_thinking() {
        return None;
    }

    if request.thinking_allowed {
        request
            .thinking_effort
            .as_deref()
            .and_then(|effort| effort.parse::<ReasoningEffort>().ok())
            .filter(|effort| *effort != ReasoningEffort::None)
            .or_else(|| mantle_default_reasoning_effort(model))
    } else {
        Some(ReasoningEffort::None)
    }
}

fn mantle_supported_effort_levels(model: &MantleModel) -> Vec<LanguageModelEffortLevel> {
    let Some(default_effort) = mantle_default_reasoning_effort(model) else {
        return Vec::new();
    };

    MANTLE_SELECTABLE_REASONING_EFFORTS
        .iter()
        .copied()
        .map(|effort| LanguageModelEffortLevel {
            name: effort.label().into(),
            value: effort.value().into(),
            is_default: effort == default_effort,
        })
        .collect()
}

fn into_mantle_anthropic(
    request: LanguageModelRequest,
    model: &MantleModel,
) -> Result<anthropic::Request> {
    let mode = if model.supports_thinking() {
        anthropic::AnthropicModelMode::AdaptiveThinking
    } else {
        anthropic::AnthropicModelMode::Default
    };
    let mut request = into_anthropic(
        request,
        model.request_id().to_string(),
        1.0,
        model.max_output_tokens(),
        mode,
        AnthropicPromptCacheMode::Automatic,
    )?;
    if matches!(
        model,
        MantleModel::ClaudeOpus4_7 | MantleModel::ClaudeOpus4_8
    ) {
        request.temperature = None;
    }
    Ok(request)
}

/// Special-cases Mantle authorization failures with a message that points at
/// the separate `bedrock-mantle` IAM policy namespace instead of regular
/// `bedrock-runtime` permissions.
fn map_mantle_error(model: &MantleModel, error: RequestError) -> LanguageModelCompletionError {
    if let RequestError::HttpResponseError { status_code, .. } = &error
        && *status_code == http_client::http::StatusCode::FORBIDDEN
    {
        return LanguageModelCompletionError::PermissionError {
            provider: PROVIDER_NAME,
            message: format!(
                "Bedrock Mantle denied this request for {}. Mantle-only models require IAM \
                 permissions for the `bedrock-mantle` endpoint (for example via the \
                 `AmazonBedrockMantleInferenceAccess` managed policy) in addition to whatever \
                 permissions your existing Bedrock credentials already have.",
                model.display_name()
            ),
        };
    }
    error.into()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MantleChatStreamResult {
    Ok(ResponseStreamEvent),
    Err { error: MantleChatStreamError },
}

#[derive(Deserialize)]
struct MantleChatStreamError {
    message: String,
}

fn parse_mantle_chat_stream_line(line: &str) -> Result<ResponseStreamEvent> {
    match serde_json::from_str(line) {
        Ok(MantleChatStreamResult::Ok(response)) => Ok(response),
        Ok(MantleChatStreamResult::Err { error }) => Err(anyhow!(error.message)),
        Err(error) => Err(anyhow!(
            "failed to parse Mantle chat completion event: {error}"
        )),
    }
}

fn parse_mantle_response_stream_line(line: &str) -> Result<open_ai::responses::StreamEvent> {
    serde_json::from_str(line)
        .map_err(|error| anyhow!("failed to parse Mantle responses event: {error}"))
}

fn parse_mantle_anthropic_stream_line(line: &str) -> Result<anthropic::Event> {
    serde_json::from_str(line)
        .map_err(|error| anyhow!("failed to parse Mantle Anthropic event: {error}"))
}

fn build_mantle_http_request(
    method: Method,
    url: &str,
    body: &[u8],
    region: &str,
    auth: &MantleAuth,
    extra_headers: &CustomHeaders,
    required_headers: &'static [(&'static str, &'static str)],
) -> Result<HttpRequest<AsyncBody>> {
    let mut request = HttpRequest::builder()
        .method(method)
        .uri(url)
        .header("Content-Type", "application/json")
        .extra_headers(extra_headers)
        .body(AsyncBody::from(body.to_vec()))?;
    for (name, value) in required_headers {
        request.headers_mut().insert(
            http_client::http::header::HeaderName::from_static(name),
            HeaderValue::from_static(value),
        );
    }
    auth.apply(&mut request, body, region)?;
    Ok(request)
}

async fn stream_mantle_sse<Request, Event>(
    client: &dyn HttpClient,
    provider_name: &str,
    url: &str,
    region: &str,
    auth: &MantleAuth,
    request: Request,
    extra_headers: &CustomHeaders,
    required_headers: &'static [(&'static str, &'static str)],
    parse_stream_line: fn(&str) -> Result<Event>,
) -> std::result::Result<BoxStream<'static, Result<Event>>, RequestError>
where
    Request: Serialize,
    Event: Send + 'static,
{
    let body = serde_json::to_vec(&request).map_err(|error| RequestError::Other(error.into()))?;
    let request = build_mantle_http_request(
        Method::POST,
        url,
        &body,
        region,
        auth,
        extra_headers,
        required_headers,
    )
    .map_err(RequestError::Other)?;

    let mut response = client.send(request).await?;
    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        Ok(reader
            .lines()
            .filter_map(move |line| async move {
                match line {
                    Ok(line) => {
                        let line = line
                            .strip_prefix("data: ")
                            .or_else(|| line.strip_prefix("data:"))?;
                        if line == "[DONE]" || line.is_empty() {
                            None
                        } else {
                            Some(parse_stream_line(line))
                        }
                    }
                    Err(error) => Some(Err(anyhow!(error))),
                }
            })
            .boxed())
    } else {
        let mut body = String::new();
        response
            .body_mut()
            .read_to_string(&mut body)
            .await
            .map_err(|error| RequestError::Other(error.into()))?;

        Err(RequestError::HttpResponseError {
            provider: provider_name.to_owned(),
            status_code: response.status(),
            body,
            headers: response.headers().clone(),
        })
    }
}

fn strip_unsupported_mantle_response_fields(request: &mut OpenAiResponseRequest) {
    request.context_management = None;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum MantleMessageSnapshotKind {
    Undetermined,
    Cumulative,
    Independent,
}

struct MantleMessageSnapshot {
    item_id: String,
    text: String,
    emitted_text_length: usize,
    kind: MantleMessageSnapshotKind,
    phase: Option<String>,
    output_index: usize,
    content_index: Option<usize>,
}

/// The most recently completed message item, used to detect Mantle's cumulative
/// snapshot replays.
#[derive(Clone)]
struct MantlePreviousMessage {
    text: String,
    phase: Option<String>,
}

/// Adapts Mantle's cumulative Responses message snapshots to incremental events.
///
/// Mantle can emit a new message item containing all text produced so far, then
/// replay the corresponding output deltas. The shared OpenAI mapper assumes each
/// delta is new text, so forwarding those events directly duplicates the reply.
///
/// Only adjacent, same-phase, strict-prefix extensions are collapsed. Equal,
/// shrinking, divergent, or different-phase messages stay visible. Any non-message
/// output item (reasoning, tool call, etc.) is a hard boundary: it is forwarded
/// unchanged and clears the snapshot used for collapsing, so a message after a
/// tool call is never merged with the one before it.
struct MantleResponseEventMapper {
    open_ai_mapper: OpenAiResponseEventMapper,
    current_message: Option<MantleMessageSnapshot>,
    previous_message: Option<MantlePreviousMessage>,
    pending_message_events: Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
}

impl MantleResponseEventMapper {
    fn new() -> Self {
        Self {
            open_ai_mapper: OpenAiResponseEventMapper::new(),
            current_message: None,
            previous_message: None,
            pending_message_events: Vec::new(),
        }
    }

    fn map_stream(
        self,
        events: BoxStream<'static, Result<OpenAiResponseStreamEvent>>,
    ) -> BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        futures::stream::unfold(
            (self, events, VecDeque::new(), false),
            |(mut mapper, mut events, mut pending, mut finished)| async move {
                loop {
                    if let Some(event) = pending.pop_front() {
                        return Some((event, (mapper, events, pending, finished)));
                    }
                    if finished {
                        return None;
                    }

                    match events.next().await {
                        Some(Ok(event)) => pending.extend(mapper.map_event(event)),
                        Some(Err(error)) => {
                            pending.extend(mapper.finish_current_message());
                            pending.push_back(Err(LanguageModelCompletionError::from(error)));
                            finished = true;
                        }
                        None => {
                            pending.extend(mapper.finish_current_message());
                            finished = true;
                        }
                    }
                }
            },
        )
        .boxed()
    }

    fn map_event(
        &mut self,
        event: OpenAiResponseStreamEvent,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        match event {
            OpenAiResponseStreamEvent::OutputItemAdded {
                output_index,
                sequence_number,
                item,
            } => match &item {
                ResponseOutputItem::Message(message) => {
                    let mut events = self.finish_current_message();
                    let item_id = message.id.clone();
                    let phase = message.phase.clone();
                    if let Some(item_id) = item_id {
                        self.current_message = Some(MantleMessageSnapshot {
                            item_id,
                            text: String::new(),
                            emitted_text_length: 0,
                            kind: MantleMessageSnapshotKind::Undetermined,
                            phase,
                            output_index,
                            content_index: None,
                        });

                        self.map_message_added(OpenAiResponseStreamEvent::OutputItemAdded {
                            output_index,
                            sequence_number,
                            item,
                        });
                    } else {
                        self.flush_pending_message(&mut events, false);
                        events.extend(self.open_ai_mapper.map_event(
                            OpenAiResponseStreamEvent::OutputItemAdded {
                                output_index,
                                sequence_number,
                                item,
                            },
                        ));
                    }
                    events
                }
                _ => {
                    let mut events = self.finish_current_message();
                    self.previous_message = None;
                    events.extend(self.open_ai_mapper.map_event(
                        OpenAiResponseStreamEvent::OutputItemAdded {
                            output_index,
                            sequence_number,
                            item,
                        },
                    ));
                    events
                }
            },
            OpenAiResponseStreamEvent::OutputTextDelta {
                item_id,
                output_index,
                content_index,
                delta,
            } => self.map_text_delta(item_id, output_index, content_index, delta),
            OpenAiResponseStreamEvent::OutputTextDone {
                item_id,
                output_index,
                content_index,
                text,
            } => {
                let suffix = self
                    .current_message
                    .as_ref()
                    .filter(|message| message.item_id == item_id)
                    .and_then(|message| text.strip_prefix(&message.text))
                    .map(str::to_string);
                if let Some(suffix) = suffix {
                    if suffix.is_empty() {
                        return Vec::new();
                    }
                    return self.map_text_delta(item_id, output_index, content_index, suffix);
                }

                let mut events = self.finish_current_message();
                events.extend(self.open_ai_mapper.map_event(
                    OpenAiResponseStreamEvent::OutputTextDone {
                        item_id,
                        output_index,
                        content_index,
                        text,
                    },
                ));
                events
            }
            OpenAiResponseStreamEvent::ContentPartAdded { ref item_id, .. }
            | OpenAiResponseStreamEvent::ContentPartDone { ref item_id, .. }
            | OpenAiResponseStreamEvent::RefusalDelta { ref item_id, .. }
            | OpenAiResponseStreamEvent::RefusalDone { ref item_id, .. }
                if self
                    .current_message
                    .as_ref()
                    .is_some_and(|message| &message.item_id == item_id) =>
            {
                self.open_ai_mapper.map_event(event)
            }
            OpenAiResponseStreamEvent::OutputItemDone {
                output_index,
                sequence_number,
                item,
            } => {
                if let ResponseOutputItem::Message(message) = &item
                    && let Some(item_id) = message.id.as_ref()
                    && let Some(current_message) = self.current_message.as_mut()
                    && &current_message.item_id == item_id
                {
                    current_message.phase = message.phase.clone();
                }

                let is_non_message = !matches!(item, ResponseOutputItem::Message(_));
                let was_merged = matches!(
                    self.current_message.as_ref(),
                    Some(current_message)
                        if current_message.kind == MantleMessageSnapshotKind::Cumulative
                );
                let mut events = self.finish_current_message();
                if is_non_message {
                    self.previous_message = None;
                }
                let done_events =
                    self.open_ai_mapper
                        .map_event(OpenAiResponseStreamEvent::OutputItemDone {
                            output_index,
                            sequence_number,
                            item,
                        });
                if was_merged {
                    // The message was merged into the previous one, so the phase/
                    // reasoning metadata this `OutputItemDone` would otherwise emit
                    // again is already reflected by the metadata emitted earlier.
                    events.extend(done_events.into_iter().filter(|event| {
                        !matches!(event, Ok(LanguageModelCompletionEvent::ReasoningDetails(_)))
                    }));
                } else {
                    events.extend(done_events);
                }
                events
            }
            event => {
                let mut events = self.finish_current_message();
                events.extend(self.open_ai_mapper.map_event(event));
                events
            }
        }
    }

    fn map_message_added(&mut self, event: OpenAiResponseStreamEvent) {
        debug_assert!(
            self.pending_message_events.is_empty(),
            "pending message events must be flushed before a new message is added"
        );
        self.pending_message_events = self.open_ai_mapper.map_event(event);
    }

    fn map_text_delta(
        &mut self,
        item_id: String,
        output_index: usize,
        content_index: Option<usize>,
        delta: String,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        if !self
            .current_message
            .as_ref()
            .is_some_and(|current_message| current_message.item_id == item_id)
        {
            let mut events = self.finish_current_message();
            events.extend(self.open_ai_mapper.map_event(
                OpenAiResponseStreamEvent::OutputTextDelta {
                    item_id,
                    output_index,
                    content_index,
                    delta,
                },
            ));
            return events;
        }

        let Some(current_message) = self.current_message.as_mut() else {
            return Vec::new();
        };
        current_message.text.push_str(&delta);
        current_message.output_index = output_index;
        current_message.content_index = content_index;
        let snapshot = current_message.text.clone();
        let kind = current_message.kind;
        let phase = current_message.phase.clone();
        let previous = self.previous_message.clone();

        match kind {
            MantleMessageSnapshotKind::Undetermined => {
                if let Some(previous) = previous
                    && previous.phase == phase
                {
                    // Strict same-phase extension: previous text is a proper prefix of
                    // the new snapshot. Collapse the replayed prefix and emit only the
                    // suffix.
                    if snapshot.starts_with(&previous.text) && snapshot.len() > previous.text.len()
                    {
                        if let Some(current_message) = self.current_message.as_mut() {
                            current_message.kind = MantleMessageSnapshotKind::Cumulative;
                            current_message.emitted_text_length = previous.text.len();
                        }
                        let mut events = Vec::new();
                        self.flush_pending_message(&mut events, true);
                        events.extend(self.map_snapshot_suffix(
                            output_index,
                            content_index,
                            snapshot,
                        ));
                        return events;
                    }

                    // The snapshot is still a prefix of (or equal to) the previous
                    // message. A genuine cumulative replay keeps growing past the
                    // previous text, so defer the decision and withhold emission
                    // until it either exceeds the previous text or diverges.
                    if previous.text.starts_with(&snapshot) {
                        return Vec::new();
                    }
                }

                // Diverged from the previous message, had a different phase, or there
                // was no previous message: this is an independent message.
                if let Some(current_message) = self.current_message.as_mut() {
                    current_message.kind = MantleMessageSnapshotKind::Independent;
                }
                let mut events = Vec::new();
                self.flush_pending_message(&mut events, false);
                events.extend(self.map_text_event(item_id, output_index, content_index, snapshot));
                events
            }
            MantleMessageSnapshotKind::Cumulative => {
                self.map_snapshot_suffix(output_index, content_index, snapshot)
            }
            MantleMessageSnapshotKind::Independent => {
                self.map_text_event(item_id, output_index, content_index, delta)
            }
        }
    }

    fn map_snapshot_suffix(
        &mut self,
        output_index: usize,
        content_index: Option<usize>,
        snapshot: String,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        let (item_id, suffix) = {
            let Some(current_message) = self.current_message.as_mut() else {
                return Vec::new();
            };

            if snapshot.len() <= current_message.emitted_text_length {
                return Vec::new();
            }

            let suffix = snapshot[current_message.emitted_text_length..].to_string();
            current_message.emitted_text_length = snapshot.len();
            (current_message.item_id.clone(), suffix)
        };

        self.map_text_event(item_id, output_index, content_index, suffix)
    }

    fn map_text_event(
        &mut self,
        item_id: String,
        output_index: usize,
        content_index: Option<usize>,
        delta: String,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        if delta.is_empty() {
            return Vec::new();
        }

        self.open_ai_mapper
            .map_event(OpenAiResponseStreamEvent::OutputTextDelta {
                item_id,
                output_index,
                content_index,
                delta,
            })
    }

    fn finish_current_message(
        &mut self,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        let mut events = Vec::new();
        self.finish_current_message_into(&mut events);
        events
    }

    fn finish_current_message_into(
        &mut self,
        events: &mut Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
    ) {
        let Some(current_message) = self.current_message.take() else {
            self.flush_pending_message(events, false);
            return;
        };

        match current_message.kind {
            MantleMessageSnapshotKind::Cumulative => {
                self.flush_pending_message(events, true);
            }
            MantleMessageSnapshotKind::Independent => {
                self.flush_pending_message(events, false);
            }
            MantleMessageSnapshotKind::Undetermined => {
                // The message never exceeded or diverged from the previous one, so
                // it could not be confirmed as a cumulative replay. Treat it as an
                // independent message: emit the withheld StartMessage and any text
                // accumulated while deferring the decision.
                self.flush_pending_message(events, false);
                if !current_message.text.is_empty() {
                    events.extend(self.map_text_event(
                        current_message.item_id.clone(),
                        current_message.output_index,
                        current_message.content_index,
                        current_message.text.clone(),
                    ));
                }
            }
        }

        if !current_message.text.is_empty() {
            self.previous_message = Some(MantlePreviousMessage {
                text: current_message.text,
                phase: current_message.phase,
            });
        }
    }

    fn flush_pending_message(
        &mut self,
        events: &mut Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
        suppress_start_message: bool,
    ) {
        for event in self.pending_message_events.drain(..) {
            // When merging into the previous message, the phase/reasoning metadata
            // can't have changed since it was last emitted (a merge only happens for
            // adjacent, same-phase messages with no reasoning item in between), so
            // suppress both the StartMessage and the now-redundant ReasoningDetails.
            if suppress_start_message
                && matches!(
                    event,
                    Ok(LanguageModelCompletionEvent::StartMessage { .. })
                        | Ok(LanguageModelCompletionEvent::ReasoningDetails(_))
                )
            {
                continue;
            }
            events.push(event);
        }
    }
}

struct BedrockMantleModel {
    id: LanguageModelId,
    model: MantleModel,
    http_client: Arc<dyn HttpClient>,
    aws_http_client: AwsHttpClient,
    state: Entity<State>,
    request_limiter: RateLimiter,
}

impl BedrockMantleModel {
    fn stream_mantle_request<Request, Event>(
        &self,
        request: Request,
        cx: &AsyncApp,
        endpoint: &'static str,
        required_headers: &'static [(&'static str, &'static str)],
        parse_stream_line: fn(&str) -> Result<Event>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<Event>>, LanguageModelCompletionError>>
    where
        Request: Serialize + Send + 'static,
        Event: Send + 'static,
    {
        let http_client = self.http_client.clone();
        let aws_http_client = self.aws_http_client.clone();
        let model = self.model.clone();
        let (session_cache, configuration, region) = cx.read_entity(&self.state, |state, _cx| {
            (
                state.aws_session_cache.clone(),
                state.aws_session_configuration(),
                state.get_region(),
            )
        });
        let url = mantle_endpoint_url(&region, endpoint);
        let extra_headers = cx.read_entity(&self.state, |_, cx| {
            AllLanguageModelSettings::get_global(cx)
                .bedrock
                .custom_headers
                .clone()
        });
        let provider_name = PROVIDER_NAME.0.to_string();
        let auth_task = Tokio::spawn_result(cx, async move {
            let session = session_cache
                .cell
                .get_or_try_init(|| build_aws_session(configuration, aws_http_client))
                .await
                .context("initializing Bedrock session")?;
            session.resolve_mantle_auth().await
        });

        let future = self.request_limiter.stream(async move {
            let auth = auth_task
                .await
                .map_err(LanguageModelCompletionError::Other)?;
            stream_mantle_sse(
                http_client.as_ref(),
                &provider_name,
                &url,
                &region,
                &auth,
                request,
                &extra_headers,
                required_headers,
                parse_stream_line,
            )
            .await
            .map_err(|err| map_mantle_error(&model, err))
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }

    fn stream_completion(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<BoxStream<'static, Result<ResponseStreamEvent>>, LanguageModelCompletionError>,
    > {
        self.stream_mantle_request(
            request,
            cx,
            "openai/v1/chat/completions",
            &[],
            parse_mantle_chat_stream_line,
        )
    }

    fn stream_response(
        &self,
        request: OpenAiResponseRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<open_ai::responses::StreamEvent>>,
            LanguageModelCompletionError,
        >,
    > {
        let mut request = request;
        strip_unsupported_mantle_response_fields(&mut request);
        self.stream_mantle_request(
            request,
            cx,
            "openai/v1/responses",
            &[],
            parse_mantle_response_stream_line,
        )
    }

    fn stream_anthropic(
        &self,
        request: anthropic::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<BoxStream<'static, Result<anthropic::Event>>, LanguageModelCompletionError>,
    > {
        self.stream_mantle_request(
            anthropic::StreamingRequest {
                base: request,
                stream: true,
            },
            cx,
            "anthropic/v1/messages",
            ANTHROPIC_MANTLE_HEADERS,
            parse_mantle_anthropic_stream_line,
        )
    }
}

impl LanguageModel for BedrockMantleModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        self.model.supports_tools()
    }

    fn tool_input_format(&self) -> LanguageModelToolSchemaFormat {
        LanguageModelToolSchemaFormat::JsonSchemaSubset
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto | LanguageModelToolChoice::Any => {
                self.model.supports_tools()
            }
            LanguageModelToolChoice::None => true,
        }
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn supports_thinking(&self) -> bool {
        self.model.supports_thinking()
    }

    fn refusal_fallback_model_id(&self) -> Option<&'static str> {
        matches!(self.model, MantleModel::ClaudeFable5).then_some("bedrock-mantle/claude-opus-4-8")
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        mantle_supported_effort_levels(&self.model)
    }

    fn supports_split_token_display(&self) -> bool {
        true
    }

    fn telemetry_id(&self) -> String {
        format!("bedrock-mantle/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        Some(self.model.max_output_tokens())
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        let region = cx.read_entity(&self.state, |state, _cx| state.get_region());

        if !MANTLE_SUPPORTED_REGIONS.contains(&region.as_str()) {
            let display_name = self.model.display_name().to_string();
            let supported = MANTLE_SUPPORTED_REGIONS.join(", ");
            return futures::future::ready(Err(LanguageModelCompletionError::Other(anyhow!(
                "{display_name} is not available in {region} because Bedrock Mantle isn't offered \
                 there. Try switching to one of the following regions: {supported}."
            ))))
            .boxed();
        }

        let model_id = self.model.request_id().to_string();
        let max_output_tokens = Some(self.model.max_output_tokens());

        match self.model.protocol() {
            MantleProtocol::AnthropicMessages => {
                let request = match into_mantle_anthropic(request, &self.model) {
                    Ok(request) => request,
                    Err(error) => return async move { Err(error.into()) }.boxed(),
                };
                let completions = self.stream_anthropic(request, cx);
                async move {
                    let events = completions
                        .await?
                        .map(|event| event.map_err(anthropic::AnthropicError::HttpSend));
                    Ok(AnthropicEventMapper::new(PROVIDER_NAME)
                        .map_stream(events.boxed())
                        .boxed())
                }
                .boxed()
            }
            MantleProtocol::Responses => {
                let request = into_open_ai_response(
                    request,
                    &model_id,
                    self.model.supports_tools(),
                    false,
                    max_output_tokens,
                    mantle_default_reasoning_effort(&self.model),
                    self.model.supports_thinking(),
                );
                let completions = self.stream_response(request, cx);
                async move {
                    let mapper = MantleResponseEventMapper::new();
                    Ok(mapper.map_stream(completions.await?).boxed())
                }
                .boxed()
            }
            MantleProtocol::ChatCompletions => {
                let reasoning_effort = mantle_selected_reasoning_effort(&request, &self.model);
                let request = match into_open_ai(
                    request,
                    &model_id,
                    self.model.supports_tools(),
                    false,
                    max_output_tokens,
                    ChatCompletionMaxTokensParameter::MaxCompletionTokens,
                    reasoning_effort,
                    false,
                ) {
                    Ok(request) => request,
                    Err(error) => return async move { Err(error.into()) }.boxed(),
                };
                let completions = self.stream_completion(request, cx);
                async move {
                    let mapper = OpenAiEventMapper::new();
                    Ok(mapper.map_stream(completions.await?).boxed())
                }
                .boxed()
            }
        }
    }
}

fn deny_tool_use_events(
    events: impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
    events.map(|event| {
        match event {
            Ok(LanguageModelCompletionEvent::ToolUse(tool_use)) => {
                // Convert tool use to an error message if model decided to call it
                Ok(LanguageModelCompletionEvent::Text(format!(
                    "\n\n[Error: Tool calls are disabled in this context. Attempted to call '{}']",
                    tool_use.name
                )))
            }
            other => other,
        }
    })
}

pub fn into_bedrock(
    request: LanguageModelRequest,
    model: String,
    default_temperature: f32,
    max_output_tokens: u64,
    thinking_mode: BedrockModelMode,
    supports_caching: bool,
    supports_tool_use: bool,
    guardrail_identifier: Option<String>,
    guardrail_version: Option<String>,
) -> Result<bedrock::Request> {
    if request.contains_custom_tool_input() {
        anyhow::bail!("Bedrock does not support custom tools");
    }

    let mut new_messages: Vec<BedrockMessage> = Vec::new();
    let mut system_message = String::new();

    // Track whether messages contain tool content - Bedrock requires toolConfig
    // when tool blocks are present, so we may need to add a dummy tool
    let mut messages_contain_tool_content = false;

    for message in request.messages {
        if message.contents_empty() {
            continue;
        }

        match message.role {
            Role::User | Role::Assistant => {
                let mut message_content = message.content;
                if message.role == Role::User {
                    // Bedrock requires tool results to precede other blocks in a user turn.
                    message_content
                        .sort_by_key(|content| !matches!(content, MessageContent::ToolResult(_)));
                }
                let mut bedrock_message_content: Vec<BedrockInnerContent> = message_content
                    .into_iter()
                    .filter_map(|content| match content {
                        MessageContent::Text(text) => {
                            if !text.is_empty() {
                                Some(BedrockInnerContent::Text(text))
                            } else {
                                None
                            }
                        }
                        MessageContent::Compaction(_) => None,
                        MessageContent::Thinking { text, signature } => {
                            if model.contains(ConverseModel::DeepSeekR1.request_id()) {
                                // DeepSeekR1 doesn't support thinking blocks
                                // And the AWS API demands that you strip them
                                return None;
                            }
                            if signature.is_none() {
                                // Thinking blocks without a signature are invalid
                                // (e.g. from cancellation mid-think) and must be
                                // stripped to avoid API errors.
                                return None;
                            }
                            let thinking = BedrockThinkingTextBlock::builder()
                                .text(text)
                                .set_signature(signature)
                                .build()
                                .context("failed to build reasoning block")
                                .log_err()?;

                            Some(BedrockInnerContent::ReasoningContent(
                                BedrockThinkingBlock::ReasoningText(thinking),
                            ))
                        }
                        MessageContent::RedactedThinking(blob) => {
                            if model.contains(ConverseModel::DeepSeekR1.request_id()) {
                                // DeepSeekR1 doesn't support thinking blocks
                                // And the AWS API demands that you strip them
                                return None;
                            }
                            let redacted =
                                BedrockThinkingBlock::RedactedContent(BedrockBlob::new(blob));

                            Some(BedrockInnerContent::ReasoningContent(redacted))
                        }
                        MessageContent::ToolUse(tool_use) => {
                            messages_contain_tool_content = true;
                            let input =
                                if let language_model::LanguageModelToolUseInput::Json(input) =
                                    &tool_use.input
                                {
                                    if input.is_null() {
                                        // Bedrock API requires valid JsonValue, not null, for tool use input
                                        value_to_aws_document(&serde_json::json!({}))
                                    } else {
                                        value_to_aws_document(input)
                                    }
                                } else {
                                    value_to_aws_document(&serde_json::json!({}))
                                };
                            BedrockToolUseBlock::builder()
                                .name(tool_use.name.to_string())
                                .tool_use_id(tool_use.id.to_string())
                                .input(input)
                                .build()
                                .context("failed to build Bedrock tool use block")
                                .log_err()
                                .map(BedrockInnerContent::ToolUse)
                        }
                        MessageContent::ToolResult(tool_result) => {
                            messages_contain_tool_content = true;
                            let mut builder = BedrockToolResultBlock::builder()
                                .tool_use_id(tool_result.tool_use_id.to_string());
                            for part in tool_result.content {
                                let block = match part {
                                    LanguageModelToolResultContent::Text(text) => {
                                        BedrockToolResultContentBlock::Text(text.to_string())
                                    }
                                    LanguageModelToolResultContent::Image(image) => {
                                        use base64::Engine;

                                        match base64::engine::general_purpose::STANDARD
                                            .decode(image.source.as_bytes())
                                        {
                                            Ok(image_bytes) => {
                                                match BedrockImageBlock::builder()
                                                    .format(BedrockImageFormat::Png)
                                                    .source(BedrockImageSource::Bytes(
                                                        BedrockBlob::new(image_bytes),
                                                    ))
                                                    .build()
                                                {
                                                    Ok(image_block) => {
                                                        BedrockToolResultContentBlock::Image(
                                                            image_block,
                                                        )
                                                    }
                                                    Err(err) => {
                                                        BedrockToolResultContentBlock::Text(
                                                            format!(
                                                                "[Failed to build image block: {}]",
                                                                err
                                                            ),
                                                        )
                                                    }
                                                }
                                            }
                                            Err(err) => {
                                                BedrockToolResultContentBlock::Text(format!(
                                                    "[Failed to decode tool result image: {}]",
                                                    err
                                                ))
                                            }
                                        }
                                    }
                                };
                                builder = builder.content(block);
                            }
                            builder
                                .status({
                                    if tool_result.is_error {
                                        BedrockToolResultStatus::Error
                                    } else {
                                        BedrockToolResultStatus::Success
                                    }
                                })
                                .build()
                                .context("failed to build Bedrock tool result block")
                                .log_err()
                                .map(BedrockInnerContent::ToolResult)
                        }
                        MessageContent::Image(image) => {
                            use base64::Engine;

                            let image_bytes = base64::engine::general_purpose::STANDARD
                                .decode(image.source.as_bytes())
                                .context("failed to decode base64 image data")
                                .log_err()?;

                            BedrockImageBlock::builder()
                                .format(BedrockImageFormat::Png)
                                .source(BedrockImageSource::Bytes(BedrockBlob::new(image_bytes)))
                                .build()
                                .context("failed to build Bedrock image block")
                                .log_err()
                                .map(BedrockInnerContent::Image)
                        }
                    })
                    .collect();
                if message.cache && supports_caching && !bedrock_message_content.is_empty() {
                    bedrock_message_content.push(BedrockInnerContent::CachePoint(
                        CachePointBlock::builder()
                            .r#type(CachePointType::Default)
                            .build()
                            .context("failed to build cache point block")?,
                    ));
                }
                let bedrock_role = match message.role {
                    Role::User => bedrock::BedrockRole::User,
                    Role::Assistant => bedrock::BedrockRole::Assistant,
                    Role::System => unreachable!("System role should never occur here"),
                };
                if bedrock_message_content.is_empty() {
                    continue;
                }

                if let Some(last_message) = new_messages.last_mut()
                    && last_message.role == bedrock_role
                {
                    last_message.content.extend(bedrock_message_content);
                    continue;
                }
                new_messages.push(
                    BedrockMessage::builder()
                        .role(bedrock_role)
                        .set_content(Some(bedrock_message_content))
                        .build()
                        .context("failed to build Bedrock message")?,
                );
            }
            Role::System => {
                if !system_message.is_empty() {
                    system_message.push_str("\n\n");
                }
                system_message.push_str(&message.string_contents());
            }
        }
    }

    let mut tool_spec: Vec<BedrockTool> = if supports_tool_use {
        request
            .tools
            .iter()
            .map(|tool| {
                let language_model::LanguageModelRequestToolInput::Function {
                    input_schema, ..
                } = &tool.input
                else {
                    anyhow::bail!("Bedrock does not support custom tools");
                };
                Ok(BedrockTool::ToolSpec(
                    BedrockToolSpec::builder()
                        .name(tool.name.clone())
                        .description(tool.description.clone())
                        .input_schema(BedrockToolInputSchema::Json(value_to_aws_document(
                            input_schema,
                        )))
                        .build()
                        .context("failed to build Bedrock tool spec")?,
                ))
            })
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };

    // Bedrock requires toolConfig when messages contain tool use/result blocks.
    // If no tools are defined but messages contain tool content (e.g., when
    // summarising a conversation that used tools), add a dummy tool to satisfy
    // the API requirement.
    if supports_tool_use && tool_spec.is_empty() && messages_contain_tool_content {
        tool_spec.push(BedrockTool::ToolSpec(
            BedrockToolSpec::builder()
                .name("_placeholder")
                .description("Placeholder tool to satisfy Bedrock API requirements when conversation history contains tool usage")
                .input_schema(BedrockToolInputSchema::Json(value_to_aws_document(
                    &serde_json::json!({"type": "object", "properties": {}}),
                )))
                .build()
                .context("failed to build placeholder tool spec")?,
        ));
    }

    if !tool_spec.is_empty() && supports_caching {
        tool_spec.push(BedrockTool::CachePoint(
            CachePointBlock::builder()
                .r#type(CachePointType::Default)
                .build()
                .context("failed to build cache point block")?,
        ));
    }

    let tool_choice = match request.tool_choice {
        Some(LanguageModelToolChoice::Auto) | None => {
            BedrockToolChoice::Auto(BedrockAutoToolChoice::builder().build())
        }
        Some(LanguageModelToolChoice::Any) => {
            BedrockToolChoice::Any(BedrockAnyToolChoice::builder().build())
        }
        Some(LanguageModelToolChoice::None) => {
            // For None, we still use Auto but will filter out tool calls in the response
            BedrockToolChoice::Auto(BedrockAutoToolChoice::builder().build())
        }
    };
    let tool_config = if tool_spec.is_empty() {
        None
    } else {
        Some(
            BedrockToolConfig::builder()
                .set_tools(Some(tool_spec))
                .tool_choice(tool_choice)
                .build()?,
        )
    };

    let mut system_blocks: Vec<BedrockSystemContentBlock> = Vec::new();
    if !system_message.is_empty() {
        system_blocks.push(BedrockSystemContentBlock::Text(system_message));
        if supports_caching {
            system_blocks.push(BedrockSystemContentBlock::CachePoint(
                CachePointBlock::builder()
                    .r#type(CachePointType::Default)
                    .build()
                    .context("failed to build system cache point block")?,
            ));
        }
    }

    Ok(bedrock::Request {
        model,
        messages: new_messages,
        max_tokens: max_output_tokens,
        system: system_blocks,
        tools: tool_config,
        thinking: if request.thinking_allowed {
            match thinking_mode {
                BedrockModelMode::Thinking { budget_tokens } => {
                    Some(bedrock::Thinking::Enabled { budget_tokens })
                }
                BedrockModelMode::AdaptiveThinking {
                    effort: default_effort,
                } => {
                    let effort = request
                        .thinking_effort
                        .as_deref()
                        .and_then(|e| match e {
                            "low" => Some(bedrock::BedrockAdaptiveThinkingEffort::Low),
                            "medium" => Some(bedrock::BedrockAdaptiveThinkingEffort::Medium),
                            "high" => Some(bedrock::BedrockAdaptiveThinkingEffort::High),
                            "xhigh" => Some(bedrock::BedrockAdaptiveThinkingEffort::XHigh),
                            "max" => Some(bedrock::BedrockAdaptiveThinkingEffort::Max),
                            _ => None,
                        })
                        .unwrap_or(default_effort);
                    Some(bedrock::Thinking::Adaptive { effort })
                }
                BedrockModelMode::Default => None,
            }
        } else {
            None
        },
        metadata: None,
        stop_sequences: Vec::new(),
        temperature: request.temperature.or(Some(default_temperature)),
        top_k: None,
        top_p: None,
        guardrail_identifier,
        guardrail_version,
    })
}

pub fn map_to_language_model_completion_events(
    events: Pin<Box<dyn Send + Stream<Item = Result<BedrockStreamingResponse, anyhow::Error>>>>,
) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
    struct RawToolUse {
        id: String,
        name: String,
        input_json: String,
    }

    struct State {
        events: Pin<Box<dyn Send + Stream<Item = Result<BedrockStreamingResponse, anyhow::Error>>>>,
        tool_uses_by_index: HashMap<i32, RawToolUse>,
        emitted_tool_use: bool,
    }

    let initial_state = State {
        events,
        tool_uses_by_index: HashMap::default(),
        emitted_tool_use: false,
    };

    futures::stream::unfold(initial_state, |mut state| async move {
        match state.events.next().await {
            Some(event_result) => match event_result {
                Ok(event) => {
                    let result = match event {
                        ConverseStreamOutput::ContentBlockDelta(cb_delta) => match cb_delta.delta {
                            Some(ContentBlockDelta::Text(text)) => {
                                Some(Ok(LanguageModelCompletionEvent::Text(text)))
                            }
                            Some(ContentBlockDelta::ToolUse(tool_output)) => {
                                if let Some(tool_use) = state
                                    .tool_uses_by_index
                                    .get_mut(&cb_delta.content_block_index)
                                {
                                    tool_use.input_json.push_str(tool_output.input());
                                    if let Ok(input) = serde_json::from_str::<serde_json::Value>(
                                        &fix_streamed_json(&tool_use.input_json),
                                    ) {
                                        Some(Ok(LanguageModelCompletionEvent::ToolUse(
                                            LanguageModelToolUse {
                                                id: tool_use.id.clone().into(),
                                                name: tool_use.name.clone().into(),
                                                is_input_complete: false,
                                                raw_input: tool_use.input_json.clone(),
                                                input:
                                                    language_model::LanguageModelToolUseInput::Json(
                                                        input,
                                                    ),
                                                thought_signature: None,
                                            },
                                        )))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            Some(ContentBlockDelta::ReasoningContent(thinking)) => match thinking {
                                ReasoningContentBlockDelta::Text(thoughts) => {
                                    Some(Ok(LanguageModelCompletionEvent::Thinking {
                                        text: thoughts,
                                        signature: None,
                                    }))
                                }
                                ReasoningContentBlockDelta::Signature(sig) => {
                                    Some(Ok(LanguageModelCompletionEvent::Thinking {
                                        text: "".into(),
                                        signature: Some(sig),
                                    }))
                                }
                                ReasoningContentBlockDelta::RedactedContent(redacted) => {
                                    let content = String::from_utf8(redacted.into_inner())
                                        .unwrap_or("REDACTED".to_string());
                                    Some(Ok(LanguageModelCompletionEvent::Thinking {
                                        text: content,
                                        signature: None,
                                    }))
                                }
                                _ => None,
                            },
                            _ => None,
                        },
                        ConverseStreamOutput::ContentBlockStart(cb_start) => {
                            if let Some(ContentBlockStart::ToolUse(tool_start)) = cb_start.start {
                                state.tool_uses_by_index.insert(
                                    cb_start.content_block_index,
                                    RawToolUse {
                                        id: tool_start.tool_use_id,
                                        name: tool_start.name,
                                        input_json: String::new(),
                                    },
                                );
                            }
                            None
                        }
                        ConverseStreamOutput::MessageStart(_) => None,
                        ConverseStreamOutput::ContentBlockStop(cb_stop) => state
                            .tool_uses_by_index
                            .remove(&cb_stop.content_block_index)
                            .map(|tool_use| {
                                state.emitted_tool_use = true;

                                let input = parse_tool_arguments(&tool_use.input_json)
                                    .unwrap_or_else(|_| Value::Object(Default::default()));

                                Ok(LanguageModelCompletionEvent::ToolUse(
                                    LanguageModelToolUse {
                                        id: tool_use.id.into(),
                                        name: tool_use.name.into(),
                                        is_input_complete: true,
                                        raw_input: tool_use.input_json,
                                        input: language_model::LanguageModelToolUseInput::Json(
                                            input,
                                        ),
                                        thought_signature: None,
                                    },
                                ))
                            }),
                        ConverseStreamOutput::Metadata(cb_meta) => cb_meta.usage.map(|metadata| {
                            Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                                input_tokens: metadata.input_tokens as u64,
                                output_tokens: metadata.output_tokens as u64,
                                cache_creation_input_tokens: metadata
                                    .cache_write_input_tokens
                                    .unwrap_or_default()
                                    as u64,
                                cache_read_input_tokens: metadata
                                    .cache_read_input_tokens
                                    .unwrap_or_default()
                                    as u64,
                            }))
                        }),
                        ConverseStreamOutput::MessageStop(message_stop) => {
                            let stop_reason = if state.emitted_tool_use {
                                // Some models (e.g. Kimi) send EndTurn even when
                                // they've made tool calls. Trust the content over
                                // the stop reason.
                                language_model::StopReason::ToolUse
                            } else {
                                match message_stop.stop_reason {
                                    StopReason::ToolUse => language_model::StopReason::ToolUse,
                                    _ => language_model::StopReason::EndTurn,
                                }
                            };
                            Some(Ok(LanguageModelCompletionEvent::Stop(stop_reason)))
                        }
                        _ => None,
                    };

                    Some((result, state))
                }
                Err(err) => Some((
                    Some(Err(LanguageModelCompletionError::Other(anyhow!(err)))),
                    state,
                )),
            },
            None => None,
        }
    })
    .filter_map(|result| async move { result })
}

struct ConfigurationView {
    access_key_id_editor: Entity<InputField>,
    secret_access_key_editor: Entity<InputField>,
    session_token_editor: Entity<InputField>,
    bearer_token_editor: Entity<InputField>,
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
    aws_profiles: Vec<AwsProfile>,
    load_profiles_task: Option<Task<()>>,
    authentication_error: Option<SharedString>,
    profile_error: Option<SharedString>,
    pending_auth_selection: Option<BedrockAuthSelection>,
    focus_handle: FocusHandle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BedrockAuthSelection {
    Automatic,
    Profile,
    StaticCredentials,
}

impl BedrockAuthSelection {
    const ALL: [Self; 3] = [Self::Automatic, Self::Profile, Self::StaticCredentials];

    fn from_state(state: &State) -> Self {
        match state
            .settings
            .as_ref()
            .and_then(|settings| settings.authentication_method.as_ref())
        {
            Some(BedrockAuthMethod::Automatic) => Self::Automatic,
            Some(BedrockAuthMethod::NamedProfile) => Self::Profile,
            Some(BedrockAuthMethod::ApiKey) => Self::StaticCredentials,
            None => match state.auth {
                Some(BedrockAuth::Automatic) => Self::Automatic,
                Some(BedrockAuth::NamedProfile { .. }) => Self::Profile,
                _ => Self::StaticCredentials,
            },
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Automatic => "Automatic (AWS credential chain)",
            Self::Profile => "AWS Profile",
            Self::StaticCredentials => "Static credentials or API key",
        }
    }

    fn settings_value(self) -> settings::BedrockAuthMethodContent {
        match self {
            Self::Automatic => settings::BedrockAuthMethodContent::Automatic,
            Self::Profile => settings::BedrockAuthMethodContent::NamedProfile,
            Self::StaticCredentials => settings::BedrockAuthMethodContent::ApiKey,
        }
    }

    fn uses_profile(self) -> bool {
        self == Self::Profile
    }
}

const BEDROCK_REGIONS: &[(&str, &str)] = &[
    ("us-east-1", "US East (N. Virginia)"),
    ("us-east-2", "US East (Ohio)"),
    ("us-west-1", "US West (N. California)"),
    ("us-west-2", "US West (Oregon)"),
    ("ca-central-1", "Canada (Central)"),
    ("ca-west-1", "Canada West (Calgary)"),
    ("eu-central-1", "Europe (Frankfurt)"),
    ("eu-central-2", "Europe (Zurich)"),
    ("eu-north-1", "Europe (Stockholm)"),
    ("eu-south-1", "Europe (Milan)"),
    ("eu-south-2", "Europe (Spain)"),
    ("eu-west-1", "Europe (Ireland)"),
    ("eu-west-2", "Europe (London)"),
    ("eu-west-3", "Europe (Paris)"),
    ("ap-east-2", "Asia Pacific (Taipei)"),
    ("ap-northeast-1", "Asia Pacific (Tokyo)"),
    ("ap-northeast-2", "Asia Pacific (Seoul)"),
    ("ap-northeast-3", "Asia Pacific (Osaka)"),
    ("ap-south-1", "Asia Pacific (Mumbai)"),
    ("ap-south-2", "Asia Pacific (Hyderabad)"),
    ("ap-southeast-1", "Asia Pacific (Singapore)"),
    ("ap-southeast-2", "Asia Pacific (Sydney)"),
    ("ap-southeast-3", "Asia Pacific (Jakarta)"),
    ("ap-southeast-4", "Asia Pacific (Melbourne)"),
    ("ap-southeast-5", "Asia Pacific (Malaysia)"),
    ("ap-southeast-6", "Asia Pacific (New Zealand)"),
    ("ap-southeast-7", "Asia Pacific (Thailand)"),
    ("il-central-1", "Israel (Tel Aviv)"),
    ("me-central-1", "Middle East (UAE)"),
    ("me-south-1", "Middle East (Bahrain)"),
    ("af-south-1", "Africa (Cape Town)"),
    ("sa-east-1", "South America (São Paulo)"),
    ("us-gov-east-1", "AWS GovCloud (US-East)"),
    ("us-gov-west-1", "AWS GovCloud (US-West)"),
];

impl ConfigurationView {
    const PLACEHOLDER_ACCESS_KEY_ID_TEXT: &'static str = "XXXXXXXXXXXXXXXX";
    const PLACEHOLDER_SECRET_ACCESS_KEY_TEXT: &'static str =
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
    const PLACEHOLDER_SESSION_TOKEN_TEXT: &'static str = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
    const PLACEHOLDER_BEARER_TOKEN_TEXT: &'static str = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";

    fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();

        cx.observe(&state, |this, state, cx| {
            let state = state.read(cx);
            if state.is_authenticated() {
                this.authentication_error = None;
            }
            if this.pending_auth_selection == Some(BedrockAuthSelection::from_state(state)) {
                this.pending_auth_selection = None;
            }
            cx.notify();
        })
        .detach();

        let access_key_id_editor = cx.new(|cx| {
            InputField::new(window, cx, Self::PLACEHOLDER_ACCESS_KEY_ID_TEXT)
                .label("Access Key ID")
                .tab_index(3)
                .tab_stop(true)
        });

        let secret_access_key_editor = cx.new(|cx| {
            InputField::new(window, cx, Self::PLACEHOLDER_SECRET_ACCESS_KEY_TEXT)
                .label("Secret Access Key")
                .tab_index(4)
                .tab_stop(true)
        });

        let session_token_editor = cx.new(|cx| {
            InputField::new(window, cx, Self::PLACEHOLDER_SESSION_TOKEN_TEXT)
                .label("Session Token (Optional)")
                .tab_index(5)
                .tab_stop(true)
        });

        let bearer_token_editor = cx.new(|cx| {
            InputField::new(window, cx, Self::PLACEHOLDER_BEARER_TOKEN_TEXT)
                .label("Bedrock API Key")
                .tab_index(6)
                .tab_stop(true)
        });

        let load_credentials_task = Some(cx.spawn({
            let state = state.clone();
            async move |this, cx| {
                let task = state.update(cx, |state, cx| state.authenticate_and_refresh(true, cx));
                let authentication_error = match task.await {
                    Ok(()) | Err(AuthenticateError::CredentialsNotFound) => None,
                    Err(error) => {
                        log::warn!("Failed to authenticate Amazon Bedrock: {error}");
                        Some(SharedString::from(error.to_string()))
                    }
                };
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    this.authentication_error = authentication_error;
                    cx.notify();
                })
                .log_err();
            }
        }));

        let profiles = cx.background_spawn(load_aws_profiles());
        let load_profiles_task = Some(cx.spawn(async move |this, cx| {
            let (aws_profiles, profile_error) = match profiles.await {
                Ok(profiles) => (profiles, None),
                Err(error) => {
                    log::warn!("Failed to load AWS profiles: {error:#}");
                    (Vec::new(), Some(SharedString::from(format!("{error:#}"))))
                }
            };
            this.update(cx, |this, cx| {
                this.aws_profiles = aws_profiles;
                this.profile_error = profile_error;
                this.load_profiles_task = None;
                cx.notify();
            })
            .log_err();
        }));

        Self {
            access_key_id_editor,
            secret_access_key_editor,
            session_token_editor,
            bearer_token_editor,
            state,
            load_credentials_task,
            aws_profiles: Vec::new(),
            load_profiles_task,
            authentication_error: None,
            profile_error: None,
            pending_auth_selection: None,
            focus_handle,
        }
    }

    fn select_authentication_mode(
        &mut self,
        selection: BedrockAuthSelection,
        cx: &mut Context<Self>,
    ) {
        if selection.uses_profile() {
            self.pending_auth_selection = Some(selection);
            cx.notify();
        } else {
            self.pending_auth_selection = None;
            Self::persist_authentication(selection, None, cx);
        }
    }

    fn persist_authentication(
        selection: BedrockAuthSelection,
        profile_name: Option<String>,
        cx: &mut App,
    ) {
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, move |settings, _| {
            let bedrock = settings
                .language_models
                .get_or_insert_default()
                .bedrock
                .get_or_insert_default();
            bedrock.authentication_method = Some(selection.settings_value());
            if let Some(profile_name) = profile_name {
                bedrock.profile = Some(profile_name);
            }
        });
    }

    fn persist_region(region: String, cx: &mut App) {
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, move |settings, _| {
            settings
                .language_models
                .get_or_insert_default()
                .bedrock
                .get_or_insert_default()
                .region = Some(region);
        });
    }

    fn reload_profiles(&mut self, cx: &mut Context<Self>) {
        self.profile_error = None;
        let profiles = cx.background_spawn(load_aws_profiles());
        self.load_profiles_task = Some(cx.spawn(async move |this, cx| {
            let (aws_profiles, profile_error) = match profiles.await {
                Ok(profiles) => (profiles, None),
                Err(error) => {
                    log::warn!("Failed to load AWS profiles: {error:#}");
                    (Vec::new(), Some(SharedString::from(format!("{error:#}"))))
                }
            };
            this.update(cx, |this, cx| {
                this.aws_profiles = aws_profiles;
                this.profile_error = profile_error;
                this.load_profiles_task = None;
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn retry_discovery(&mut self, cx: &mut Context<Self>) {
        self.state.update(cx, |state, cx| state.refresh_targets(cx));
    }

    fn render_configuration_controls(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let (selection, profile_name, region) = {
            let state = self.state.read(cx);
            let selection = self
                .pending_auth_selection
                .unwrap_or_else(|| BedrockAuthSelection::from_state(state));
            let profile_name = state
                .settings
                .as_ref()
                .and_then(|settings| settings.profile_name.clone())
                .or_else(|| match state.auth.as_ref() {
                    Some(BedrockAuth::NamedProfile { profile_name }) => Some(profile_name.clone()),
                    _ => None,
                });
            (selection, profile_name, state.get_region())
        };

        let view = cx.entity();
        let auth_menu = ContextMenu::build(window, cx, move |mut menu, _, _| {
            for candidate in BedrockAuthSelection::ALL {
                let view = view.clone();
                menu.push_item(
                    ContextMenuEntry::new(candidate.label())
                        .toggleable(IconPosition::End, candidate == selection)
                        .handler(move |_, cx| {
                            view.update(cx, |this, cx| {
                                this.select_authentication_mode(candidate, cx)
                            });
                        }),
                );
            }
            menu
        });

        let auth_dropdown = DropdownMenu::new(
            "bedrock-authentication-method",
            selection.label(),
            auth_menu,
        )
        .style(DropdownStyle::Outlined)
        .full_width(true)
        .tab_index(0)
        .aria_label("Amazon Bedrock authentication method")
        .aria_description(
            "Choose the AWS credential chain, any local AWS profile including SSO, or static credentials.",
        )
        .aria_value(selection.label());

        let profile_control = selection.uses_profile().then(|| {
            let profiles = self.aws_profiles.clone();
            let current_profile = profile_name
                .clone()
                .or_else(|| profiles.first().map(|profile| profile.name.clone()))
                .unwrap_or_else(|| "No local AWS profiles".to_string());
            let has_profiles = !profiles.is_empty();
            let profile_menu = ContextMenu::build(window, cx, {
                let current_profile = current_profile.clone();
                move |mut menu, _, _| {
                    if profiles.is_empty() {
                        menu.push_item(
                            ContextMenuEntry::new("No local profiles found").disabled(true),
                        );
                    } else {
                        for profile in profiles.iter() {
                            let profile_name = profile.name.clone();
                            menu.push_item(
                                ContextMenuEntry::new(profile.name.clone())
                                    .toggleable(IconPosition::End, profile.name == current_profile)
                                    .handler(move |_, cx| {
                                        Self::persist_authentication(
                                            BedrockAuthSelection::Profile,
                                            Some(profile_name.clone()),
                                            cx,
                                        );
                                    }),
                            );
                        }
                    }
                    menu
                }
            });
            let loading = self.load_profiles_task.is_some();
            let disabled = loading || !has_profiles;
            let disabled_reason = if loading {
                "Local AWS profiles are still loading."
            } else {
                "No AWS profiles were found in your AWS config or credentials files."
            };
            v_flex()
                .gap_1()
                .child(Label::new("AWS Profile").size(LabelSize::Small))
                .child(
                    DropdownMenu::new(
                        "bedrock-profile",
                        if loading {
                            "Loading AWS profiles…".to_string()
                        } else {
                            current_profile.clone()
                        },
                        profile_menu,
                    )
                    .style(DropdownStyle::Outlined)
                    .full_width(true)
                    .tab_index(1)
                    .disabled(disabled)
                    .aria_label("AWS Profile")
                    .aria_description(if disabled {
                        disabled_reason
                    } else {
                        "Choose a profile loaded from your local AWS config and credentials files."
                    })
                    .aria_value(current_profile),
                )
                .when(!loading && !has_profiles, |this| {
                    this.child(
                        Button::new("reload-empty-bedrock-profiles", "Reload AWS profiles")
                            .style(ButtonStyle::Outlined)
                            .on_click(cx.listener(|this, _, _, cx| this.reload_profiles(cx))),
                    )
                })
                .into_any_element()
        });

        let region_from_environment = ZED_BEDROCK_REGION_VAR
            .value
            .as_deref()
            .is_some_and(|region| !region.is_empty());
        let region_menu = ContextMenu::build(window, cx, {
            let current_region = region.clone();
            move |mut menu, _, _| {
                if !BEDROCK_REGIONS
                    .iter()
                    .any(|(region, _)| *region == current_region)
                {
                    let region = current_region.clone();
                    menu.push_item(
                        ContextMenuEntry::new(region.clone())
                            .toggleable(IconPosition::End, true)
                            .handler(move |_, cx| Self::persist_region(region.clone(), cx)),
                    );
                }
                for (region, name) in BEDROCK_REGIONS {
                    let label = format!("{name} — {region}");
                    let selected = *region == current_region;
                    let region = (*region).to_string();
                    menu.push_item(
                        ContextMenuEntry::new(label)
                            .toggleable(IconPosition::End, selected)
                            .handler(move |_, cx| Self::persist_region(region.clone(), cx)),
                    );
                }
                menu
            }
        });
        let region_description = if region_from_environment {
            format!(
                "Region is controlled by the {} environment variable. Unset it and restart Zed to change regions here.",
                ZED_BEDROCK_REGION_VAR.name
            )
        } else {
            "Choose the AWS Region used for Bedrock Runtime and Mantle model discovery.".to_string()
        };

        v_flex()
            .gap_2()
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new("Authentication").size(LabelSize::Small))
                    .child(auth_dropdown),
            )
            .when_some(profile_control, |this, control| this.child(control))
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new("AWS Region").size(LabelSize::Small))
                    .child(
                        DropdownMenu::new("bedrock-region", region.clone(), region_menu)
                            .style(DropdownStyle::Outlined)
                            .full_width(true)
                            .tab_index(2)
                            .disabled(region_from_environment)
                            .aria_label("Amazon Bedrock AWS Region")
                            .aria_description(region_description)
                            .aria_value(region),
                    ),
            )
            .into_any_element()
    }

    fn render_catalog_status(&self, cx: &mut Context<Self>) -> AnyElement {
        let state = self.state.read(cx);
        if !state.is_authenticated() {
            return v_flex()
                .gap_1()
                .child(Label::new("Model catalogs").size(LabelSize::Small))
                .child(
                    Label::new("Authenticate to discover the models available to this AWS account and Region.")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .into_any_element();
        }
        let runtime_status = if state.discovery_in_progress {
            "Discovering Bedrock Runtime models…".to_string()
        } else if let Some(error) = state.runtime_discovery_error.as_ref() {
            format!(
                "Bedrock Runtime discovery failed: {error}. Built-in and configured models remain available."
            )
        } else {
            format!(
                "Bedrock Runtime: {} model targets discovered",
                state.runtime_targets.len()
            )
        };
        let mantle_status = if state.mantle_endpoint_available == Some(false) {
            format!(
                "Bedrock Mantle is not available in {}. Runtime models remain available.",
                state.get_region()
            )
        } else if state.discovery_in_progress {
            "Discovering Bedrock Mantle models…".to_string()
        } else if let Some(error) = state.mantle_discovery_error.as_ref() {
            format!(
                "Bedrock Mantle discovery failed: {error}. Built-in and configured Mantle models remain available."
            )
        } else {
            format!(
                "Bedrock Mantle: {} model targets discovered",
                state.mantle_targets.len()
            )
        };
        let has_error =
            state.runtime_discovery_error.is_some() || state.mantle_discovery_error.is_some();
        let can_retry = state.is_authenticated() && !state.discovery_in_progress;
        let endpoint_status = state
            .settings
            .as_ref()
            .and_then(|settings| settings.endpoint.as_ref())
            .filter(|endpoint| !endpoint.is_empty())
            .map(|endpoint| {
                format!(
                    "Custom Bedrock Runtime endpoint: {endpoint}. Mantle continues to use its regional AWS endpoint."
                )
            });

        v_flex()
            .gap_1()
            .child(Label::new("Model catalogs").size(LabelSize::Small))
            .when_some(endpoint_status, |this, status| {
                this.child(
                    Label::new(status)
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
            })
            .child(Label::new(runtime_status).size(LabelSize::Small).color(
                if state.runtime_discovery_error.is_some() {
                    Color::Error
                } else {
                    Color::Muted
                },
            ))
            .child(Label::new(mantle_status).size(LabelSize::Small).color(
                if state.mantle_discovery_error.is_some() {
                    Color::Error
                } else {
                    Color::Muted
                },
            ))
            .when(has_error, |this| {
                this.child(
                    Button::new("retry-bedrock-discovery", "Retry model discovery")
                        .style(ButtonStyle::Outlined)
                        .disabled(!can_retry)
                        .on_click(cx.listener(|this, _, _, cx| this.retry_discovery(cx))),
                )
            })
            .into_any_element()
    }

    fn save_credentials(
        &mut self,
        _: &menu::Confirm,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let access_key_id = self
            .access_key_id_editor
            .read(cx)
            .text(cx)
            .trim()
            .to_string();
        let secret_access_key = self
            .secret_access_key_editor
            .read(cx)
            .text(cx)
            .trim()
            .to_string();
        let session_token = self
            .session_token_editor
            .read(cx)
            .text(cx)
            .trim()
            .to_string();
        let session_token = if session_token.is_empty() {
            None
        } else {
            Some(session_token)
        };
        let bearer_token = self
            .bearer_token_editor
            .read(cx)
            .text(cx)
            .trim()
            .to_string();
        let bearer_token = if bearer_token.is_empty() {
            None
        } else {
            Some(bearer_token)
        };

        let state = self.state.clone();
        cx.spawn(async move |_, cx| {
            state
                .update(cx, |state, cx| {
                    let credentials = BedrockCredentials {
                        access_key_id,
                        secret_access_key,
                        session_token,
                        bearer_token,
                    };

                    state.set_static_credentials(credentials, cx)
                })
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_credentials(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.access_key_id_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));
        self.secret_access_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));
        self.session_token_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));
        self.bearer_token_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn(async move |_, cx| state.update(cx, |state, cx| state.reset_auth(cx)).await)
            .detach_and_log_err(cx);
    }

    fn on_tab(&mut self, _: &menu::SelectNext, window: &mut Window, cx: &mut Context<Self>) {
        window.focus_next(cx);
    }

    fn on_tab_prev(
        &mut self,
        _: &menu::SelectPrevious,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus_prev(cx);
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (env_var_set, auth, selection, is_authenticated) = {
            let state = self.state.read(cx);
            (
                state.credentials_from_env,
                state.auth.clone(),
                self.pending_auth_selection
                    .unwrap_or_else(|| BedrockAuthSelection::from_state(state)),
                state.is_authenticated(),
            )
        };

        let configured_label = match &auth {
            Some(BedrockAuth::Automatic) => {
                "Using automatic credentials (AWS default chain)".into()
            }
            Some(BedrockAuth::NamedProfile { profile_name }) => {
                format!("Using AWS profile: {profile_name}")
            }
            Some(BedrockAuth::IamCredentials { .. }) if env_var_set => {
                format!(
                    "Using IAM credentials from {} and {} environment variables",
                    ZED_BEDROCK_ACCESS_KEY_ID_VAR.name, ZED_BEDROCK_SECRET_ACCESS_KEY_VAR.name
                )
            }
            Some(BedrockAuth::IamCredentials { .. }) => "Using IAM credentials".into(),
            Some(BedrockAuth::ApiKey { .. }) if env_var_set => {
                format!(
                    "Using Bedrock API Key from {} environment variable",
                    ZED_BEDROCK_BEARER_TOKEN_VAR.name
                )
            }
            Some(BedrockAuth::ApiKey { .. }) => "Using Bedrock API Key".into(),
            None => "Not authenticated".into(),
        };

        let tooltip_label = if env_var_set {
            Some(format!(
                "To reset your credentials, unset the {}, {}, and {} or {} environment variables.",
                ZED_BEDROCK_ACCESS_KEY_ID_VAR.name,
                ZED_BEDROCK_SECRET_ACCESS_KEY_VAR.name,
                ZED_BEDROCK_SESSION_TOKEN_VAR.name,
                ZED_BEDROCK_BEARER_TOKEN_VAR.name
            ))
        } else {
            None
        };

        let credentials_control =
            if selection == BedrockAuthSelection::StaticCredentials && is_authenticated {
                ConfiguredApiCard::new("bedrock-reset", configured_label)
                    .disabled(env_var_set)
                    .on_click(cx.listener(|this, _, window, cx| this.reset_credentials(window, cx)))
                    .when_some(tooltip_label, |this, label| this.tooltip_label(label))
                    .into_any_element()
            } else if selection == BedrockAuthSelection::StaticCredentials {
                self.render_static_credentials_ui().into_any_element()
            } else {
                ConfiguredApiCard::new("bedrock-auth-status", configured_label)
                    .disabled(true)
                    .tooltip_label(
                        "Change the authentication method above to use different credentials.",
                    )
                    .into_any_element()
            };

        let configuration_controls = self.render_configuration_controls(window, cx);
        let catalog_status = self.render_catalog_status(cx);
        let loading_credentials = self.load_credentials_task.is_some();
        let authentication_error = self.authentication_error.clone();
        let profile_error = self.profile_error.clone();

        v_flex()
            .min_w_0()
            .w_full()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::on_tab))
            .on_action(cx.listener(Self::on_tab_prev))
            .on_action(cx.listener(ConfigurationView::save_credentials))
            .gap_1()
            .child(Headline::new("Amazon Bedrock").size(HeadlineSize::Small))
            .child(
                Label::new(
                    "Choose an AWS authentication method and Region. Zed discovers Bedrock Runtime and Mantle models for the selected account without requiring model IDs or ARNs.",
                )
                .color(Color::Muted),
            )
            .child(
                Label::new("But first, to access models on AWS, you need to:")
                    .mt_1()
                    .color(Color::Muted),
            )
            .child(
                List::new()
                    .child(
                        ListBulletItem::new("")
                            .child(
                                Label::new(
                                    "Grant permissions to the strategy you'll use according to the:",
                                )
                                .color(Color::Muted),
                            )
                            .child(ButtonLink::new(
                                "Prerequisites",
                                "https://docs.aws.amazon.com/bedrock/latest/userguide/inference-prereq.html",
                            )),
                    )
                    .child(
                        ListBulletItem::new("")
                            .child(
                                Label::new("Select the models you would like access to:")
                                    .color(Color::Muted),
                            )
                            .child(ButtonLink::new(
                                "Bedrock Model Catalog",
                                "https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog",
                            )),
                    ),
            )
            .child(Divider::horizontal().my_1())
            .child(configuration_controls)
            .when(loading_credentials, |this| {
                this.child(
                    Label::new("Loading credentials…")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
            })
            .when_some(authentication_error, |this, error| {
                this.child(
                    Label::new(format!("Authentication failed: {error}"))
                        .size(LabelSize::Small)
                        .color(Color::Error),
                )
            })
            .when_some(profile_error, |this, error| {
                this.child(
                    v_flex()
                        .gap_1()
                        .child(
                            Label::new(format!("Could not load local AWS profiles: {error}"))
                                .size(LabelSize::Small)
                                .color(Color::Error),
                        )
                        .child(
                            Button::new("reload-bedrock-profiles", "Reload AWS profiles")
                                .style(ButtonStyle::Outlined)
                                .on_click(cx.listener(|this, _, _, cx| this.reload_profiles(cx))),
                        ),
                )
            })
            .child(Divider::horizontal().my_1())
            .child(catalog_status)
            .child(credentials_control)
            .into_any()
    }
}

impl ConfigurationView {
    fn render_static_credentials_ui(&self) -> impl IntoElement {
        let list_item = List::new()
            .child(
                ListBulletItem::new("")
                    .child(
                        Label::new(
                            "For access keys: Create an IAM user in the AWS console with programmatic access",
                        )
                        .color(Color::Muted),
                    )
                    .child(ButtonLink::new(
                        "IAM Console",
                        "https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-1#/users",
                    )),
            )
            .child(
                ListBulletItem::new("")
                    .child(
                        Label::new("For Bedrock API Keys: Generate an API key from the")
                            .color(Color::Muted),
                    )
                    .child(ButtonLink::new(
                        "Bedrock Console",
                        "https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html",
                    )),
            )
            .child(
                ListBulletItem::new("")
                    .child(
                        Label::new("Attach the necessary Bedrock permissions to")
                            .color(Color::Muted),
                    )
                    .child(ButtonLink::new(
                        "this user",
                        "https://docs.aws.amazon.com/bedrock/latest/userguide/inference-prereq.html",
                    )),
            )
            .child(
                ListBulletItem::new(
                    "Enter either access keys OR a Bedrock API Key below (not both)",
                )
                .label_color(Color::Muted),
            );

        v_flex()
            .my_2()
            .tab_group()
            .gap_1p5()
            .child(Divider::horizontal())
            .child(Label::new("Static Credentials").mt_2())
            .child(
                Label::new(
                    "This method uses your AWS access key ID and secret access key, or a Bedrock API Key.",
                )
                .color(Color::Muted),
            )
            .child(list_item)
            .child(
                v_flex()
                    .gap_1()
                    .child(self.access_key_id_editor.clone())
                    .child(self.secret_access_key_editor.clone())
                    .child(self.session_token_editor.clone()),
            )
            .child(
                Label::new(format!(
                    "You can also set the {}, {} and {} environment variables (or {} for Bedrock API Key authentication) and restart Zed.",
                    ZED_BEDROCK_ACCESS_KEY_ID_VAR.name,
                    ZED_BEDROCK_SECRET_ACCESS_KEY_VAR.name,
                    ZED_BEDROCK_REGION_VAR.name,
                    ZED_BEDROCK_BEARER_TOKEN_VAR.name
                ))
                .size(LabelSize::Small)
                .color(Color::Muted),
            )
            .child(
                Label::new(format!(
                    "The profile selector above also supports {}; custom endpoints can use {}; and temporary IAM credentials can use {}.",
                    ZED_AWS_PROFILE_VAR.name,
                    ZED_AWS_ENDPOINT_VAR.name,
                    ZED_BEDROCK_SESSION_TOKEN_VAR.name
                ))
                .size(LabelSize::Small)
                .color(Color::Muted)
                .mt_1()
                .mb_2p5(),
            )
            .child(Divider::horizontal())
            .child(Label::new("Using the API key").mt_2().mb_1())
            .child(self.bearer_token_editor.clone())
            .child(
                Label::new(format!(
                    "The Region selected above can be overridden with {}.",
                    ZED_BEDROCK_REGION_VAR.name
                ))
                .size(LabelSize::Small)
                .color(Color::Muted)
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use language_model::{
        LanguageModelRequestMessage, LanguageModelToolResult, LanguageModelToolUseId,
    };
    use open_ai::responses::{
        ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem,
    };

    fn into_bedrock_request(messages: Vec<LanguageModelRequestMessage>) -> bedrock::Request {
        into_bedrock(
            LanguageModelRequest {
                messages,
                ..Default::default()
            },
            "claude-sonnet-4-5".to_string(),
            1.0,
            4096,
            BedrockModelMode::Default,
            true,
            true,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_aws_profile_discovery_merges_all_profile_types() {
        use aws_runtime::env_config::file::EnvConfigFileKind;

        let files = EnvConfigFiles::builder()
            .with_contents(
                EnvConfigFileKind::Config,
                "[profile sso]\nsso_session = company\nsso_account_id = 123456789012\n\
                 [profile static]\nregion = us-west-2\n",
            )
            .with_contents(
                EnvConfigFileKind::Credentials,
                "[default]\naws_access_key_id = default\n\
                 [static]\naws_access_key_id = static\n",
            )
            .build();
        let profiles = smol::block_on(load_aws_profiles_from(
            &AwsFs::from_slice(&[]),
            &AwsEnv::from_slice(&[]),
            &files,
        ))
        .unwrap();

        assert_eq!(
            profiles,
            vec![
                AwsProfile {
                    name: "default".to_string(),
                },
                AwsProfile {
                    name: "sso".to_string(),
                },
                AwsProfile {
                    name: "static".to_string(),
                },
            ]
        );
    }

    #[test]
    fn test_v39_bedrock_auth_selection_unifies_profiles() {
        assert_eq!(
            BedrockAuthSelection::ALL.to_vec(),
            vec![
                BedrockAuthSelection::Automatic,
                BedrockAuthSelection::Profile,
                BedrockAuthSelection::StaticCredentials,
            ],
            "CLI/IAM and SSO profiles must share one AWS Profile option",
        );
        let legacy_sso: settings::BedrockAuthMethodContent =
            serde_json::from_value(serde_json::json!("sso"))
                .expect("deserialize legacy sso authentication method");
        assert_eq!(
            serde_json::to_value(&legacy_sso).expect("normalize legacy sso authentication method"),
            serde_json::json!("named_profile")
        );
        assert_eq!(
            BedrockAuthMethod::from(legacy_sso),
            BedrockAuthMethod::NamedProfile,
            "legacy sso settings must normalize to AWS Profile",
        );
        let schema =
            serde_json::to_value(schemars::schema_for!(settings::BedrockAuthMethodContent))
                .expect("serialize Bedrock authentication schema");
        assert!(
            !schema.to_string().contains("\"sso\""),
            "legacy sso alias must not remain a separate schema option",
        );
        assert_eq!(
            serde_json::to_value(BedrockAuthSelection::Automatic.settings_value())
                .expect("serialize automatic authentication"),
            serde_json::json!("default")
        );
        assert_eq!(
            serde_json::to_value(BedrockAuthSelection::Profile.settings_value())
                .expect("serialize named-profile authentication"),
            serde_json::json!("named_profile")
        );
        assert_eq!(
            serde_json::to_value(BedrockAuthSelection::StaticCredentials.settings_value())
                .expect("serialize static authentication"),
            serde_json::json!("api_key")
        );
        assert!(BedrockAuthSelection::Profile.uses_profile());
        assert!(!BedrockAuthSelection::Automatic.uses_profile());
    }

    #[gpui::test]
    fn test_bedrock_profile_settings_update_is_non_panicking(cx: &mut App) {
        let store = SettingsStore::new(cx, &settings::default_settings());
        let updated = store
            .new_text_for_update("{}".to_string(), |settings| {
                let bedrock = settings
                    .language_models
                    .get_or_insert_default()
                    .bedrock
                    .get_or_insert_default();
                bedrock.authentication_method =
                    Some(settings::BedrockAuthMethodContent::NamedProfile);
                bedrock.profile = Some("company-sso".to_string());
            })
            .expect("persist profile authentication settings");

        let value: serde_json::Value =
            serde_json::from_str(&updated).expect("parse updated settings");
        assert_eq!(
            value.pointer("/language_models/bedrock/authentication_method"),
            Some(&serde_json::json!("named_profile"))
        );
        assert_eq!(
            value.pointer("/language_models/bedrock/profile"),
            Some(&serde_json::json!("company-sso"))
        );
    }

    #[gpui::test]
    async fn test_v40_bedrock_discovery_runs_on_tokio(cx: &mut TestAppContext) {
        cx.update(gpui_tokio::init);
        cx.executor().allow_parking();
        let discovery_task = cx.update(|cx| {
            spawn_bedrock_discovery(cx, async { tokio::runtime::Handle::try_current().is_ok() })
        });

        assert!(
            discovery_task.await.expect("run Bedrock discovery"),
            "Bedrock Runtime and Mantle discovery must run on Tokio"
        );
    }

    #[test]
    fn test_v40_bedrock_discovery_task_failure_is_contained() {
        let (runtime_result, mantle_result) =
            contain_bedrock_discovery_task_failure(Err(anyhow!("synthetic task panic")));
        assert!(
            runtime_result
                .expect_err("Runtime discovery must receive the task error")
                .to_string()
                .contains("synthetic task panic")
        );
        assert!(
            mantle_result
                .expect_err("Mantle discovery must receive the task error")
                .to_string()
                .contains("synthetic task panic")
        );
    }

    #[test]
    fn test_aws_session_cache_shares_until_invalidated() {
        let mut cache = AwsSessionCache::default();
        let shared = cache.clone();
        assert!(Arc::ptr_eq(&cache.cell, &shared.cell));

        cache.invalidate();
        assert!(!Arc::ptr_eq(&cache.cell, &shared.cell));
    }

    struct FakeRuntimeCatalogSource {
        foundation_models: Vec<RuntimeFoundationModel>,
        profile_pages: BTreeMap<Option<String>, RuntimeInferenceProfilePage>,
    }

    impl RuntimeCatalogSource for FakeRuntimeCatalogSource {
        fn list_foundation_models(&self) -> BoxFuture<'_, Result<Vec<RuntimeFoundationModel>>> {
            futures::future::ready(Ok(self.foundation_models.clone())).boxed()
        }

        fn list_inference_profiles(
            &self,
            next_token: Option<String>,
        ) -> BoxFuture<'_, Result<RuntimeInferenceProfilePage>> {
            futures::future::ready(
                self.profile_pages
                    .get(&next_token)
                    .cloned()
                    .with_context(|| format!("missing fake profile page for {next_token:?}")),
            )
            .boxed()
        }
    }

    fn runtime_foundation_model(
        model_id: &str,
        output_modalities: &[&str],
        lifecycle_status: &str,
    ) -> RuntimeFoundationModel {
        RuntimeFoundationModel {
            model_arn: format!("arn:aws:bedrock:us-east-1::foundation-model/{model_id}"),
            model_id: model_id.to_string(),
            model_name: Some(format!("Display {model_id}")),
            output_modalities: output_modalities
                .iter()
                .map(|modality| modality.to_string())
                .collect(),
            response_streaming_supported: Some(true),
            inference_types_supported: vec!["ON_DEMAND".to_string()],
            lifecycle_status: Some(lifecycle_status.to_string()),
        }
    }

    fn runtime_inference_profile(
        id: &str,
        base_model_id: &str,
        kind: &str,
        status: &str,
    ) -> RuntimeInferenceProfile {
        RuntimeInferenceProfile {
            name: format!("Profile {id}"),
            arn: format!("arn:aws:bedrock:us-east-1:123456789012:inference-profile/{id}"),
            model_arns: vec![format!(
                "arn:aws:bedrock:us-east-1::foundation-model/{base_model_id}"
            )],
            id: id.to_string(),
            status: status.to_string(),
            kind: kind.to_string(),
        }
    }

    #[test]
    fn test_runtime_discovery_paginates_filters_and_preserves_profile_targets() {
        let source = FakeRuntimeCatalogSource {
            foundation_models: vec![
                runtime_foundation_model(
                    "anthropic.claude-sonnet-4-5-20250929-v1:0",
                    &["TEXT"],
                    "ACTIVE",
                ),
                runtime_foundation_model("image-only", &["IMAGE"], "ACTIVE"),
                runtime_foundation_model("legacy-text", &["TEXT"], "LEGACY"),
            ],
            profile_pages: BTreeMap::from([
                (
                    None,
                    RuntimeInferenceProfilePage {
                        profiles: vec![runtime_inference_profile(
                            "us.anthropic.claude-sonnet-4-5-v1:0",
                            "anthropic.claude-sonnet-4-5-20250929-v1:0",
                            "SYSTEM_DEFINED",
                            "ACTIVE",
                        )],
                        next_token: Some("page-2".to_string()),
                    },
                ),
                (
                    Some("page-2".to_string()),
                    RuntimeInferenceProfilePage {
                        profiles: vec![
                            runtime_inference_profile(
                                "my-cost-profile",
                                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                                "APPLICATION",
                                "ACTIVE",
                            ),
                            runtime_inference_profile(
                                "inactive-profile",
                                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                                "APPLICATION",
                                "INACTIVE",
                            ),
                        ],
                        next_token: None,
                    },
                ),
            ]),
        };

        let targets = smol::block_on(discover_runtime_targets(&source)).unwrap();

        assert_eq!(targets.len(), 3);
        assert_eq!(targets[0].kind, BedrockTargetKind::FoundationModel);
        assert_eq!(targets[0].id, "claude-sonnet-4-5");
        assert_eq!(targets[1].kind, BedrockTargetKind::SystemInferenceProfile);
        assert_eq!(
            targets[1].id,
            "bedrock-runtime:profile:us.anthropic.claude-sonnet-4-5-v1:0"
        );
        assert_eq!(
            targets[2].kind,
            BedrockTargetKind::ApplicationInferenceProfile
        );
        assert_eq!(targets[2].invocation_id, "my-cost-profile");
        assert_eq!(
            targets[2].arn.as_deref(),
            Some("arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-cost-profile")
        );
    }

    #[test]
    fn test_discovered_profile_uses_exact_target_and_known_base_capabilities() {
        let model = BedrockTargetDescriptor {
            endpoint: BedrockEndpoint::Runtime,
            kind: BedrockTargetKind::ApplicationInferenceProfile,
            protocol: BedrockTargetProtocol::Converse,
            id: "bedrock-runtime:profile:cost-center".to_string(),
            invocation_id: "cost-center".to_string(),
            arn: Some(
                "arn:aws:bedrock:us-east-1:123456789012:inference-profile/cost-center".to_string(),
            ),
            base_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0".to_string(),
            display_name: "Cost center".to_string(),
            capabilities: converse_target_capabilities("anthropic.claude-sonnet-4-5-20250929-v1:0"),
        }
        .into_converse_model();

        assert_eq!(
            model
                .cross_region_inference_id("not-a-region", false)
                .unwrap(),
            "cost-center"
        );
        assert!(model.supports_tool_use());
        assert!(model.supports_images());
        assert!(model.supports_thinking());
        assert_eq!(model.max_token_count(), 1_000_000);
    }

    #[test]
    fn test_mantle_catalog_merges_known_models_and_future_fallbacks() {
        let targets = parse_mantle_model_catalog(
            r#"{
                "object": "list",
                "data": [
                    {"id": "anthropic.claude-fable-5", "owned_by": "aws"},
                    {"id": "openai.gpt-5.6-sol"},
                    {"id": "anthropic.claude-future"},
                    {"id": "openai.gpt-future"},
                    {"id": "future-provider.text-model"},
                    {"id": "anthropic.claude-fable-5"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(targets.len(), 5);
        let fable = targets
            .iter()
            .find(|target| target.invocation_id == "anthropic.claude-fable-5")
            .unwrap();
        assert_eq!(fable.id, "bedrock-mantle/claude-fable-5");
        assert_eq!(fable.protocol, BedrockTargetProtocol::AnthropicMessages);
        assert!(fable.capabilities.supports_tools);
        assert!(fable.capabilities.supports_images);
        assert!(fable.capabilities.supports_thinking);
        assert_eq!(fable.capabilities.max_token_count, 1_000_000);

        let future_anthropic = targets
            .iter()
            .find(|target| target.invocation_id == "anthropic.claude-future")
            .unwrap();
        assert_eq!(
            future_anthropic.protocol,
            BedrockTargetProtocol::AnthropicMessages
        );
        assert!(!future_anthropic.capabilities.supports_tools);
        assert!(!future_anthropic.capabilities.supports_images);
        assert!(!future_anthropic.capabilities.supports_thinking);

        let future_openai = targets
            .iter()
            .find(|target| target.invocation_id == "openai.gpt-future")
            .unwrap();
        assert_eq!(
            future_openai.protocol,
            BedrockTargetProtocol::OpenAiResponses
        );
        let future_other = targets
            .iter()
            .find(|target| target.invocation_id == "future-provider.text-model")
            .unwrap();
        assert_eq!(
            future_other.protocol,
            BedrockTargetProtocol::OpenAiChatCompletions
        );
    }

    #[test]
    fn test_discovered_mantle_model_preserves_ui_and_request_ids() {
        let model = mantle_targets_from_model_ids(["anthropic.claude-fable-5".to_string()])
            .into_iter()
            .next()
            .unwrap()
            .into_mantle_model();

        assert_eq!(model.id(), "bedrock-mantle/claude-fable-5");
        assert_eq!(model.request_id(), "anthropic.claude-fable-5");
        assert_eq!(model.protocol(), MantleProtocol::AnthropicMessages);
        assert!(model.supports_tools());
        assert_eq!(model.max_output_tokens(), 128_000);
    }

    #[test]
    fn test_mantle_catalog_request_supports_bearer_auth() {
        let url = mantle_endpoint_url("us-east-1", "v1/models");
        let request = build_mantle_http_request(
            Method::GET,
            &url,
            &[],
            "us-east-1",
            &MantleAuth::ApiKey {
                api_key: "test-api-key".to_string(),
            },
            &CustomHeaders::default(),
            &[],
        )
        .unwrap();

        assert_eq!(request.method(), Method::GET);
        assert_eq!(request.uri().to_string(), url);
        assert_eq!(
            request
                .headers()
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer test-api-key")
        );
    }

    #[test]
    fn test_user_tool_results_precede_other_content() {
        let tool_result = |id: &'static str| {
            MessageContent::ToolResult(LanguageModelToolResult {
                tool_use_id: LanguageModelToolUseId::from(id),
                tool_name: Arc::from("read_file"),
                is_error: false,
                content: vec![LanguageModelToolResultContent::Text(Arc::from(id))],
                output: None,
            })
        };
        let request = into_bedrock_request(vec![LanguageModelRequestMessage {
            role: Role::User,
            content: vec![
                MessageContent::Text("before".into()),
                tool_result("call-1"),
                MessageContent::Text("between".into()),
                tool_result("call-2"),
                MessageContent::Text("after".into()),
            ],
            cache: false,
            reasoning_details: None,
        }]);

        assert!(matches!(
            request.messages[0].content(),
            [
                BedrockInnerContent::ToolResult(_),
                BedrockInnerContent::ToolResult(_),
                BedrockInnerContent::Text(_),
                BedrockInnerContent::Text(_),
                BedrockInnerContent::Text(_),
            ]
        ));
    }

    #[test]
    fn test_cache_marked_message_that_filters_to_empty_is_dropped() {
        let request = into_bedrock_request(vec![
            LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text("What's the weather?".into())],
                cache: false,
                reasoning_details: None,
            },
            LanguageModelRequestMessage {
                role: Role::Assistant,
                content: vec![MessageContent::Thinking {
                    text: "Let me think about this...".into(),
                    signature: None,
                }],
                cache: true,
                reasoning_details: None,
            },
            LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text("Summarize this conversation.".into())],
                cache: false,
                reasoning_details: None,
            },
        ]);

        for message in &request.messages {
            assert!(
                message
                    .content()
                    .iter()
                    .any(|block| !matches!(block, BedrockInnerContent::CachePoint(_))),
                "message must not consist solely of cache points: {:?}",
                message
            );
        }
        assert!(
            request
                .messages
                .iter()
                .all(|message| *message.role() == bedrock::BedrockRole::User),
            "the assistant message stripped to empty content should be dropped entirely"
        );
    }

    #[test]
    fn test_cache_marked_message_with_content_gets_cache_point() {
        let request = into_bedrock_request(vec![LanguageModelRequestMessage {
            role: Role::User,
            content: vec![MessageContent::Text("What's the weather?".into())],
            cache: true,
            reasoning_details: None,
        }]);

        assert_eq!(request.messages.len(), 1);
        assert!(
            matches!(
                request.messages[0].content().last(),
                Some(BedrockInnerContent::CachePoint(_))
            ),
            "a cache-marked message with content should end with a cache point"
        );
    }

    #[test]
    fn test_sign_mantle_request_sigv4_uses_mantle_service() {
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        );
        let body = br#"{"model":"openai.gpt-5.5"}"#;
        let mut request = HttpRequest::builder()
            .method(Method::POST)
            .uri("https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses")
            .header("Content-Type", "application/json")
            .body(AsyncBody::from(body.to_vec()))
            .unwrap();
        let time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);

        sign_mantle_request_sigv4_at(&mut request, body, &credentials, "us-east-1", time).unwrap();

        assert_eq!(
            request
                .headers()
                .get(http_client::http::header::HOST)
                .and_then(|value| value.to_str().ok()),
            Some("bedrock-mantle.us-east-1.api.aws")
        );
        assert_eq!(
            request
                .headers()
                .get("x-amz-date")
                .and_then(|value| value.to_str().ok()),
            Some("20231114T221320Z")
        );
        let authorization = request
            .headers()
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap();
        assert!(authorization.starts_with("AWS4-HMAC-SHA256 "));
        assert!(
            authorization
                .contains("Credential=AKIDEXAMPLE/20231114/us-east-1/bedrock-mantle/aws4_request")
        );
        assert!(authorization.contains("SignedHeaders=content-type;host"));
        assert!(authorization.contains("Signature="));
    }

    #[test]
    fn test_mantle_endpoint_url_preserves_protocol_path() {
        assert_eq!(
            mantle_endpoint_url("us-east-1", "openai/v1/responses"),
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
        );
        assert_eq!(
            mantle_endpoint_url("us-west-2", "anthropic/v1/messages"),
            "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages"
        );
    }

    #[test]
    fn test_mantle_anthropic_version_header_is_signed() {
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        );
        let body = br#"{"model":"anthropic.claude-mythos-5","stream":true}"#;
        let url = mantle_endpoint_url("us-east-1", "anthropic/v1/messages");
        let request = build_mantle_http_request(
            Method::POST,
            &url,
            body,
            "us-east-1",
            &MantleAuth::SigV4 { credentials },
            &CustomHeaders::default(),
            ANTHROPIC_MANTLE_HEADERS,
        )
        .unwrap();

        assert_eq!(request.uri().to_string(), url);
        assert_eq!(
            request
                .headers()
                .get("anthropic-version")
                .and_then(|value| value.to_str().ok()),
            Some("2023-06-01")
        );
        let authorization = request
            .headers()
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap();
        assert!(authorization.contains("anthropic-version"));
        assert!(authorization.contains("/us-east-1/bedrock-mantle/aws4_request"));
    }

    #[test]
    fn test_mantle_protocol_from_settings() {
        assert_eq!(
            mantle_protocol_from_settings(settings::BedrockMantleProtocolContent::ChatCompletions),
            MantleProtocol::ChatCompletions
        );
        assert_eq!(
            mantle_protocol_from_settings(settings::BedrockMantleProtocolContent::Responses),
            MantleProtocol::Responses
        );
        assert_eq!(
            mantle_protocol_from_settings(
                settings::BedrockMantleProtocolContent::AnthropicMessages
            ),
            MantleProtocol::AnthropicMessages
        );
    }

    #[test]
    fn test_mantle_anthropic_request_uses_adaptive_thinking_and_cache() {
        let request = into_mantle_anthropic(
            LanguageModelRequest {
                messages: vec![LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("Plan the change".to_string())],
                    cache: true,
                    reasoning_details: None,
                }],
                thinking_allowed: true,
                thinking_effort: Some("xhigh".to_string()),
                ..Default::default()
            },
            &MantleModel::ClaudeMythos5,
        )
        .unwrap();

        assert_eq!(request.model, "anthropic.claude-mythos-5");
        assert!(matches!(
            request.thinking,
            Some(anthropic::Thinking::Adaptive { .. })
        ));
        assert_eq!(
            request.output_config.and_then(|config| config.effort),
            Some(anthropic::Effort::XHigh)
        );
        assert!(request.cache_control.is_some());
    }

    #[test]
    fn test_mantle_opus_request_omits_unsupported_temperature() {
        let request = into_mantle_anthropic(
            LanguageModelRequest {
                messages: vec![LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("Plan the change".to_string())],
                    cache: false,
                    reasoning_details: None,
                }],
                temperature: Some(0.5),
                ..Default::default()
            },
            &MantleModel::ClaudeOpus4_7,
        )
        .unwrap();

        assert_eq!(request.temperature, None);
    }

    #[test]
    fn test_mantle_anthropic_stream_parser_preserves_signatures() {
        let event = parse_mantle_anthropic_stream_line(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}"#,
        )
        .unwrap();

        assert!(matches!(
            event,
            anthropic::Event::ContentBlockDelta {
                delta: anthropic::ContentDelta::SignatureDelta { signature },
                ..
            } if signature == "signed"
        ));
    }

    #[test]
    fn test_mantle_supported_regions_matches_docs() {
        assert!(MANTLE_SUPPORTED_REGIONS.contains(&"us-east-1"));
        assert!(MANTLE_SUPPORTED_REGIONS.contains(&"eu-west-1"));
        assert!(!MANTLE_SUPPORTED_REGIONS.contains(&"ap-southeast-1"));

        let regions = BEDROCK_REGIONS
            .iter()
            .map(|(region, _)| *region)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(regions.len(), BEDROCK_REGIONS.len());
        for mantle_region in MANTLE_SUPPORTED_REGIONS {
            assert!(regions.contains(mantle_region));
        }
    }

    fn mantle_message_item(id: &str) -> ResponseOutputItem {
        mantle_message_item_with_phase(id, None)
    }

    fn mantle_message_item_with_phase(id: &str, phase: Option<&str>) -> ResponseOutputItem {
        ResponseOutputItem::Message(ResponseOutputMessage {
            id: Some(id.to_string()),
            content: Vec::new(),
            role: Some("assistant".to_string()),
            status: Some("in_progress".to_string()),
            phase: phase.map(str::to_string),
        })
    }

    fn mantle_function_call_item(id: &str, name: &str, call_id: &str) -> ResponseOutputItem {
        ResponseOutputItem::FunctionCall(ResponseFunctionToolCall {
            id: Some(id.to_string()),
            arguments: String::new(),
            call_id: Some(call_id.to_string()),
            name: Some(name.to_string()),
            status: Some("in_progress".to_string()),
        })
    }

    fn mantle_reasoning_item(id: &str) -> ResponseOutputItem {
        ResponseOutputItem::Reasoning(ResponseReasoningItem {
            id: Some(id.to_string()),
            summary: Vec::new(),
            content: Vec::new(),
            encrypted_content: None,
            status: Some("in_progress".to_string()),
        })
    }

    fn text_delta(item_id: &str, output_index: usize, delta: &str) -> OpenAiResponseStreamEvent {
        OpenAiResponseStreamEvent::OutputTextDelta {
            item_id: item_id.to_string(),
            output_index,
            content_index: Some(0),
            delta: delta.to_string(),
        }
    }

    fn item_added(output_index: usize, item: ResponseOutputItem) -> OpenAiResponseStreamEvent {
        OpenAiResponseStreamEvent::OutputItemAdded {
            output_index,
            sequence_number: None,
            item,
        }
    }

    fn item_done(output_index: usize, item: ResponseOutputItem) -> OpenAiResponseStreamEvent {
        OpenAiResponseStreamEvent::OutputItemDone {
            output_index,
            sequence_number: None,
            item,
        }
    }

    fn map_mantle_response_events(
        events: Vec<OpenAiResponseStreamEvent>,
    ) -> Vec<LanguageModelCompletionEvent> {
        let mut mapper = MantleResponseEventMapper::new();
        events
            .into_iter()
            .flat_map(|event| mapper.map_event(event))
            .map(|event| event.expect("Mantle response event should map successfully"))
            .collect()
    }

    // Keeps only StartMessage and Text events so phase/reasoning tests can assert on
    // message boundaries without depending on incidental ReasoningDetails metadata.
    fn start_messages_and_texts(
        events: Vec<LanguageModelCompletionEvent>,
    ) -> Vec<LanguageModelCompletionEvent> {
        events
            .into_iter()
            .filter(|event| {
                matches!(
                    event,
                    LanguageModelCompletionEvent::StartMessage { .. }
                        | LanguageModelCompletionEvent::Text(_)
                )
            })
            .collect()
    }

    #[test]
    fn mantle_response_mapper_coalesces_cumulative_message_snapshots() {
        let first_message = mantle_message_item("msg_1");
        let second_message = mantle_message_item("msg_2");
        let events = map_mantle_response_events(vec![
            OpenAiResponseStreamEvent::OutputItemAdded {
                output_index: 0,
                sequence_number: None,
                item: first_message.clone(),
            },
            OpenAiResponseStreamEvent::OutputTextDelta {
                item_id: "msg_1".to_string(),
                output_index: 0,
                content_index: Some(0),
                delta: "Plan: rename".to_string(),
            },
            OpenAiResponseStreamEvent::OutputItemDone {
                output_index: 0,
                sequence_number: None,
                item: first_message,
            },
            OpenAiResponseStreamEvent::OutputItemAdded {
                output_index: 1,
                sequence_number: None,
                item: second_message.clone(),
            },
            OpenAiResponseStreamEvent::OutputTextDelta {
                item_id: "msg_2".to_string(),
                output_index: 1,
                content_index: Some(0),
                delta: "Plan: rename".to_string(),
            },
            OpenAiResponseStreamEvent::OutputTextDelta {
                item_id: "msg_2".to_string(),
                output_index: 1,
                content_index: Some(0),
                delta: " and regenerate".to_string(),
            },
            OpenAiResponseStreamEvent::OutputItemDone {
                output_index: 1,
                sequence_number: None,
                item: second_message,
            },
            OpenAiResponseStreamEvent::Completed {
                response: open_ai::responses::ResponseSummary::default(),
            },
        ]);

        assert_eq!(
            events,
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan: rename".to_string()),
                LanguageModelCompletionEvent::Text(" and regenerate".to_string()),
                LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_flushes_undetermined_message_at_eof() {
        let first_message = mantle_message_item("msg_1");
        let second_message = mantle_message_item("msg_2");
        let input = vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan"),
            item_done(0, first_message),
            item_added(1, second_message),
            text_delta("msg_2", 1, "Plan"),
        ];

        let events = futures::executor::block_on(async {
            MantleResponseEventMapper::new()
                .map_stream(
                    futures::stream::iter(input.into_iter().map(Ok::<_, anyhow::Error>)).boxed(),
                )
                .collect::<Vec<_>>()
                .await
        });
        let events = start_messages_and_texts(
            events
                .into_iter()
                .map(|event| event.expect("Mantle response event should map successfully"))
                .collect(),
        );

        assert_eq!(
            events,
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_2".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_preserves_independent_messages() {
        let first_message = mantle_message_item("msg_1");
        let second_message = mantle_message_item("msg_2");
        let events = map_mantle_response_events(vec![
            OpenAiResponseStreamEvent::OutputItemAdded {
                output_index: 0,
                sequence_number: None,
                item: first_message.clone(),
            },
            OpenAiResponseStreamEvent::OutputTextDelta {
                item_id: "msg_1".to_string(),
                output_index: 0,
                content_index: Some(0),
                delta: "Plan".to_string(),
            },
            OpenAiResponseStreamEvent::OutputItemDone {
                output_index: 0,
                sequence_number: None,
                item: first_message,
            },
            OpenAiResponseStreamEvent::OutputItemAdded {
                output_index: 1,
                sequence_number: None,
                item: second_message.clone(),
            },
            OpenAiResponseStreamEvent::OutputTextDelta {
                item_id: "msg_2".to_string(),
                output_index: 1,
                content_index: Some(0),
                delta: "Final answer".to_string(),
            },
            OpenAiResponseStreamEvent::OutputItemDone {
                output_index: 1,
                sequence_number: None,
                item: second_message,
            },
        ]);

        assert_eq!(
            events,
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_2".to_string(),
                },
                LanguageModelCompletionEvent::Text("Final answer".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_coalesces_multi_delta_cumulative_snapshots() {
        // The first message is finalized, then a second item replays the full text
        // across several growing deltas, each of which is a strict prefix extension.
        let first_message = mantle_message_item("msg_1");
        let second_message = mantle_message_item("msg_2");
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan"),
            item_done(0, first_message),
            item_added(1, second_message.clone()),
            text_delta("msg_2", 1, "Pl"),
            text_delta("msg_2", 1, "an"),
            text_delta("msg_2", 1, " and go"),
            item_done(1, second_message),
        ]);

        assert_eq!(
            events,
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
                LanguageModelCompletionEvent::Text(" and go".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_keeps_message_state_across_content_events() {
        let first_message = mantle_message_item("msg_1");
        let second_message = mantle_message_item("msg_2");
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan"),
            item_done(0, first_message),
            item_added(1, second_message.clone()),
            OpenAiResponseStreamEvent::ContentPartAdded {
                item_id: "msg_2".to_string(),
                output_index: 1,
                content_index: 0,
                part: serde_json::json!({"type": "output_text", "text": ""}),
            },
            text_delta("msg_2", 1, "Plan"),
            text_delta("msg_2", 1, " and continue"),
            OpenAiResponseStreamEvent::OutputTextDone {
                item_id: "msg_2".to_string(),
                output_index: 1,
                content_index: Some(0),
                text: "Plan and continue".to_string(),
            },
            OpenAiResponseStreamEvent::ContentPartDone {
                item_id: "msg_2".to_string(),
                output_index: 1,
                content_index: 0,
                part: serde_json::json!({"type": "output_text", "text": "Plan and continue"}),
            },
            item_done(1, second_message),
        ]);

        assert_eq!(
            events,
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
                LanguageModelCompletionEvent::Text(" and continue".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_resets_on_non_message_item_done() {
        let first_message = mantle_message_item("msg_1");
        let second_message = mantle_message_item("msg_2");
        let tool_call = mantle_function_call_item("call_1", "get_weather", "server_call_1");
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan"),
            item_done(0, first_message),
            item_done(1, tool_call),
            item_added(2, second_message.clone()),
            text_delta("msg_2", 2, "Plan continued"),
            item_done(2, second_message),
        ]);

        assert_eq!(
            start_messages_and_texts(events),
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_2".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan continued".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_keeps_post_tool_message_separate_when_extending_prior_text() {
        // A tool call sits between two messages. Even though the second message's
        // text happens to extend the first message's text, the tool call is a hard
        // boundary and the second message must remain a separate, visible message.
        let first_message = mantle_message_item("msg_1");
        let tool_call = mantle_function_call_item("call_1", "get_weather", "server_call_1");
        let second_message = mantle_message_item("msg_2");
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan"),
            item_done(0, first_message),
            item_added(1, tool_call.clone()),
            OpenAiResponseStreamEvent::FunctionCallArgumentsDone {
                item_id: "call_1".to_string(),
                output_index: 1,
                arguments: "{\"city\":\"Boston\"}".to_string(),
                sequence_number: None,
            },
            item_done(1, tool_call),
            item_added(2, second_message.clone()),
            text_delta("msg_2", 2, "Plan continued"),
            item_done(2, second_message),
        ]);

        assert_eq!(
            events,
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
                LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                    id: language_model::LanguageModelToolUseId::from("server_call_1"),
                    name: Arc::<str>::from("get_weather"),
                    is_input_complete: true,
                    input: language_model::LanguageModelToolUseInput::Json(
                        serde_json::json!({"city": "Boston"}),
                    ),
                    raw_input: "{\"city\":\"Boston\"}".to_string(),
                    thought_signature: None,
                }),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_2".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan continued".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_treats_reasoning_as_collapse_boundary() {
        // A reasoning item between two messages breaks adjacency, so the second
        // message is emitted independently even though it repeats the first's text.
        let first_message = mantle_message_item("msg_1");
        let reasoning = mantle_reasoning_item("rsn_1");
        let second_message = mantle_message_item("msg_2");
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan"),
            item_done(0, first_message),
            item_added(1, reasoning.clone()),
            item_done(1, reasoning),
            item_added(2, second_message.clone()),
            text_delta("msg_2", 2, "Plan"),
            item_done(2, second_message),
        ]);

        assert_eq!(
            start_messages_and_texts(events),
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_2".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_preserves_equal_and_shrinking_messages() {
        // Equal and strict-prefix (shrinking) adjacent messages must not be dropped:
        // they are emitted as independent messages rather than collapsed away.
        let first_message = mantle_message_item("msg_1");
        let second_message = mantle_message_item("msg_2");
        let third_message = mantle_message_item("msg_3");
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Hello world"),
            item_done(0, first_message),
            // Equal to the previous message.
            item_added(1, second_message.clone()),
            text_delta("msg_2", 1, "Hello world"),
            item_done(1, second_message),
            // Strict prefix of (shorter than) the previous message.
            item_added(2, third_message.clone()),
            text_delta("msg_3", 2, "Hello"),
            item_done(2, third_message),
        ]);

        assert_eq!(
            events,
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Hello world".to_string()),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_2".to_string(),
                },
                LanguageModelCompletionEvent::Text("Hello world".to_string()),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_3".to_string(),
                },
                LanguageModelCompletionEvent::Text("Hello".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_keeps_different_phase_messages_separate() {
        // Two messages share a text prefix but have different phases; only same-phase
        // strict extensions collapse, so the second message stays visible.
        let first_message = mantle_message_item_with_phase("msg_1", Some("commentary"));
        let second_message = mantle_message_item_with_phase("msg_2", Some("final_answer"));
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan: rename"),
            item_done(0, first_message),
            item_added(1, second_message.clone()),
            text_delta("msg_2", 1, "Plan: rename and regenerate"),
            item_done(1, second_message),
        ]);

        assert_eq!(
            start_messages_and_texts(events),
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan: rename".to_string()),
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_2".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan: rename and regenerate".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_coalesces_same_phase_strict_extension() {
        // Same phase, strict extension: the replayed prefix is collapsed and only the
        // new suffix is emitted, so no second StartMessage appears.
        let first_message = mantle_message_item_with_phase("msg_1", Some("final_answer"));
        let second_message = mantle_message_item_with_phase("msg_2", Some("final_answer"));
        let events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan: rename"),
            item_done(0, first_message),
            item_added(1, second_message.clone()),
            text_delta("msg_2", 1, "Plan: rename"),
            text_delta("msg_2", 1, " and regenerate"),
            item_done(1, second_message),
        ]);

        assert_eq!(
            start_messages_and_texts(events),
            vec![
                LanguageModelCompletionEvent::StartMessage {
                    message_id: "msg_1".to_string(),
                },
                LanguageModelCompletionEvent::Text("Plan: rename".to_string()),
                LanguageModelCompletionEvent::Text(" and regenerate".to_string()),
            ]
        );
    }

    #[test]
    fn mantle_response_mapper_does_not_duplicate_reasoning_details_on_merge() {
        // Merging a second message into the first must not emit any additional
        // phase/reasoning metadata events beyond what the first message alone would
        // have produced, since the metadata can't have changed by merging.
        fn reasoning_details_count(events: &[LanguageModelCompletionEvent]) -> usize {
            events
                .iter()
                .filter(|event| matches!(event, LanguageModelCompletionEvent::ReasoningDetails(_)))
                .count()
        }

        let first_message = mantle_message_item_with_phase("msg_1", Some("final_answer"));
        let baseline_events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan: rename"),
            item_done(0, first_message.clone()),
        ]);

        let second_message = mantle_message_item_with_phase("msg_2", Some("final_answer"));
        let merged_events = map_mantle_response_events(vec![
            item_added(0, first_message.clone()),
            text_delta("msg_1", 0, "Plan: rename"),
            item_done(0, first_message),
            item_added(1, second_message.clone()),
            text_delta("msg_2", 1, "Plan: rename"),
            text_delta("msg_2", 1, " and regenerate"),
            item_done(1, second_message),
        ]);

        assert_eq!(
            reasoning_details_count(&merged_events),
            reasoning_details_count(&baseline_events),
        );
    }

    #[test]
    fn test_builtin_mantle_models_support_thinking() {
        assert!(MantleModel::Gpt5_6Sol.supports_thinking());
        assert!(MantleModel::Gpt5_6Terra.supports_thinking());
        assert!(MantleModel::Gpt5_6Luna.supports_thinking());
        assert!(MantleModel::Gpt5_5.supports_thinking());
        assert!(MantleModel::Gpt5_4.supports_thinking());
        assert!(MantleModel::Grok4_3.supports_thinking());
        assert_eq!(
            mantle_default_reasoning_effort(&MantleModel::Gpt5_6Sol),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(
            mantle_default_reasoning_effort(&MantleModel::Gpt5_6Terra),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(
            mantle_default_reasoning_effort(&MantleModel::Gpt5_6Luna),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(
            mantle_default_reasoning_effort(&MantleModel::Gpt5_5),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(
            mantle_default_reasoning_effort(&MantleModel::Grok4_3),
            Some(ReasoningEffort::Medium)
        );
    }

    #[test]
    fn test_mantle_supported_effort_levels_hide_none() {
        let effort_levels = mantle_supported_effort_levels(&MantleModel::Gpt5_5);
        let values = effort_levels
            .iter()
            .map(|level| level.value.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(values, ["low", "medium", "high", "xhigh"]);
        assert_eq!(
            effort_levels
                .iter()
                .find(|level| level.is_default)
                .map(|level| level.value.as_ref()),
            Some("medium")
        );
    }

    #[test]
    fn test_custom_mantle_model_can_disable_thinking() {
        let model = MantleModel::Custom {
            name: "custom-mantle-model".to_string(),
            display_name: None,
            max_tokens: 128_000,
            max_output_tokens: None,
            protocol: MantleProtocol::Responses,
            supports_tools: true,
            supports_images: false,
            supports_thinking: false,
        };

        assert!(!model.supports_thinking());
        assert_eq!(mantle_default_reasoning_effort(&model), None);
        assert!(mantle_supported_effort_levels(&model).is_empty());
        assert_eq!(
            mantle_selected_reasoning_effort(
                &LanguageModelRequest {
                    thinking_effort: Some("high".to_string()),
                    ..Default::default()
                },
                &model,
            ),
            None
        );
    }

    #[test]
    fn test_disabled_mantle_thinking_serializes_none() {
        let request = into_open_ai_response(
            LanguageModelRequest {
                thinking_allowed: false,
                ..Default::default()
            },
            MantleModel::Grok4_3.request_id(),
            true,
            false,
            Some(MantleModel::Grok4_3.max_output_tokens()),
            mantle_default_reasoning_effort(&MantleModel::Grok4_3),
            MantleModel::Grok4_3.supports_thinking(),
        );

        assert_eq!(
            serde_json::to_value(&request).unwrap()["reasoning"],
            serde_json::json!({ "effort": "none" })
        );
    }

    #[test]
    fn test_mantle_reasoning_passes_known_efforts_through() {
        for effort in ["low", "medium", "high", "xhigh", "minimal", "max"] {
            assert_eq!(
                mantle_selected_reasoning_effort(
                    &LanguageModelRequest {
                        thinking_allowed: true,
                        thinking_effort: Some(effort.to_string()),
                        ..Default::default()
                    },
                    &MantleModel::Gpt5_5,
                )
                .map(|effort| effort.value()),
                Some(effort)
            );
        }

        assert_eq!(
            mantle_selected_reasoning_effort(
                &LanguageModelRequest {
                    thinking_allowed: true,
                    thinking_effort: Some("none".to_string()),
                    ..Default::default()
                },
                &MantleModel::Gpt5_5,
            ),
            Some(ReasoningEffort::Medium)
        );
    }

    #[test]
    fn test_strip_unsupported_mantle_response_fields_removes_context_management() {
        let mut request = into_open_ai_response(
            LanguageModelRequest {
                compact_at_tokens: Some(10_000),
                ..Default::default()
            },
            "openai.gpt-5.5",
            true,
            false,
            Some(128_000),
            Some(ReasoningEffort::Medium),
            false,
        );

        assert!(request.context_management.is_some());
        strip_unsupported_mantle_response_fields(&mut request);
        assert!(request.context_management.is_none());

        let request = serde_json::to_value(&request).unwrap();
        assert!(request.get("context_management").is_none());
    }
}
