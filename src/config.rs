//! Configuration loading and validation.
//!
//! Loads TOML from `$PRISM_CONFIG` (or `./prism.toml`), resolves API keys from
//! inline fields or environment variables, and validates the result:
//! - every backend must carry a non-empty API key (from `api_key` or `api_key_env`),
//! - route prefixes must be unique and non-empty,
//! - port defaults to [`DEFAULT_PORT`] when unspecified.
//!
//! Errors surface as [`ConfigError`] with human-readable messages suitable for
//! printing at startup; anything invalid aborts the process before binding.

use crate::{
    provider::ProviderKind,
    types::{Backend, FileConfig, RouteConfigSource},
};
use std::{collections::HashSet, env, fs, num::ParseIntError};
use thiserror::Error;
use url::Url;

const DEFAULT_PORT: u16 = 8088;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub backends: Vec<Backend>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing API key environment variable `{0}`")]
    MissingApiKeyEnv(String),
    #[error("route `{prefix}` must define either `api_key` or `key_env`")]
    MissingApiKeySource { prefix: String },
    #[error("invalid PRISM_PORT value: {0}")]
    InvalidPort(#[from] ParseIntError),
    #[error("invalid route entry `{entry}`: {reason}")]
    InvalidRoute { entry: String, reason: String },
    #[error("duplicate route prefix `{0}`")]
    DuplicatePrefix(String),
    #[error("failed to read config file `{path}`: {source}")]
    ReadFile {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML config `{path}`: {source}")]
    ParseToml {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid backend base URL `{url}`: {source}")]
    InvalidBaseUrl {
        url: String,
        #[source]
        source: url::ParseError,
    },
}

impl Config {
    pub fn load() -> Result<Self, ConfigError> {
        let file_config = load_file_config()?;
        let port = env::var("PRISM_PORT")
            .ok()
            .map(|value| value.parse::<u16>())
            .transpose()?
            .or_else(|| file_config.as_ref().and_then(|cfg| cfg.port))
            .unwrap_or(DEFAULT_PORT);

        let route_sources = if let Ok(raw_routes) = env::var("PRISM_ROUTES") {
            parse_routes_env(&raw_routes)?
        } else if let Some(file_config) = file_config {
            file_config.routes
        } else {
            Vec::new()
        };

        let mut backends = Vec::with_capacity(route_sources.len());
        let mut seen_prefixes = HashSet::new();
        for route in route_sources {
            let prefix = route.prefix.trim().to_string();
            if prefix.is_empty() {
                return Err(ConfigError::InvalidRoute {
                    entry: route.base.clone(),
                    reason: "`prefix` is required and must be non-empty".into(),
                });
            }
            if !seen_prefixes.insert(prefix.clone()) {
                return Err(ConfigError::DuplicatePrefix(prefix));
            }

            let (api_key, credential_label) = resolve_api_key(&route, &prefix)?;
            let base = normalize_base_url(&route.base)?;
            let provider = ProviderKind::resolve(route.provider, &base);
            let default_model = route
                .model
                .as_ref()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(ToString::to_string);

            backends.push(Backend {
                prefix,
                provider,
                base,
                api_key,
                credential_label,
                default_model,
                anthropic_format: route.anthropic_format.unwrap_or(false),
            });
        }

        Ok(Self { port, backends })
    }
}

fn load_file_config() -> Result<Option<FileConfig>, ConfigError> {
    let Some(path) = env::var("PRISM_CONFIG").ok() else {
        return Ok(None);
    };

    let contents = fs::read_to_string(&path).map_err(|source| ConfigError::ReadFile {
        path: path.clone(),
        source,
    })?;

    let parsed = toml::from_str(&contents).map_err(|source| ConfigError::ParseToml {
        path: path.clone(),
        source,
    })?;

    Ok(Some(parsed))
}

fn parse_routes_env(raw: &str) -> Result<Vec<RouteConfigSource>, ConfigError> {
    raw.split(';')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(parse_route_entry)
        .collect()
}

fn parse_route_entry(entry: &str) -> Result<RouteConfigSource, ConfigError> {
    let mut prefix = None;
    let mut provider = None;
    let mut base = None;
    let mut key_env = None;
    let mut api_key = None;
    let mut model = None;
    let mut anthropic_format = None;

    for part in entry
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| ConfigError::InvalidRoute {
                entry: entry.to_string(),
                reason: "expected key=value pairs".into(),
            })?;

        match key.trim() {
            "prefix" => prefix = Some(value.trim().to_string()),
            "provider" => {
                provider = Some(parse_provider(value.trim()).map_err(|reason| {
                    ConfigError::InvalidRoute {
                        entry: entry.to_string(),
                        reason,
                    }
                })?)
            }
            "base" | "url" => base = Some(value.trim().to_string()),
            "key_env" | "api_key_env" => key_env = Some(value.trim().to_string()),
            "api_key" => api_key = Some(value.trim().to_string()),
            "model" => model = Some(value.trim().to_string()),
            "anthropic_format" | "anthropic-format" => {
                anthropic_format = Some(parse_bool_field(value.trim()).map_err(|reason| {
                    ConfigError::InvalidRoute {
                        entry: entry.to_string(),
                        reason,
                    }
                })?);
            }
            unknown => {
                return Err(ConfigError::InvalidRoute {
                    entry: entry.to_string(),
                    reason: format!("unsupported field `{unknown}`"),
                })
            }
        }
    }

    Ok(RouteConfigSource {
        prefix: prefix.ok_or_else(|| ConfigError::InvalidRoute {
            entry: entry.to_string(),
            reason: "missing `prefix`".into(),
        })?,
        provider,
        base: base.ok_or_else(|| ConfigError::InvalidRoute {
            entry: entry.to_string(),
            reason: "missing `base`".into(),
        })?,
        key_env,
        api_key,
        model,
        anthropic_format,
    })
}

fn parse_bool_field(raw: &str) -> Result<bool, String> {
    match raw.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" | "" => Ok(false),
        other => Err(format!("expected a boolean, got `{other}`")),
    }
}

fn normalize_base_url(raw: &str) -> Result<Url, ConfigError> {
    let mut url = Url::parse(raw).map_err(|source| ConfigError::InvalidBaseUrl {
        url: raw.to_string(),
        source,
    })?;

    if !url.path().ends_with('/') {
        let current = url.path().trim_end_matches('/');
        let normalized = if current.is_empty() {
            "/".to_string()
        } else {
            format!("{current}/")
        };
        url.set_path(&normalized);
    }

    Ok(url)
}

fn resolve_api_key(
    route: &RouteConfigSource,
    prefix: &str,
) -> Result<(String, String), ConfigError> {
    if let Some(api_key) = route
        .api_key
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        return Ok((api_key.to_string(), "inline api_key".into()));
    }

    if let Some(key_env) = route
        .key_env
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        let api_key =
            env::var(key_env).map_err(|_| ConfigError::MissingApiKeyEnv(key_env.to_string()))?;
        return Ok((api_key, format!("env:{key_env}")));
    }

    Err(ConfigError::MissingApiKeySource {
        prefix: prefix.to_string(),
    })
}

fn parse_provider(raw: &str) -> Result<ProviderKind, String> {
    match raw {
        "auto" => Ok(ProviderKind::Auto),
        "openai_compatible" | "openai-compatible" | "openai" => Ok(ProviderKind::OpenAiCompatible),
        "fireworks" => Ok(ProviderKind::Fireworks),
        "zai" | "z.ai" | "z_ai" => Ok(ProviderKind::Zai),
        other => Err(format!("unsupported provider `{other}`")),
    }
}
