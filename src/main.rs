mod config;
mod provider;
mod proxy;
mod router;
mod types;
mod ui;

use axum::{
    routing::{get, post},
    Router,
};
use config::Config;
use proxy::{anthropic_messages, health, openai_responses, AppState};
use router::ModelRouter;
use std::net::SocketAddr;
use tracing::info;
use ui::{builder, presets_js, resolve_preview, test_stream, test_upstream};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "prism=info".into()))
        .with_target(false)
        .compact()
        .init();

    let config = Config::load()?;
    let port = config.port;
    let routes = config.backends.len();
    let state = AppState::new(ModelRouter::new(config.backends));

    let app = Router::new()
        .route("/", get(builder))
        .route("/builder", get(builder))
        .route("/presets.js", get(presets_js))
        .route("/healthz", get(health))
        .route("/api/test-upstream", post(test_upstream))
        .route("/api/test-stream", post(test_stream))
        .route("/api/resolve-preview", post(resolve_preview))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!(
        "Prism listening on http://{} with {} route(s)",
        listener.local_addr()?,
        routes
    );

    axum::serve(listener, app).await?;
    Ok(())
}
