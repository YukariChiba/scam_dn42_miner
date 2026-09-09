use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::task::{Solution, Task};

#[derive(Deserialize)]
struct GetTasksResponse {
    #[serde(default)]
    tasks: Vec<Task>,
}

pub struct Client {
    http: reqwest::blocking::Client,
    base: String,
    token: String,
}

impl Client {
    pub fn new(base: String, token: String) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .danger_accept_invalid_certs(true)
            .user_agent("Scummybank-Miner/3.0")
            .build()?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_string(),
            token,
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn get_tasks(&self, difficulty: u32, count: u32) -> std::result::Result<Vec<Task>, ApiError> {
        let resp = self
            .http
            .get(format!("{}/api/mining/task", self.base))
            .bearer_auth(&self.token)
            .query(&[("difficulty", difficulty), ("count", count)])
            .send()
            .map_err(|e| ApiError::Other(format!("Request failed: {e}")))?;

        let status = resp.status();
        if status == 401 {
            return Err(ApiError::Unauthorized);
        }
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            let error = extract_error(&body).unwrap_or_else(|| body.clone());
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                return Err(ApiError::RateLimited(error));
            }
            return Err(ApiError::Other(format!("Server returned HTTP {status}: {body}")));
        }

        let body: GetTasksResponse = resp
            .json()
            .map_err(|e| ApiError::Other(format!("Invalid response: {e}")))?;
        Ok(body.tasks)
    }

    pub fn submit(&self, solutions: &[Solution], account: Option<&str>) -> Result<SubmitResponse> {
        let mut payload = serde_json::json!({ "solutions": solutions });
        if let Some(a) = account {
            payload["account_number"] = serde_json::Value::String(a.to_string());
        }

        let resp = self
            .http
            .post(format!("{}/api/mining/submit_batch", self.base))
            .bearer_auth(&self.token)
            .json(&payload)
            .send()
            .context("submit request failed")?;

        let status = resp.status();
        let data: serde_json::Value = resp.json().unwrap_or(serde_json::Value::Null);
        if !status.is_success() {
            bail!(
                "{}",
                data.get("error")
                    .and_then(|e| e.as_str())
                    .unwrap_or("batch rejected")
            );
        }

        serde_json::from_value(data).context("invalid submit response")
    }
}

/// Structured error from the mining task endpoint, so the orchestrator can
/// distinguish rate-limiting and auth failures from generic server errors.
#[derive(Debug)]
pub enum ApiError {
    RateLimited(String),
    Unauthorized,
    Other(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::RateLimited(m) => write!(f, "{m}"),
            ApiError::Unauthorized => write!(f, "Unauthorized. Check your --token parameter."),
            ApiError::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ApiError {}

/// Extract the server-provided `error` string from a JSON error body.
fn extract_error(body: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
}

#[derive(Deserialize)]
pub struct SubmitResponse {
    #[serde(rename = "accepted_count")]
    pub accepted: u64,
    #[serde(rename = "consolidated_blocks")]
    pub consolidated: u64,
    pub new_balance: f64,
    #[serde(default)]
    pub results: Vec<SubmitResult>,
}

#[derive(Deserialize)]
pub struct SubmitResult {
    pub status: String,
    #[serde(default)]
    pub reward: f64,
    #[serde(default)]
    pub error: Option<String>,
}
