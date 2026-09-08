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

    pub fn get_tasks(&self, difficulty: u32, count: u32) -> Result<Vec<Task>> {
        let resp = self
            .http
            .get(format!("{}/api/mining/task", self.base))
            .bearer_auth(&self.token)
            .query(&[("difficulty", difficulty), ("count", count)])
            .send()
            .context("request failed")?;

        if resp.status() == 401 {
            bail!("Unauthorized. Check your --token parameter.");
        }
        if !resp.status().is_success() {
            bail!("HTTP {}", resp.status());
        }

        let body: GetTasksResponse = resp.json().context("invalid JSON")?;
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

#[derive(Deserialize)]
pub struct SubmitResponse {
    #[serde(rename = "accepted_count")]
    pub accepted: u64,
    #[serde(rename = "consolidated_blocks")]
    pub consolidated: u64,
    pub new_balance: f64,
}
