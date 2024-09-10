use crate::models::general::llm::{ Message, ChatCompletion };
use dotenv::dotenv;
use reqwest::Client;
use std::env;

use reqwest::header::{ HeaderMap, HeaderValue };

// Call Large Language Model (GPT-3.5 - Turbo)
pub async fn call_gpt(messages: Vec<Message>) {
    dotenv().ok();

    // Extract API Key information
    let api_key: String = env::var("OPEN_AI_KEY").expect("OPEN_AI_KEY not found in environment variables");
    let api_org: String = env::var("OPEN_AI_ORG").expect("OPEN_AI_ORG not found in environment variables");

    // Confirm endpoint
    let url: &str = "https://api.openai.com/v1/chat/completions";

    // Create headers
    let mut headers = HeaderMap::new();

    // Create api key header
    headers.insert(
        "authorization", 
        HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap()
    );

    // Create Open AI Org header
    headers.insert(
        "OpenAI-Organization", 
        HeaderValue::from_str(api_org.as_str()).unwrap()
    );

    // Create client
    let client = Client::builder()
        .default_headers(headers)
        .build()
        .unwrap();

    // Create chat completion
    let chat_completion: ChatCompletion = ChatCompletion {
        model: "gpt-3.5-turbo".to_string(),
        messages,
        temperature: 0.1
    };

    // Troubleshooting
    let res_raw = client
        .post(url)
        .json(&chat_completion)
        .send()
        .await
        .unwrap();

    dbg!(res_raw.text().await.unwrap());
}