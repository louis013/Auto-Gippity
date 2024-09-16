use crate::apis::call_request::call_gpt;
use crate::models::general::llm::Message;
use crate::helpers::command_line::PrintCommand;

// Extend ai function to encourage specific output
pub fn extend_ai_function(ai_func: fn(&str) -> &'static str, func_input: &str) -> Message {
    let ai_function_str = ai_func(func_input);

    // Extend the string to encourage only printing the output
    let msg: String = format!("FUNCTION {} INSTRUCTION: You are a function printer. You ONLY print the results of functions. Nothing else, No commentary. Here is the input to the function {}.
    Print out what the function will return.", 
    ai_function_str, func_input);

    // Return Message
    Message {
        role: "system".to_string(),
        content: msg
    }
}

// Performs call to LLM GPT
pub async fn ai_task_request(
    msg_context: String,
    agent_position: &str,
    agent_operation: &str,
    function_pass: for<'a> fn(&'a str) -> &'static str,
) -> String {
    // Extend AI function
    let extended_msg: Message = extend_ai_function(function_pass, &msg_context);

    // Print current status
    PrintCommand::AICall.print_agent_message(agent_position, agent_operation);

    // Get LLM response
    let llm_response_res: Result<String, Box<dyn std::error::Error + Send>> =
        call_gpt(vec![extended_msg.clone()]).await;

    // Return Success or try again
    match llm_response_res {
        Ok(llm_resp) => llm_resp,
        Err(_) => call_gpt(vec![extended_msg.clone()])
            .await
            .expect("Failed twice to call OpenAI"),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ai_functions::managing::convert_user_input_to_goal;

    #[test]
    fn test_extending_ai_function() {

        let extended_msg: Message = extend_ai_function(convert_user_input_to_goal, "dummy variable");

        assert_eq!(extended_msg.role, "system".to_string());

    }

    #[tokio::test]
    async fn tests_ai_task_request() {
        let ai_func_param: String =
            "Build me a webserver for making stock price api requests.".to_string();

        let res: String = ai_task_request(
            ai_func_param,
            "Managing Agent",
            "Defining user requirements",
            convert_user_input_to_goal,
        )
        .await;

        assert!(res.len() > 20);
    }
}