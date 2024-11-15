

import openai
import pandas as pd


def agent_turn(system_prompt, conversation_history, model_nm="gpt-3.5-turbo"):
    # Build the messages list starting with the system prompt
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    response = openai.ChatCompletion.create(
        model=model_nm,
        messages=messages
    )

    assistant_reply = response['choices'][0]['message']

    # Append assistant's reply to the conversation history
    conversation_history.append(assistant_reply)

    return assistant_reply['content']


def extract_known_information(final_summary, curr_knowledge, model_nm="gpt-3.5-turbo"):
    """
    Use the Personalization Agent to identify known information in the final summary.
    """
    # Initialize conversation history with the final summary
    conversation_history = [{"role": "assistant", "content": final_summary}]

    # Ask the Personalization Agent to identify known topics based on the summary
    extraction_prompt = (
        f"Known information from the summary includes:\n{curr_knowledge}\n\n"
        "Please identify the topics and details in the summary that align with the user's known interests. "
        "Exclude any known information and only list new or missing details. Please store them in key phrases separated by semicolons."
    )

    # Run the Personalization Agent to extract known information
    known_info_response = agent_turn(extraction_prompt, conversation_history, model_nm)
    print(f"Extracted Known Information:\n{known_info_response}\n")

    # Return the known information extracted by the Personalization Agent
    return known_info_response


def update_personalization_prompt(original_prompt, known_info_response):
    """
    Update the Personalization Agent's prompt with the known information.
    """
    # Use the known information response to refine the feedback prompt
    updated_prompt = (
        f"{original_prompt}\n\n"
        f"Known information from the summary includes:\n{known_info_response}\n\n"
        "Please focus your feedback on any additional details or gaps not covered by this information."
    )
    return updated_prompt


def update_agent_knowledge(df_agent, agent_id, known_info):
    """
    Update the knowledge column for the specified agent in df_agent.
    """
    # Retrieve the current knowledge, handling NaN values
    current_knowledge = df_agent.loc[df_agent['agent_id'] == agent_id, 'knowledge'].values[0]
    
    # If current_knowledge is NaN, set it to an empty string
    if pd.isna(current_knowledge):
        current_knowledge = ""

    # Append known information to the existing knowledge
    updated_knowledge = current_knowledge + "; " + known_info if current_knowledge else known_info
    df_agent.loc[df_agent['agent_id'] == agent_id, 'knowledge'] = updated_knowledge


def agent_conversation(df_article, articledb, num_iterations, summary_agent, persona, model_nm="gpt-3.5-turbo"):
    # Load agent prompts and knowledge from the original CSV file
    df_agent = pd.read_csv('data/agents.csv', encoding='ISO-8859-1')

    # Get original prompts for each agent
    summary_prompt = df_agent.loc[df_agent['agent_id'] == summary_agent, 'prompt'].values[0]

    # Update the Personalization Agent prompt with known information
    persona_prompt = df_agent.loc[df_agent['agent_id'] == persona, 'prompt'].values[0]
    curr_knowledge = df_agent.loc[df_agent['agent_id'] == persona, 'knowledge'].values[0]
    updated_persona_prompt = update_personalization_prompt(persona_prompt, curr_knowledge)

    ids = 0
    data_feedback = []

    for index, article in df_article.iterrows():
        print(f"\nProcessing Article ID: {article['article_id']}")

        # Initialize conversation history
        conversation_history = []

        # Create the user message with the article content
        user_message = {
            "role": "user",
            "content": (
                f"Article ID: {article['article_id']}\n"
                f"Date: {article['Date of Article']}\n"
                f"Title: {article['Title']}\n"
                f"Body: {article['Body']}"
            )
        }
        conversation_history.append(user_message)

        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1} ---")

            # Leverage known knowledge in the Summary Agent's prompt
            current_knowledge = df_agent.loc[df_agent['agent_id'] == summary_agent, 'knowledge'].values[0]
            dynamic_summary_prompt = f"{summary_prompt}\n\nKnown information: {current_knowledge}"

            # Summary Agent generates a summary
            summary_response = agent_turn(dynamic_summary_prompt, conversation_history, model_nm)
            print(f"Summary Agent Response:\n{summary_response}\n")
            # Append the assistant's reply to the conversation history
            conversation_history.append({
                "role": "assistant",
                "content": summary_response
            })

            # Personalization Agent provides feedback
            feedback_response = agent_turn(updated_persona_prompt, conversation_history, model_nm)
            print(f"Personalization Agent Feedback:\n{feedback_response}\n")
            # Append the assistant's reply to the conversation history
            conversation_history.append({
                "role": "assistant",
                "content": feedback_response
            })

            # Update knowledge base with known information from the final summary
            data_feedback.append([
                ids, 
                persona, 
                article['article_id'], 
                article['Title'],
                summary_response, 
                feedback_response
            ])

            ids += 1

    # Final summary
    summary_response = agent_turn(dynamic_summary_prompt, conversation_history, model_nm)
    print(f"Final Summary Agent Response:\n{summary_response}\n")
    # Append the assistant's reply to the conversation history
    conversation_history.append({
        "role": "assistant",
        "content": summary_response
    })

    # Extract known information from the final summary using the Personalization Agent
    known_info_response = extract_known_information(summary_response, curr_knowledge, model_nm)

    # Update the Personalization Agent prompt with known information
    updated_persona_prompt = update_personalization_prompt(persona_prompt, known_info_response)

    # Run the final check with the updated Personalization Agent prompt
    final_feedback_response = agent_turn(updated_persona_prompt, conversation_history, model_nm)
    print(f"Final Personalized Feedback:\n{final_feedback_response}\n")

    # Update the knowledge base with final known information
    update_agent_knowledge(df_agent, persona, known_info_response)

    data_feedback.append([
        ids,
        persona,
        article['article_id'],
        article['Title'],
        summary_response,
        ""
    ])

    # Convert the 2D list to a DataFrame
    columns_feedback = ['conversation_id', 'persona_id', 'article_id', 'Title', 'summary', 'feedback']
    df_feedback = pd.DataFrame(data_feedback, columns=columns_feedback)
    df_feedback.to_csv(f'data/feedback_{summary_agent}_{persona}_{articledb}.csv', index=False)

    # Overwrite the original df_agent with updated knowledge
    df_agent.to_csv('data/agents.csv', index=False,encoding='ISO-8859-1')

    return df_feedback[df_feedback['conversation_id'] == ids]["summary"], df_feedback


if __name__ == "__main__":
    with open('../open_ai_api_key.txt', 'r') as file:
        api_key = file.readline().strip()
    openai.api_key = api_key

    df_article_tech = pd.read_csv('data/tech_articles.csv')
    df_article_tech = df_article_tech.rename(columns={'ID': 'article_id'})

    qualcomm_article = df_article_tech[df_article_tech["Title"] == "Arm is giving Qualcomm the wake-up call it needs"]

    final_summaries, df_feedback = agent_conversation(
        df_article=qualcomm_article,
        articledb="qualcomm_article",
        num_iterations=3,
        summary_agent='sa9',
        persona='jc4',
        model_nm="gpt-3.5-turbo"
    )
