import openai
import pandas as pd


def agent_turn(system_prompt, conversation_history,model_nm = "gpt-3.5-turbo"):
    # Build the messages list starting with the system prompt
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    response = openai.ChatCompletion.create(
        model=model_nm,  # or "gpt-4" if you have access
        messages=messages
    )

    assistant_reply = response['choices'][0]['message']

    # Append assistant's reply to the conversation history
    conversation_history.append(assistant_reply)

    return assistant_reply['content']

def agent_conversation(df_article,articledb,num_iterations,summary_agent,persona,model_nm = "gpt-3.5-turbo"):
    # For each article
    df_agent= pd.read_csv('data/agents.csv',encoding='ISO-8859-1')

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

            # Summary Agent generates a summary
            summary_response = agent_turn(df_agent.loc[df_agent['agent_id'] == summary_agent, 'prompt'].values[0], conversation_history,model_nm)
            print(f"Summary Agent Response:\n{summary_response}\n")
            # Append the assistant's reply to the conversation history
            conversation_history.append({
                "role": "assistant",
                "content": summary_response
            })

            # Personalization Agent provides feedback
            feedback_response = agent_turn(df_agent.loc[df_agent['agent_id'] ==persona, 'prompt'].values[0], conversation_history,model_nm)
            print(f"Personalization Agent Feedback:\n{feedback_response}\n")
            # Append the assistant's reply to the conversation history
            conversation_history.append({
                "role": "assistant",
                "content": feedback_response
            })

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
    summary_response = agent_turn(df_agent.loc[df_agent['agent_id'] == summary_agent, 'prompt'].values[0], conversation_history,model_nm)
    print(f"Summary Agent Response:\n{summary_response}\n")
    # Append the assistant's reply to the conversation history
    conversation_history.append({
        "role": "assistant",
        "content": summary_response
    })

    data_feedback.append([
                ids, 
                persona, 
                article['article_id'], 
                article['Title'],
                summary_response, 
                ""
            ])
    

    # Convert the 2D list to a DataFrame
    columns_feedback = ['conversation_id','persona_id','article_id','Title' , 'summary','feedback']
    df_feedback = pd.DataFrame(data_feedback, columns=columns_feedback)
    df_feedback.to_csv(f'data/feedback_{summary_agent}_{persona}_{articledb}.csv', index=False)

    return df_feedback[df_feedback['conversation_id'] == ids]["summary"] ,df_feedback 



if __name__ == "__main__":
    with open('../open_ai_api_key.txt', 'r') as file:
        api_key = file.readline().strip()
    openai.api_key = api_key

    df_article_tech = pd.read_csv('data/tech_articles.csv')
    df_article_tech = df_article_tech.rename(columns={'ID': 'article_id'})

    #qualcomm_article = df_article_tech[df_article_tech["Title"] == "Arm is giving Qualcomm the wake-up call it needs"]

    final_summaries,df_feedback = agent_conversation(
        df_article=df_article_tech,
        articledb = "tech",
        num_iterations=3,
        summary_agent='sa9',
        persona='jc4',
        model_nm = "gpt-4"
    )
