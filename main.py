import openai

# Replace with your OpenAI API key
#TODO


def modify_summary(summarized_article, knowledge_base):
    """
    This function takes a summarized article and a knowledge base, then prompts the model
    to modify the summary based on the given knowledge using the chat API.
    """
    # OpenAI GPT prompt to modify the existing summary using prior knowledge
    knowledge_list = ', '.join(knowledge_base)  # Join knowledge base items into a comma-separated string
    prompt = f"""
    I am already up to date about {knowledge_list}. 
    Please summarize the article again by removing any information related to this knowledge

    Summarized Article:
    {summarized_article}

    New Modified Summary:
    """

    try:
        # Make the API call to generate the modified summary using GPT-3.5's chat endpoint
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,  # Allow room for a detailed summary
            temperature=0.5,  # Adjust creativity (0.0-1.0)
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        # Extract the new modified summary from the response
        new_summary = response['choices'][0]['message']['content'].strip()
        return new_summary
    except Exception as e:
        return f"Error modifying the summary: {e}"

# Example usage
if __name__ == "__main__":
    zero_shot_summary = "The Kansas City Chiefs have traded for wide receiver DeAndre Hopkins from the Tennessee Titans, sending a conditional 2025 fifth-round pick in exchange. If the Chiefs reach the Super Bowl and Hopkins plays 60% of the snaps, the pick will be upgraded to a fourth-rounder, with Tennessee also covering $2.5 million of Hopkins' remaining salary. The Chiefs needed reinforcements at wide receiver after several injuries, including season-ending surgery for Rashee Rice and hamstring issues for JuJu Smith-Schuster. Hopkins’ addition gives Patrick Mahomes another target as Kansas City aims to strengthen their offense. For the Titans, this trade marks a shift toward rebuilding, as they also dealt linebacker Ernest Jones IV to Seattle."
    
    # Example of previous knowledge you want to incorporate
    knowledge_base = [
        "Recent NFL News", 
        "Names of NFL Players",
        "Chiefs injuries"
    ]
    
    new_summary = modify_summary(zero_shot_summary, knowledge_base)
    print("Original Zero Shot Summary:", zero_shot_summary)
    print("\n")
    print("New Modified Summary:", new_summary)

