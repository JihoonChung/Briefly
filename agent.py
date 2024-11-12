import openai
import os
import logging
import pandas as pd

# API Key
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
openai.api_key = os.environ["OPENAI_API_KEY"] # store api key in env var

class Agent: # Base agent class
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(name)
        self.configure_logger()

    def configure_logger(self): # logging code generated using chatgpt, need to check it works lol
        if not self.logger.hasHandlers():
            # Set logging level and define the log format
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # File handler for logging to a file
            log_file = "agent.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Console handler for logging to the console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def call_api(self, sys_msg, usr_msg):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {
                        "role": "user", 
                        "content": usr_msg
                    }
                ],
                max_tokens=2048,
                temperature=1,
                top_p=1.0,
                frequency_penalty=0.05,
                presence_penalty=0.05
            )
            output = response.choices[0].message.content.strip()
            self.logger.info("Received response: %s", output)
            return output
        except Exception as e:
            self.logger.error(f"Error in API call: {e}")
            return None

    def process(self, *args, **kwargs):
        raise NotImplementedError("Implement in the agent subclasses")
    
class SummarizationAgent(Agent): # Summarization Agent
    def __init__(self):
        super().__init__("Summarization Agent")
        self.summaries = []
    
    def process(self, article, feedback=None):
        # Log the start of the summarization process
        self.logger.info("Starting summarization...")
        
        # Prompts
        if feedback: # Refine summary based on feedback from Personalization Agent
            self.logger.debug("Refining summary based on feedback...")
            sys_msg = "You are a summarization agent who refines summaries based on feedback."
            usr_msg = f"Refine the summary with this feedback: {feedback}"
        else: # Initial zero-shot summary
            self.logger.debug("Generating initial summary...")
            sys_msg = "You are a summarization agent."
            usr_msg = f"Please summarize the following article in three sentences: {article}"
        
        summary = self.call_api(sys_msg, usr_msg) # Generate summary based on prompts
        if summary:
            self.logger.info("Summary generated successfully.")
            self.summaries.append(summary)
        else:
            self.logger.error("Failed to generate summary.")
        
        return summary

#################################################   WIP    #################################################

class PersonalizationAgent(Agent): # Personalization Agent
    def __init__(self, profile):
        super().__init__("Personalization Agent")
        self.profile = profile # List of keywords (How will we get this? Survey, reading history?, ideally with usage)
        self.persona = self.generate_persona()
    
    def generate_persona(self):
        # Create a persona based on profile keywords
        sys_msg = "You are an assistant that generates a user persona based on their background knowledge."
        usr_msg = (
            f"Create a user persona based on the following keywords: {', '.join(self.profile)}. "
            "Describe the user's familiarity with each topic, and indicate whether they are an expert, moderately informed, or a beginner."
        )
        persona_description = self.call_api(sys_msg, usr_msg)
        self.logger.info(f"Generated persona: {persona_description}")
        return persona_description
    
    def generate_feedback(self, summary):
        # Use the persona and summary to generate feedback about the summary
        sys_msg = (
            "You are a personalization assistant who provides feedback to tailor an article summary "
            "based on the user's background knowledge and familiarity level."
        )
        usr_msg = (
            f"Based on this persona: '{self.persona}', analyze the following summary: '{summary}'. "
            "Suggest whether the user would need more context, less context, or if the summary is appropriate as is. "
            "Provide specific feedback on which details to add or remove."
        )
        feedback = self.call_api(sys_msg, usr_msg)
        self.logger.info(f"Generated feedback: {feedback}")
        return feedback

    def process(self, summary):
        # Generate and return feedback based on the summary produced by Summarization Agent
        feedback = self.analyze_summary(summary)
        return feedback
    
class ArbiterAgent(Agent): # Arbiter Agent
    def __init__(self):
        super().__init__("Arbiter Agent")

    def process(self, article, summaries):
        raise NotImplementedError("To be completed.")
