# Essay-writer
We are building a multi-agent system that is capable of turning your prompt into an Essay. It uses a team of agents to accompolish this

- __A Plan Agent__ that generates a plan for the given user input. This happens just once for the user prompt. 
- __A Research Plan Agent__ Based on the plan, the agent does some research and retrieves some documents. It uses a web-search tool for this purpose. 
- __A Writer Agent__ It follows the plan using the researched documents to write the essay.  
- __A Critique Agent__ looks at the essay and generates a critique.
- __Research Critique Agent__ Uses the critique to do further research and suggest new documents for the writer. The new documents are appended to the existing set of documents and critique and sent back to the Writer agent. 

Each agent has its own role and they will communicate using the AutoGen's built-in caht-based coordination system. 

Further, we will use a local LLM with Ollama to run the system.

# Setting up the project

## Create and activate a virtual environment

## Install required dependencies
With your virtual environment activated, install autogen and any additional dependencies. 

``` python
pip install autogen-agentchat
pip install autogen-ext[openai]
pip install python-dotenv

```
## Create a .env file
You will need to create an .env file to store your API keys and any other environment configuration parameters that you desire to use within the application
``` python
touch .env
``` 
Then open the file and add your API keys
```
OPENAI_API_KEY=your-openai-api-key
```
## Load environment variavles into python
In your `main.py` file, use the following code to load the environment variables defined in your `.env ` file
```
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
```



