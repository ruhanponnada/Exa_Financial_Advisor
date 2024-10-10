# Stock Analyzer
## Example project using the Exa Python SDK

1. Using Exa's Auto Search to pick the best search setting for each query (keyword or neural)
2. Using search_and_contents() through Exa's Python SDK
3. Summarizing webpage contents using an LLM

In this example, we will build an LLM-based stock analyzer with the Exa API to help us have all the information and news about a certain investment decision. We will use Exa to retrive recent news articles, financial reports, historical trends, and public sentiment which will then be fed into GPT-4o for summarizattion. 


To play with this code, we just need a Exa API key and an OpenAI API key.
If you would like to se the full code for this tutorial as a Colab notebook, [click here](https://colab.research.google.com/drive/1ew6HnGNkB9WtJuwvaZSHkVtCn0r6bueu?usp=sharing). 

## Setup
Let's import the Exa and OpenAI SDKs and set up our API keys to create client objects for each. In this example we stored our keys in the secrets pane of Colab.
```
from google.colab import userdata

EXA_API_KEY = userdata.get('EXA_API_KEY')
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
import openai
from exa_py import Exa

openai.api_key = OPENAI_API_KEY
exa = Exa(EXA_API_KEY)
```

## Creating Utility Functions
We'll be making several requests to the OpenAI API for completions from GPT-3.5 Turbo. To simplify this process, let’s create a utility function that takes the system and user messages as input, sends them to the API, and returns the LLM's response as a string.

```
def get_llm_response(system='You are a helpful assistant.', user='', temperature=1, model='gpt-3.5-turbo'):
    completion = openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]
    )
    return completion.choices[0].message.content
```

### Exa Search Utility Function
We will use Exa's auto search feature to decide to use either keyword or neural search. 
Now we will create a helper function to generate multiple search queries for our topic so that we can have more context and accuracy over our responses!

```
def generate_search_queries(stock, n):
    user_prompt = f"""I'm trying to figure out if this {stock} is a good investment or not and need help coming up with diverse search queries.
Please generate a list of {n} search queries that would be useful for generating an investment assessment on {topic} based on past data, market reports, financial statements and sentiment. These queries can be in various formats, from simple keywords to more complex phrases. Do not add any formatting or numbering to the queries."""

    completion = get_llm_response(
        system='The user will ask you to help generate some search queries. Respond with only the suggested queries in plain text with no extra formatting, each on its own line.',
        user=user_prompt,
        temperature=1
    )
    return [s.strip() for s in completion.split('\n') if s.strip()][:n]
```

Next, let's write another function that actually calls the Exa API to perform searches using Auto Search.
```
def get_search_results(queries, links_per_query=2):
    results = []
    for query in queries:
        search_response = exa.search_and_contents(query, 
            num_results=links_per_query, 
            use_autoprompt=False
        )
        results.extend(search_response.results)
    return results
```

## Getting a Investment Thesis Back
The final step is to instruct the LLM to summarize the content into a clear investment decision by including both the content and the URLs in the prompt.

```
def summarize_findings(stock, search_contents, content_slice=750):
    input_data = '\n'.join([f"--START ITEM--\nURL: {item.url}\nCONTENT: {item.text[:content_slice]}\n--END ITEM--\n" for item in search_contents])
    return get_llm_response(
        system='You are a helpful financial stock advisor. Give a recommendation on an investing strategy.',
        user=f'Input Data:\n{input_data}Give me advice about this {stock} based on the provided information. Include as many sources from financial data, historial data, expert opinions and more.',
    )
```

## Putting Everything Together
Now, lets put everything into one analyzer function that lets us input the stock we want to learn about, generate multiple search queries about it, feed those into Exa's Auto Search and then generate a decision!

```
def analyzer(stock):
    print(f'Starting research on stock: "{stock}"')

    search_queries = generate_search_queries(stock, 3)
    print("Generated search queries:", search_queries)

    search_results = get_search_results(search_queries)
    print(f"Found {len(search_results)} search results. Here's the first one:", search_results[0])

    print("Getting you a recommendation...")
    report = summarize_findings(stock, search_results)

    return report
```

Now we can run some examples as seen in the [colab notebook!](https://colab.research.google.com/drive/1ew6HnGNkB9WtJuwvaZSHkVtCn0r6bueu?usp=sharing).


We're done! We built our own personal financial advisor that translates a question about a stock into a solid investment thesis using Exa's Auto Search + Open AI. No web scraping or crawling needed! We've created a system that can quickly gather and synthesize information on any given stock.




