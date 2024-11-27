# Report Generator

This library generates reports based on scientific literature queries using Scopus, Voyage AI, and OpenaiAI.

## Installation

You can install the Report Generator directly from GitHub:

```
pip install git+https://github.com/DataGomes/TopicsAI

## API Keys

Before using the library, you need to set up your API keys. You can do this in two ways:

1. Set environment variables:
   - PYBLIOMETRICS_API_KEY
   - VOYAGE_API_KEY
   - OPENAI_API_KEY

2. Use the `set_api_keys()` function:

```python
from report_generator import set_api_keys

set_api_keys(
    pybliometrics_key="your_pybliometrics_key",
    voyage_key="your_openai_key",
    openai_key="your_openai_key"
)
```

Ensure you have the necessary permissions and subscriptions for Scopus API, Voyage AI, and OpenAI AI.

## Usage

### As a library

```python
from report_generator import TopicAI, set_api_keys

# Set your API keys
set_api_keys(
    pybliometrics_key="your_pybliometrics_key",
    voyage_key="your voyage key",
    openai_key="your openai key"
)

# Create a TopicAI instance
generator = TopicAI()

# Generate a report
query = "artificial intelligence in genetics"
html_output, query, html_pdf = generator.run_report(query, output_dir="your output directory")

if html_pdf == "failure":
    print(f"Error: {html_output}")
    
else:
    print("Report generated successfully.")
    print("\nSummary of results:")
    print(html_output)
```


### As a command-line tool

After installation, you can use the `generate_report` command:

```
generate_report "cars manufacturing management artificial intelligence" --output /path/to/o```
