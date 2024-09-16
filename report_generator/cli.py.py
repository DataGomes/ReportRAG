import argparse
from .report_generator import TopicAI
from .config import set_api_keys

def main():
    parser = argparse.ArgumentParser(description="Generate a report based on a query.")
    parser.add_argument("query", help="The query to generate a report for.")
    parser.add_argument("--output", help="Directory to save the output PDF.", default=".")
    args = parser.parse_args()

    # Set API keys (you might want to add arguments for these as well)
    set_api_keys()

    generator = TopicAI()
    html_output, query, html_pdf = generator.run_report(args.query, args.output)

    if html_pdf == "failure":
        print(f"Error: {html_output}")
    else:
        print("Report generated successfully.")
        print("\nSummary of results:")
        print(html_output)

if __name__ == '__main__':
    main()
