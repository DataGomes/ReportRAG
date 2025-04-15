"""
Keyword Expansion Tool

This script generates related search terms for a specified theme 
using OpenAI's API and saves them to a file.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

def create_search_terms(theme, client):
    """
    Generate search terms related to a specified theme using OpenAI API.
    
    Args:
        theme (str): The theme to generate search terms for
        client (OpenAI): Initialized OpenAI client
        
    Returns:
        list: List of search terms related to the theme
    """
    system_prompt = f"""
    You are a {theme} expert. Your task is to provide the best search terms related to {theme}.

    A high-quality list of search terms should be:

    1. Succinct and specific.
    2. Ensure all terms are relevant to {theme} but do not need to include the word {theme} itself.
    3. Make sure that the terms are ONLY related to {theme}.
    4. The terms SHOULD NOT be related to another topic that is not {theme}.
    """

    prompt = f"""
    Your task is to provide a list of search terms that will yield the most relevant articles on this topic.

    Instructions:

    1. Identify related terms that can be used to search for {theme}.
    2. Provide your result in a list format, example ['term 1', 'term 2', 'term 3'].
    3. Avoid terms that are extensions, derivatives, or direct variations of {theme}.
    4. Ensure all terms are relevant to {theme} but do not need to include the word {theme} itself.
    5. Give all the terms in singular form.
    6. Make sure that EACH ONE of the terms are ONLY related to {theme}.

    Provide terms related ONLY to {theme}

    Your response:
    """

    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ],
        model="gpt-4o",
        temperature=0,
    )
    
    response_content = response.choices[0].message.content
    terms = eval(response_content)
    return terms

def main():
    """Main function to run the keyword expansion tool."""
    
    # Load environment variables
    load_dotenv(dotenv_path='app_config.env')
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Get theme from user
    theme = input("Enter the theme for keyword expansion: ")
    
    # Get search terms from OpenAI
    print(f"Generating search terms for theme: {theme}...")
    terms = create_search_terms(theme, client)
    
    # Add the theme term to the list if needed
    if theme.lower() not in [t.lower() for t in terms]:
        terms.append(theme)
    
    # Print results
    print("\nGenerated search terms:")
    for i, term in enumerate(terms, 1):
        print(f"{i}. {term}")
    
    # Save results to file
    output_file = f"{theme}_keywords.txt"
    with open(output_file, 'w') as f:
        f.write(f"Keywords for theme: {theme}\n")
        f.write("="*30 + "\n")
        for term in terms:
            f.write(f"- {term}\n")
    
    print(f"\nSaved {len(terms)} keywords to {output_file}")

if __name__ == "__main__":
    main()
