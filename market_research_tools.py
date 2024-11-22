import requests
import re
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Set up device for using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def summarize_with_bart(text):
    """ Summarizes text using Bart-Large-CNN model. """
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=200,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def serper_search(query, api_key):
    """ Performs a web search using the Serper API. """
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("organic", [])
    else:
        print(f"Error: {response.status_code}, {response.json()}")
        return []

def identify_industry(target_company, api_key):
    """ Identifies the industry of a target company using web search data. """
    query = f"What industry does {target_company} operate in?"
    results = serper_search(query, api_key)
    snippets = [result.get("snippet", "") for result in results]
    combined_snippets = " ".join(snippets)
    return summarize_with_bart(combined_snippets) if snippets else "Industry information not found."

def identify_competitors_in_industry(target_industry, api_key):
    """ Identifies competitors within a specific industry. """
    query = f"List of companies in {target_industry} industry."
    results = serper_search(query, api_key)
    competitors = set()
    for result in results:
        snippet = result.get("snippet", "")
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', snippet)  # Extract capitalized words
        competitors.update(words)
    return list(competitors)

def extract_competitor_ai_strategies(competitors, api_key):
    """ Extracts AI strategies used by competitors. """
    strategies = []
    for competitor in competitors:
        query = f"How is {competitor} using AI?"
        results = serper_search(query, api_key)
        for result in results:
            snippet = result.get("snippet", "")
            strategies.append(snippet)

    combined_strategies = " ".join(strategies)
    return summarize_with_bart(combined_strategies) if strategies else "No AI strategies found."

def generate_recommendations_for_target(summary_text, target_company):
    """ Generates AI-related recommendations for a target company. """
    recommendations = []

    if "maintenance" in summary_text.lower():
        recommendations.append(f"{target_company} can implement AI-driven predictive maintenance tools.")
    if "autonomous" in summary_text.lower() or "self-driving" in summary_text.lower():
        recommendations.append(f"{target_company} can explore AI for advanced autonomous systems.")
    if "personalization" in summary_text.lower():
        recommendations.append(f"{target_company} can enhance customer experiences using AI personalization.")
    if "energy" in summary_text.lower():
        recommendations.append(f"{target_company} can focus on AI models for energy efficiency.")
    if "supply chain" in summary_text.lower():
        recommendations.append(f"{target_company} can optimize supply chains using AI-powered analytics.")

    if not recommendations:
        recommendations.append(f"{target_company} can invest in exploring AI/GenAI solutions for operational efficiency.")

    return recommendations
