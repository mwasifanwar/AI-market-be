from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS
import time
import json
import numpy as np
import re
from textblob import TextBlob
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

PORT = int(os.getenv("PORT", 5000))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

GROQ_API_URL = "https://api.groq.com/openai/v1"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


def simulate_market_trends(depth):
    np.random.seed(42)
    
    base_growth = np.random.randint(5, 20) 
    
    variance = max(1, 10 - depth)  
    fluctuations = np.random.normal(0, variance, 6)

    growth_values = [max(base_growth + fluctuation, 0) for fluctuation in fluctuations]

    trend_analysis = {
        "base_growth": base_growth,
        "fluctuations": fluctuations.tolist(),
        "predicted_growth": growth_values
    }

    return trend_analysis

def classify_query(query):
    """
    Classifies the user query into predefined market analysis categories.
    Uses keyword matching and basic NLP techniques.
    """
    categories = {
        "Growth Trends": ["growth", "market expansion", "trend", "future"],
        "Investment Opportunities": ["investment", "funding", "finance", "profit"],
        "Competitor Analysis": ["competitor", "market share", "rival"],
        "Consumer Demand": ["consumer", "demand", "customer behavior"]
    }
    
    query_cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", query).lower()
    
    for category, keywords in categories.items():
        if any(keyword in query_cleaned for keyword in keywords):
            return category

    return "General Market Insights"


def competitor_sentiment_analysis(name):
    """
    Performs a sentiment analysis on competitor-related reviews.
    The sentiment score is averaged over multiple samples.
    """
    reviews = [
        f"{name} is leading the industry with innovation and a strong market presence.",
        f"{name} has a significant customer base but struggles with pricing competitiveness.",
        f"Users report that {name}'s customer service needs improvement.",
        f"Many analysts believe {name} is a key disruptor in the market."
    ]
    
    sentiments = [TextBlob(review).sentiment.polarity for review in reviews]
    avg_sentiment = sum(sentiments) / len(sentiments)

    sentiment_result = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"

    return {"sentiments": sentiments, "average_sentiment": avg_sentiment, "result": sentiment_result}


def predict_growth():
    past_growth = [5, 8, 12, 15, 18]
    df = pd.DataFrame({"year": range(1, 6), "growth": past_growth})

    model = ARIMA(df["growth"], order=(2, 1, 2))
    model_fit = model.fit()

    future_forecast = model_fit.forecast(steps=5).tolist()

    return {"past_growth": past_growth, "predicted_growth": future_forecast}



def perform_swot_analysis(idea):
    swot_data = {
        "Strengths": ["Innovative product", "Strong market demand", "Scalable business model", "Cost leadership"],
        "Weaknesses": ["High initial cost", "Limited brand recognition", "Competitive market", "Regulatory risks"],
        "Opportunities": ["Growing industry trends", "Technology advancements", "New funding options", "Market expansion"],
        "Threats": ["Market volatility", "Changing regulations", "High competition", "Supply chain disruptions"]
    }

    return {
        "Strengths": swot_data["Strengths"][:2],
        "Weaknesses": swot_data["Weaknesses"][:2],
        "Opportunities": swot_data["Opportunities"][:2],
        "Threats": swot_data["Threats"][:2]
    }

def generate_keywords(idea):
    corpus = [
        f"{idea} market trends and industry analysis",
        f"Latest {idea} innovations and startup growth",
        f"How to grow a {idea} business effectively",
        f"Understanding consumer demand in the {idea} niche"
    ]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    feature_array = np.array(vectorizer.get_feature_names_out())

    # Get highest scoring words based on TF-IDF
    sorted_indices = np.argsort(X.toarray().sum(axis=0))[-4:]
    top_keywords = feature_array[sorted_indices]

    return list(top_keywords)

def fetch_groq_api(request_body, max_attempts=3):
    attempt = 1
    while attempt <= max_attempts:
        try:
            response = requests.post(f"{GROQ_API_URL}/chat/completions", json=request_body, headers=HEADERS, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == -1:
                raise ValueError("❌ No valid JSON found in response.")

            json_response = json.loads(response_text[json_start:json_end])

            return jsonify(json_response)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt} failed: {str(e)}")
            attempt += 1
            time.sleep(2)

    return jsonify({"error": "Failed after multiple attempts"}), 500


@app.route("/analyze-market", methods=["POST"])
def analyze_market():
    data = request.json
    industry = data.get("industry")
    region = data.get("region")
    depth = data.get("depth")

    if not industry or not region or not depth:
        return jsonify({"error": "❌ Missing required parameters: industry, region, depth."}), 400

    request_body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a market analyst AI."},
            {"role": "user", "content": f"""Perform an in-depth, data-driven market analysis for the {industry} industry in {region} with a {depth} level of detail.
                - The data must be **latest and authentic**, backed by real market trends and research.
                - Provide only **genuine** insights; no fabricated or vague estimates.
                - Ensure numbers, statistics, and recommendations are **credible**.
                - Return **strictly** in this JSON format without any additional text:
                {{
                    "graphData": {{
                        "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                        "values": [X1, X2, X3, X4, X5, X6]
                    }},
                    "keyFindings": {{
                        "marketGrowth": "XX%",
                        "competitionLevel": "Low/Medium/High",
                        "entryBarriers": "Low/Medium/High"
                    }},
                    "recommendations": [
                        "Recommendation 1",
                        "Recommendation 2",
                        "Recommendation 3"
                    ]
                }}"""}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }

    return fetch_analysis(request_body)


def fetch_ai_response(request_body, attempt=1):
    try:
        print(f"⏳ Attempt {attempt} - Fetching AI Response...")
        response = requests.post(f"{GROQ_API_URL}/chat/completions", json=request_body, headers=HEADERS, timeout=60)
        response.raise_for_status()
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No relevant insights found.")

        return jsonify({"assistantResponse": ai_response})

    except Exception as error:
        print(f"⚠️ Attempt {attempt} failed:", error)
        if attempt < 3:
            return fetch_ai_response(request_body, attempt + 1)
        return jsonify({"error": "Failed to fetch assistant response after multiple attempts."}), 500


@app.route("/ai-market-assistant", methods=["POST"])
def ai_market_assistant():
    data = request.json
    user_query = data.get("userQuery")
    industry = data.get("industry")

    if not user_query or not industry:
        return jsonify({"error": "❌ Missing required parameters: userQuery, industry."}), 400

    request_body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a strict market analysis AI. Provide only direct, data-backed insights."},
            {"role": "user", "content": f"""Provide a **concise, latest, data-backed response** to the following market-related question.
                - **Only return the direct answer. Do not add intros, explanations, or filler text.**
                - **Ensure insights are based on the latest trends, market data, and statistics.**
                - **If the question is unrelated to {industry}, respond with:** "Please update the inputs on the previous page for new analysis."

                User's Question: "{user_query}" """}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }

    return fetch_ai_response(request_body)

@app.route("/analyze-competitor", methods=["POST"])
def analyze_competitor():
    data = request.json
    competitor_name = data.get("competitorName")

    if not competitor_name:
        return jsonify({"error": "❌ Missing required parameter: competitorName."}), 400

    request_body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an AI competitor analysis expert. Provide only **latest, fact-based, and structured** insights in JSON format."},
            {"role": "user", "content": f"""Perform a strict, **data-backed competitor analysis** for {competitor_name}.
                🔹 **STRICT REQUIREMENTS:**
                - **All insights must be the latest and 100% authentic.** No outdated, vague, or fabricated data.
                - **Use real, fact-based industry trends.** No assumptions or estimates.
                - **Return the response in the exact JSON format below.** No extra text or explanations.
                - **Each field should be only 2-3 words long, exactly matching the UI.**
                {{
                    "marketPosition": "Emerging Leader / Market Leader / Challenger / Niche Player",
                    "strategyAnalysis": {{
                        "pricing": "Premium pricing / Budget-friendly / Value-based",
                        "marketing": "Digital-first / Influencer-driven / Traditional media",
                        "product": "Rapid innovation / High customization / Mass production"
                    }},
                    "strengths": [
                        "Strong brand identity",
                        "Innovative technology",
                        "Customer service"
                    ],
                    "weaknesses": [
                        "Limited market reach",
                        "High operational costs",
                        "Product gaps"
                    ]
                }}"""}
        ],
        "temperature": 0.3,
        "max_tokens": 600
    }

    response = fetch_groq_api(request_body)
    
    if isinstance(response, dict):  
        response["marketPosition"] = " ".join(response.get("marketPosition", "").split()[:3])
        response["strategyAnalysis"]["pricing"] = " ".join(response["strategyAnalysis"]["pricing"].split()[:3])
        response["strategyAnalysis"]["marketing"] = " ".join(response["strategyAnalysis"]["marketing"].split()[:3])
        response["strategyAnalysis"]["product"] = " ".join(response["strategyAnalysis"]["product"].split()[:3])
        response["strengths"] = [" ".join(strength.split()[:3]) for strength in response["strengths"]]
        response["weaknesses"] = [" ".join(weakness.split()[:3]) for weakness in response["weaknesses"]]

    return response


@app.route("/predict-market", methods=["POST"])
def predict_market():
    data = request.json
    industry = data.get("industry")

    if not industry:
        return jsonify({"error": "❌ Missing required parameter: industry."}), 400

    request_body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an AI market prediction expert. Provide only **latest, fact-based, and structured** insights in JSON format."},
            {"role": "user", "content": f"""Provide a **highly accurate, data-backed 5-year market prediction** for the {industry} industry.
                🔹 **STRICT REQUIREMENTS:**
                - **All insights must be the latest and 100% authentic.** No outdated, vague, or fabricated data.
                - **Use real, fact-based industry trends.** No assumptions or estimates.
                - **Ensure the response strictly follows the JSON format below.** No extra text or explanations.

                {{
                    "growthScore": {{
                        "score": 85,
                        "trend": "Rising / Stable / Declining"
                    }},
                    "futureBusinessModels": [
                        "Short phrase 1",
                        "Short phrase 2",
                        "Short phrase 3"
                    ],
                    "growthProjection": {{
                        "years": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
                        "values": [X1, X2, X3, X4, X5]
                    }}
                }}"""}
        ],
        "temperature": 0.3,
        "max_tokens": 600
    }

    return fetch_groq_api(request_body)

@app.route("/generate-business-strategy", methods=["POST"])
def generate_business_strategy():
    data = request.json
    business_idea = data.get("businessIdea")

    if not business_idea:
        return jsonify({"error": "❌ Missing required parameter: businessIdea."}), 400

    request_body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an AI business strategy expert. Provide only structured, latest, and concise responses in JSON format."},
            {"role": "user", "content": f"""Generate a **highly structured** business strategy for "{business_idea}" following these **STRICT GUIDELINES**:
                - **All insights must be the latest and data-backed.** No vague or fabricated data.
                - **Use short, precise, and actionable phrases** as shown in the format below.
                - **Do NOT add unnecessary explanations or intro text.**
                - **Ensure formatting is strictly followed.**
                {{
                    "targetAudience": [
                        "Segment 1 (e.g., Tech-savvy professionals 25-45)",
                        "Segment 2 (e.g., Small business owners)",
                        "Segment 3 (e.g., Remote workers)"
                    ],
                    "productStrategy": [
                        "MVP launch within 3 months",
                        "Iterative development based on user feedback",
                        "Premium features for enterprise clients"
                    ],
                    "financialProjections": {{
                        "estimatedROI": "XXX% in first year",
                        "breakEvenPoint": "X months",
                        "initialInvestment": "$XX,XXX - $XX,XXX"
                    }},
                    "marketingPlan": [
                        "Content marketing focus on LinkedIn and Medium",
                        "Early adopter program with X% discount",
                        "Partnership with industry influencers"
                    ]
                }}"""}
        ],
        "temperature": 0.3,
        "max_tokens": 700
    }

    return fetch_groq_api(request_body)


@app.route("/generate-content-strategy", methods=["POST"])
def generate_content_strategy():
    data = request.json
    business_idea = data.get("businessIdea")

    if not business_idea:
        return jsonify({"error": "❌ Missing required parameter: businessIdea."}), 400

    request_body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a professional AI content strategist. Provide only **latest, concise, fact-based insights** in JSON format."},
            {"role": "user", "content": f"""Generate a **100% latest, fact-based AI content strategy** for the business niche: "{business_idea}". 
                🔹 **STRICT FORMAT REQUIREMENTS**:
                - **All insights MUST be the latest and authentic.** No outdated, vague, or fabricated data.
                - **Strictly follow the structure below. Do NOT add any intros or explanations.**
                - **Ensure descriptions are short and to the point.**
                {{
                    "targetKeywords": [
                        "Keyword 1",
                        "Keyword 2",
                        "Keyword 3",
                        "Keyword 4"
                    ],
                    "blogTopics": [
                        "Blog Title 1",
                        "Blog Title 2",
                        "Blog Title 3",
                        "Blog Title 4"
                    ],
                    "socialMediaStrategy": {{
                        "LinkedIn": {{
                            "description": "Short description",
                            "frequency": "X posts/week"
                        }},
                        "Twitter": {{
                            "description": "Short description",
                            "frequency": "Daily"
                        }},
                        "Medium": {{
                            "description": "Short description",
                            "frequency": "X posts/week"
                        }}
                    }},
                    "viralMarketingAngles": [
                        "Angle 1",
                        "Angle 2",
                        "Angle 3",
                        "Angle 4"
                    ],
                    "contentPlatforms": {{
                        "Company Blog": {{
                            "traffic": "Traffic Level",
                            "conversion": "X% conversion"
                        }},
                        "Medium Publication": {{
                            "traffic": "Traffic Level",
                            "conversion": "X% conversion"
                        }},
                        "LinkedIn Articles": {{
                            "traffic": "Traffic Level",
                            "conversion": "X% conversion"
                        }}
                    }}
                }}"""}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }

    return fetch_groq_api(request_body)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
