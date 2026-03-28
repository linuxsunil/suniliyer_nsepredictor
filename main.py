import functions_framework
import google.generativeai as genai
import requests
import json
from google.cloud import secretmanager

PROJECT_ID = "gen-lang-client-0773813468"

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    return client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")

def get_gemini_client():
    api_key = get_secret("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

def select_model(theory, period):
    return "gemini-2.5-flash"

def fetch_stock(ticker, range_val):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": range_val}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        raise Exception(f"Yahoo Finance returned HTTP {r.status_code}")
    data = r.json()
    if data.get("chart", {}).get("error"):
        raise Exception(data["chart"]["error"].get("description", "Invalid ticker"))
    return data

def run_gemini_analysis(stock_name, theory, period, signals, indicators):
    get_gemini_client()
    model_name = select_model(theory, period)

    theory_labels = {
        "all":     "all four theories combined (Dow Theory, Elliott Wave, Moving Averages, RSI & Momentum)",
        "dow":     "Dow Theory (trend identification via moving average crossovers)",
        "elliott": "Elliott Wave Theory (wave cycle positioning)",
        "sma":     "Moving Average analysis (SMA/EMA stack alignment)",
        "rsi":     "RSI & Momentum analysis (overbought/oversold conditions)",
    }
    theory_label = theory_labels.get(theory, theory)

    signals_text = "\n".join(
        f"- {s['name']}: {s['sig']} ({s['conf']}% confidence) — {s['reason']}"
        for s in signals
    )

    prompt = f"""You are a technical analyst for Indian equity markets. Based ONLY on the data below, give a sharp 3-sentence analysis. You MUST complete all 3 sentences fully before stopping.

Stock: {stock_name} | Theory: {theory_label}

Data: Price ₹{indicators['price']} | RSI {indicators['rsi']} | EMA20 ₹{indicators['ema20']} | SMA50 ₹{indicators['sma50']} | SMA200 ₹{indicators['sma200']} | 52W High ₹{indicators['high52']} | 52W Low ₹{indicators['low52']} | From High: {indicators['fromHigh']}

Signals: {signals_text}

Write exactly 3 complete sentences — do not stop mid-sentence:
Sentence 1: What the price and MAs show right now.
Sentence 2: What the signal means and key levels to watch.
Sentence 3: Final verdict — BUY/SELL/HOLD with a specific price range. End with "This analysis is for educational purposes only and does not constitute financial advice. Please consult a SEBI-registered advisor before investing."
"""

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
        ),
        safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        }
    )

    candidate = response.candidates[0] if response.candidates else None
    if not candidate or not candidate.content or not candidate.content.parts:
        finish = candidate.finish_reason if candidate else "unknown"
        return f"Gemini could not generate analysis (blocked, reason code: {finish}). Try a different theory or shorter period.", model_name
    return candidate.content.parts[0].text, model_name

def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

@functions_framework.http
def market_observer(request):
    if request.method == "OPTIONS":
        return ("", 204, cors_headers())

    path = request.path
    headers = {**cors_headers(), "Content-Type": "application/json"}

    if path == "/api/health":
        return (json.dumps({"ok": True, "service": "nse-gemini-analyser"}), 200, headers)

    if path == "/api/stock" and request.method == "GET":
        ticker = request.args.get("ticker")
        range_val = request.args.get("range", "1y")
        if not ticker:
            return (json.dumps({"error": "ticker required"}), 400, headers)
        try:
            data = fetch_stock(ticker, range_val)
            return (json.dumps(data), 200, headers)
        except Exception as e:
            return (json.dumps({"error": str(e)}), 502, headers)

    if path == "/api/analyse" and request.method == "POST":
        body = request.get_json(silent=True) or {}
        stock_name = body.get("stockName")
        theory     = body.get("theory", "all")
        period     = body.get("period", "1y")
        signals    = body.get("signals", [])
        indicators = body.get("indicators", {})

        if not stock_name or not signals or not indicators:
            return (json.dumps({"error": "Missing: stockName, signals, indicators"}), 400, headers)

        try:
            analysis, model_used = run_gemini_analysis(stock_name, theory, period, signals, indicators)
            return (json.dumps({"analysis": analysis, "model": model_used}), 200, headers)
        except Exception as e:
            return (json.dumps({"error": str(e)}), 502, headers)

    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html = f.read()
        return (html, 200, {"Content-Type": "text/html", **cors_headers()})
    except FileNotFoundError:
        return (json.dumps({"error": "Not found", "path": path}), 404, headers)
