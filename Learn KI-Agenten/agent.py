"""
Minimal Quant Trading Agent — pure Anthropic API, no frameworks.
Lernprojekt: zeigt wie ein Agent mit Tools funktioniert.

Tools:
  1. get_price_data     — historische Kursdaten via yfinance
  2. calculate_metrics  — SMA, RSI, Volatilität
  3. get_news_sentiment — aktuelle News + Claude-Sentiment

Setup:
  pip install anthropic yfinance requests python-dotenv
  export ANTHROPIC_API_KEY=your-key
"""

import json
import anthropic
import yfinance as yf
import datetime
from dotenv import load_dotenv # um api von .env zu laden
import os

load_dotenv()

# ─────────────────────────────────────────────
# 1. TOOL DEFINITIONEN (was der Agent nutzen darf)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_price_data",
        "description": (
            "Fetches historical OHLCV price data for a stock ticker. "
            "Returns closing prices, volume, and basic stats for the given period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT, 0700.HK"
                },
                "period": {
                    "type": "string",
                    "description": "Time period: 1mo, 3mo, 6mo, 1y, 2y",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y"]
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "calculate_metrics",
        "description": (
            "Calculates technical indicators for a given ticker: "
            "Simple Moving Averages (SMA 20/50), RSI (14), and annualised volatility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_news_sentiment",
        "description": (
            "Fetches recent news headlines for a ticker and returns "
            "a sentiment summary (bullish / neutral / bearish) with reasoning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "company_name": {
                    "type": "string",
                    "description": "Full company name for better news search, e.g. Apple Inc"
                }
            },
            "required": ["ticker", "company_name"]
        }
    }
]

# ─────────────────────────────────────────────
# 2. TOOL IMPLEMENTIERUNGEN
# ─────────────────────────────────────────────

def get_price_data(ticker: str, period: str) -> dict:
    """Lädt historische Kursdaten via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return {"error": f"No data found for {ticker}"}

        closes = hist["Close"].round(2).tolist()
        dates = [str(d.date()) for d in hist.index]
        current = closes[-1]
        start = closes[0]
        change_pct = round((current - start) / start * 100, 2)

        return {
            "ticker": ticker,
            "period": period,
            "current_price": current,
            "period_start_price": start,
            "period_change_pct": change_pct,
            "high": round(max(closes), 2),
            "low": round(min(closes), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "data_points": len(closes),
            "last_date": dates[-1],
            "recent_closes": closes[-5:]  # letzte 5 Tage
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_metrics(ticker: str) -> dict:
    """Berechnet technische Indikatoren."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist.empty or len(hist) < 50:
            return {"error": f"Not enough data for {ticker}"}

        closes = hist["Close"]

        # SMA
        sma20 = round(closes.rolling(20).mean().iloc[-1], 2)
        sma50 = round(closes.rolling(50).mean().iloc[-1], 2)
        current = round(closes.iloc[-1], 2)

        # RSI (14)
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = round(100 - (100 / (1 + rs.iloc[-1])), 1)

        # Annualisierte Volatilität
        daily_returns = closes.pct_change().dropna()
        volatility = round(daily_returns.std() * (252 ** 0.5) * 100, 2)

        # Trend-Signal (einfach)
        trend = "bullish" if current > sma20 > sma50 else \
                "bearish" if current < sma20 < sma50 else "mixed"

        return {
            "ticker": ticker,
            "current_price": current,
            "sma_20": sma20,
            "sma_50": sma50,
            "rsi_14": rsi,
            "rsi_signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
            "annualised_volatility_pct": volatility,
            "trend_signal": trend,
            "price_vs_sma20_pct": round((current - sma20) / sma20 * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}


def get_news_sentiment(ticker: str, company_name: str) -> dict:
    """
    Holt News-Headlines via yfinance und gibt eine strukturierte
    Sentiment-Zusammenfassung zurück.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return {"error": "No news found", "ticker": ticker}

        # Top 8 Headlines extrahieren
        headlines = []
        for item in news[:8]:
            title = item.get("content", {}).get("title", "")
            if title:
                headlines.append(title)

        if not headlines:
            return {"error": "Could not extract headlines", "ticker": ticker}

        # Sentiment via Claude analysieren (nested API call)
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        prompt = f"""Analysiere diese aktuellen News-Headlines zu {company_name} ({ticker}).

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

Antworte NUR mit einem JSON-Objekt (kein Markdown):
{{
  "sentiment": "bullish" | "neutral" | "bearish",
  "confidence": "high" | "medium" | "low",
  "key_themes": ["theme1", "theme2"],
  "reasoning": "kurze Begründung auf Englisch (max 2 Sätze)"
}}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # schnell + günstig für Subtasks
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        sentiment_data = json.loads(raw)

        return {
            "ticker": ticker,
            "company": company_name,
            "headlines_analysed": len(headlines),
            "headlines": headlines,
            **sentiment_data
        }

    except json.JSONDecodeError:
        return {"ticker": ticker, "error": "Could not parse sentiment JSON", "raw": raw}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ─────────────────────────────────────────────
# 3. TOOL DISPATCHER
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Ruft das richtige Tool auf und gibt das Ergebnis als String zurück."""
    print(f"\n  🔧 Tool: {tool_name}({tool_input})")

    if tool_name == "get_price_data":
        result = get_price_data(**tool_input)
    elif tool_name == "calculate_metrics":
        result = calculate_metrics(**tool_input)
    elif tool_name == "get_news_sentiment":
        result = get_news_sentiment(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    print(f"  ✅ Result: {json.dumps(result, indent=2)[:300]}...")
    return json.dumps(result)


# ─────────────────────────────────────────────
# 4. AGENTEN-SCHLEIFE (das Herzstück)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative trading analyst. Your job is to analyse stocks 
using the available tools and provide a clear, structured investment assessment.

For any stock analysis request:
1. Always fetch price data first
2. Calculate technical metrics
3. Check news sentiment
4. Synthesize everything into a final recommendation

Be concise and data-driven. Express uncertainty where appropriate.
Format your final answer clearly with sections: Price Overview, Technical Signals, 
News Sentiment, and Recommendation (BUY / HOLD / SELL with reasoning).
"""

def run_agent(user_query: str, max_iterations: int = 10) -> str:
    """
    Die Agenten-Schleife:
    - Schickt Query an Claude
    - Wenn Claude ein Tool aufruft → Tool ausführen → Ergebnis zurückschicken
    - Wiederholen bis Claude fertig ist (stop_reason == "end_turn")
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_query}]

    print(f"\n{'='*60}")
    print(f"🤖 Agent gestartet: {user_query}")
    print(f"{'='*60}")

    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration + 1}]")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        print(f"  Stop reason: {response.stop_reason}")

        # Fertig — keine weiteren Tool-Aufrufe
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text

        # Tool-Aufrufe verarbeiten
        if response.stop_reason == "tool_use":
            # Assistant-Nachricht mit Tool-Aufruf zur History hinzufügen
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Alle Tool-Aufrufe in dieser Runde ausführen
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Tool-Ergebnisse zurück an Claude
            messages.append({
                "role": "user",
                "content": tool_results
            })

    return "Max iterations reached — agent stopped."


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Beispiel-Queries — einfach anpassen
    queries = [
        #"Analyse Apple (AAPL) for me. Should I buy, hold, or sell?",
        # "Analysiere Tesla (TSLA) über die letzten 3 Monate.",
        "Give me a quick technical overview of NVIDIA (NVDA).",
    ]

    for query in queries:
        result = run_agent(query)
        print(f"\n{'='*60}")
        print("📊 FINAL ANALYSIS:")
        print(f"{'='*60}")
        print(result)
        print()
