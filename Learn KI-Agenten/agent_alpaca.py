"""
Quant Trading Agent — Anthropic API + Alpaca Paper Trading
Tools:
  1. get_price_data      — historische Kursdaten via yfinance
  2. calculate_metrics   — SMA, RSI, Volatilität
  3. get_news_sentiment  — News + Claude-Sentiment
  4. get_account_status  — Alpaca Portfolio-Übersicht
  5. place_order         — Alpaca Paper Trade ausführen

Setup:
  pip install anthropic yfinance python-dotenv alpaca-trade-api
  .env: ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
"""

import json
import os
import anthropic
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# ALPACA CLIENT
# ─────────────────────────────────────────────

def get_alpaca_client():
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(
        key_id=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        base_url="https://paper-api.alpaca.markets"
    )

# ─────────────────────────────────────────────
# 1. TOOL DEFINITIONEN
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
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT"
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
            "SMA 20/50, RSI (14), and annualised volatility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_news_sentiment",
        "description": (
            "Fetches recent news headlines for a ticker and returns "
            "a sentiment summary (bullish / neutral / bearish)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "company_name": {"type": "string", "description": "Full company name, e.g. Apple Inc"}
            },
            "required": ["ticker", "company_name"]
        }
    },
    {
        "name": "get_account_status",
        "description": (
            "Returns current Alpaca paper trading account status: "
            "cash, portfolio value, open positions, and recent orders."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "place_order",
        "description": (
            "Places a paper trade order on Alpaca. "
            "Use ONLY after analysis confirms a clear signal. "
            "Always confirm the order details before placing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                },
                "side": {
                    "type": "string",
                    "description": "buy or sell",
                    "enum": ["buy", "sell"]
                },
                "qty": {
                    "type": "number",
                    "description": "Number of shares to buy or sell"
                },
                "order_type": {
                    "type": "string",
                    "description": "market or limit",
                    "enum": ["market", "limit"]
                },
                "limit_price": {
                    "type": "number",
                    "description": "Limit price (only required for limit orders)"
                }
            },
            "required": ["ticker", "side", "qty", "order_type"]
        }
    }
]

# ─────────────────────────────────────────────
# 2. TOOL IMPLEMENTIERUNGEN
# ─────────────────────────────────────────────

def get_price_data(ticker: str, period: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        closes = hist["Close"].round(2).tolist()
        dates = [str(d.date()) for d in hist.index]
        current = closes[-1]
        start = closes[0]
        return {
            "ticker": ticker,
            "period": period,
            "current_price": current,
            "period_change_pct": round((current - start) / start * 100, 2),
            "high": round(max(closes), 2),
            "low": round(min(closes), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "last_date": dates[-1],
            "recent_closes": closes[-5:]
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_metrics(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty or len(hist) < 50:
            return {"error": f"Not enough data for {ticker}"}
        closes = hist["Close"]
        sma20 = round(closes.rolling(20).mean().iloc[-1], 2)
        sma50 = round(closes.rolling(50).mean().iloc[-1], 2)
        current = round(closes.iloc[-1], 2)
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = round(100 - (100 / (1 + rs.iloc[-1])), 1)
        volatility = round(closes.pct_change().dropna().std() * (252 ** 0.5) * 100, 2)
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
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return {"error": "No news found", "ticker": ticker}
        headlines = []
        for item in news[:8]:
            title = item.get("content", {}).get("title", "")
            if title:
                headlines.append(title)
        if not headlines:
            return {"error": "Could not extract headlines", "ticker": ticker}
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        prompt = f"""Analyse these news headlines for {company_name} ({ticker}).
Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

Reply ONLY with JSON (no markdown):
{{
  "sentiment": "bullish" | "neutral" | "bearish",
  "confidence": "high" | "medium" | "low",
  "key_themes": ["theme1", "theme2"],
  "reasoning": "max 2 sentences"
}}"""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        return {"ticker": ticker, "company": company_name,
                "headlines_analysed": len(headlines), **json.loads(raw)}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_account_status() -> dict:
    try:
        api = get_alpaca_client()
        account = api.get_account()
        positions = api.list_positions()
        orders = api.list_orders(status="open")

        positions_list = [{
            "ticker": p.symbol,
            "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_pl_pct": round(float(p.unrealized_plpc) * 100, 2)
        } for p in positions]

        return {
            "status": account.status,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "day_pl": float(account.equity) - float(account.last_equity),
            "open_positions": len(positions_list),
            "positions": positions_list,
            "open_orders": len(orders)
        }
    except Exception as e:
        return {"error": str(e)}


def place_order(ticker: str, side: str, qty: float,
                order_type: str, limit_price: float = None) -> dict:
    try:
        api = get_alpaca_client()
        kwargs = {
            "symbol": ticker,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": "day"
        }
        if order_type == "limit" and limit_price:
            kwargs["limit_price"] = limit_price

        order = api.submit_order(**kwargs)
        return {
            "status": "submitted",
            "order_id": order.id,
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "submitted_at": str(order.submitted_at)
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# 3. TOOL DISPATCHER
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    print(f"\n  🔧 Tool: {tool_name}({tool_input})")
    if tool_name == "get_price_data":
        result = get_price_data(**tool_input)
    elif tool_name == "calculate_metrics":
        result = calculate_metrics(**tool_input)
    elif tool_name == "get_news_sentiment":
        result = get_news_sentiment(**tool_input)
    elif tool_name == "get_account_status":
        result = get_account_status()
    elif tool_name == "place_order":
        result = place_order(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(f"  ✅ Result: {json.dumps(result, indent=2)[:300]}...")
    return json.dumps(result)


# ─────────────────────────────────────────────
# 4. AGENTEN-SCHLEIFE
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative trading analyst connected to an Alpaca paper trading account.

You have 5 tools:
- get_price_data: historical prices
- calculate_metrics: SMA, RSI, volatility
- get_news_sentiment: news analysis
- get_account_status: current portfolio and cash
- place_order: execute paper trades

Workflow for trade requests:
1. Check account status (cash, existing positions)
2. Analyse the stock (price, metrics, sentiment)
3. Decide: BUY / SELL / HOLD with clear reasoning
4. If BUY/SELL: calculate sensible position size (max 10% of portfolio per trade)
5. Place the order and confirm

Always be data-driven. Never place orders without analysis first.
Format answers clearly with sections.
"""

def run_agent(user_query: str, max_iterations: int = 15) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = [{"role": "user", "content": user_query}]

    total_input_tokens = 0
    total_output_tokens = 0

    print(f"\n{'='*60}")
    print(f"🤖 Agent: {user_query}")
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

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        print(f"  Stop reason: {response.stop_reason}")
        print(f"  Tokens — in: {response.usage.input_tokens}, out: {response.usage.output_tokens}")

        if response.stop_reason == "end_turn":
            print(f"\n{'─'*60}")
            print(f"📈 Token usage total — in: {total_input_tokens}, out: {total_output_tokens}, "
                  f"sum: {total_input_tokens + total_output_tokens}")
            print(f"{'─'*60}")
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

    print(f"\n{'─'*60}")
    print(f"📈 Token usage total — in: {total_input_tokens}, out: {total_output_tokens}, "
          f"sum: {total_input_tokens + total_output_tokens}")
    print(f"{'─'*60}")
    return "Max iterations reached."


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        # Portfolio-Übersicht
        "Show me my current paper trading portfolio.",

        # Analyse + Trade
        # "Analyse AAPL and buy 5 shares if the signal is bullish.",

        # Nur Analyse
        # "Should I buy or sell NVDA right now?",
    ]

    for query in queries:
        result = run_agent(query)
        print(f"\n{'='*60}")
        print("📊 RESULT:")
        print(f"{'='*60}")
        print(result)
