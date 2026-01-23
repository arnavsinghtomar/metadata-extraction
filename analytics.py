import json
import logging
import pandas as pd
from openai import OpenAI
from ingest_excel import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_business_health(db_url, openai_key, sheet_info, lookback_periods=3):
    """
    Analyzes the business health of a specific sheet's data.
    Improved version with better column detection and robust trend analysis.
    """
    table_name = sheet_info["table_name"]
    columns_meta = sheet_info.get("columns_metadata", [])
    
    if not columns_meta:
        return {"error": "No column metadata available for this sheet."}

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_key
    )

    # 1. Identify Column Roles (Refined Prompt)
    columns_desc = [{
        "name": col["name"],
        "type": col["type"],
        "samples": col.get("samples", [])
    } for col in columns_meta]

    role_prompt = f"""
    Analyze these column schemas and classify them into core business roles:
    - revenue: Columns representing income, sales, billing, base_amount, etc.
    - cost: Columns representing expenses, spend, taxes, fees, overhead, etc.
    - time: Columns representing dates, months, years, or timestamps.
    - profit: Columns explicitly stating profit, margin, or net income (often calculated, so skip if unclear).

    Data Schema:
    {json.dumps(columns_desc, indent=2)}

    Rules:
    1. If multiple columns represent revenue/cost, list all of them.
    2. Be strict: only include columns that are clearly numeric values for revenue/cost.
    3. Return ONLY a valid JSON object.

    Output Format:
    {{
      "revenue": ["col1", "col2"],
      "cost": ["col3"],
      "time": ["col_date"],
      "profit": []
    }}
    """

    try:
        role_resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial data expert. Output valid JSON only."},
                {"role": "user", "content": role_prompt}
            ],
            temperature=0
        )
        roles = json.loads(role_resp.choices[0].message.content)
    except Exception as e:
        logging.error(f"LLM Role detection failed: {e}")
        return {"error": f"Failed to identify data roles: {e}"}

    revenue_cols = roles.get("revenue", [])
    cost_cols = roles.get("cost", [])
    time_cols = roles.get("time", [])
    profit_cols = roles.get("profit", [])

    if not revenue_cols and not cost_cols and not profit_cols:
        return {
            "status": "Insufficient Data",
            "reason": "This sheet doesn't seem to contain recognizable financial metrics (Revenue/Cost/Profit).",
            "suggested_action": "Try analyzing a different sheet or ensure columns are correctly named."
        }

    # 2. Fetch required data
    conn = get_db_connection(db_url)
    try:
        select_cols = list(set(revenue_cols + cost_cols + profit_cols + time_cols))
        # Ensure columns exist in DB (double check)
        select_sql = f"SELECT {', '.join(select_cols)} FROM {table_name}"
        df = pd.read_sql(select_sql, conn)
    except Exception as e:
        logging.error(f"Data fetch failed for {table_name}: {e}")
        return {"error": f"Failed to retrieve data for analysis: {e}"}
    finally:
        conn.close()

    if df.empty:
        return {"error": "The table exists but contains no data to analyze."}

    # 3. Aggregate Metrics & Trend Analysis
    df = df.copy()
    
    # helper for robust numeric conversion
    def to_numeric(series):
        # Convert to string and strip non-numeric except dot and minus
        clean_series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        return pd.to_numeric(clean_series, errors='coerce').fillna(0)

    for col in (revenue_cols + cost_cols + profit_cols):
        df[col] = to_numeric(df[col])

    df["total_revenue"] = df[revenue_cols].sum(axis=1) if revenue_cols else 0
    df["total_cost"] = df[cost_cols].sum(axis=1) if cost_cols else 0

    if profit_cols:
        df["total_profit"] = df[profit_cols].sum(axis=1)
    else:
        df["total_profit"] = df["total_revenue"] - df["total_cost"]

    # Overall Metrics
    total_rev = df["total_revenue"].sum()
    total_cst = df["total_cost"].sum()
    total_prf = df["total_profit"].sum()
    margin = round((total_prf / total_rev * 100), 2) if total_rev != 0 else 0

    # Trend Logic
    rev_trend = cst_trend = prf_trend = "stable"
    history_df = pd.DataFrame()

    if time_cols:
        try:
            # Try to convert first time column to datetime
            time_col = time_cols[0]
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col]).sort_values(time_col)
            
            # Group by time period (Monthly or Daily depending on count)
            if len(df) > 10:
                history_df = df.resample('ME', on=time_col).sum(numeric_only=True).tail(lookback_periods)
            else:
                history_df = df.tail(lookback_periods)

            def get_trend(series):
                if len(series) < 2: return "stable"
                # Use first and last values of the window
                v_start = series.iloc[0]
                v_end = series.iloc[-1]
                pct_change = (v_end - v_start) / (abs(v_start) + 1e-9)
                if pct_change > 0.05: return "up"
                if pct_change < -0.05: return "down"
                return "stable"

            rev_trend = get_trend(history_df["total_revenue"])
            cst_trend = get_trend(history_df["total_cost"])
            prf_trend = get_trend(history_df["total_profit"])
        except Exception as e:
            logging.warning(f"Trend analysis failed: {e}")

    # 4. Deterministic Health Status
    if total_prf < 0:
        status = "Risk"
        base_reason = "The business is currently operating at a net loss."
    elif cst_trend == "up" and rev_trend != "up":
        status = "Warning"
        base_reason = "Expenses are rising faster than revenue, threatening future margins."
    elif margin < 10:
        status = "Warning"
        base_reason = "Profit margins are low, indicating potential sensitivity to cost changes."
    else:
        status = "Healthy"
        base_reason = "The business shows positive profit margins and stable or improving trends."

    # 5. LLM Reason & Action (Refined)
    reason_prompt = f"""
    Explain the business health of this dataset.
    
    Current Metrics:
    - Total Revenue: {total_rev:,.2f} (Trend: {rev_trend})
    - Total Cost: {total_cst:,.2f} (Trend: {cst_trend})
    - Net Profit: {total_prf:,.2f} (Trend: {prf_trend})
    - Profit Margin: {margin}%
    - Calculated Status: {status}

    Provide a professional, 2-sentence summary of the business's current state and one specific actionable recommendation based on these trends.
    """

    try:
        reason_resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": reason_prompt}],
            temperature=0.3
        )
        executive_summary = reason_resp.choices[0].message.content.strip()
    except:
        executive_summary = f"{base_reason} Suggest further audit of financial statements."

    return {
        "status": status,
        "metrics": {
            "revenue": total_rev,
            "cost": total_cst,
            "profit": total_prf,
            "margin": margin
        },
        "trends": {
            "revenue": rev_trend,
            "cost": cst_trend,
            "profit": prf_trend
        },
        "summary": executive_summary,
        "history": history_df.reset_index().to_dict() if not history_df.empty else None
    }
