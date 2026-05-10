import csv
import datetime
import yfinance as yf
from scipy import stats


def get_filing_return(ticker: str, filing_date: str, days: int = 30) -> float | None:
    start = datetime.date.fromisoformat(filing_date)
    end = start + datetime.timedelta(days=days + 5)  # buffer for weekends/holidays

    # Use Ticker.history() — returns a plain DataFrame without MultiIndex columns
    hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())
    if hist.empty or len(hist) < 2:
        return None

    price_open = float(hist["Close"].iloc[0])

    target = start + datetime.timedelta(days=days)
    available_dates = [d.date() for d in hist.index]
    closest_idx = min(range(len(available_dates)), key=lambda i: abs((available_dates[i] - target).days))
    price_close = float(hist["Close"].iloc[closest_idx])

    return round((price_close / price_open) - 1, 6)


def analyze_stock_correlation(
    ticker: str,
    year_pairs: list[tuple[int, int]],
    changes_by_pair: dict[tuple[int, int], list[dict]],
    filing_dates_by_year: dict[int, str],
) -> dict:
    records: list[dict] = []

    for year_a, year_b in year_pairs:
        filing_date = filing_dates_by_year.get(year_b)
        if filing_date is None:
            print(f"  [{ticker} {year_a}→{year_b}] Filing date unavailable (loaded from cache), skipping.")
            continue

        changes = changes_by_pair.get((year_a, year_b), [])
        escalated = sum(1 for c in changes if c["status"] == "ESCALATED")
        de_escalated = sum(1 for c in changes if c["status"] == "DE-ESCALATED")
        net_risk_score = escalated - de_escalated

        print(f"  [{ticker} {year_a}→{year_b}] Fetching stock return (filing={filing_date})...", flush=True)
        return_30d = get_filing_return(ticker, filing_date)

        records.append({
            "ticker": ticker,
            "year_a": year_a,
            "year_b": year_b,
            "net_risk_score": net_risk_score,
            "escalated": escalated,
            "de_escalated": de_escalated,
            "filing_date": filing_date,
            "return_30d": return_30d,
        })

    valid = [r for r in records if r["return_30d"] is not None]
    correlation_stats: dict = {}

    if len(valid) >= 2:
        scores = [r["net_risk_score"] for r in valid]
        returns = [r["return_30d"] for r in valid]
        pearson_r, pearson_p = stats.pearsonr(scores, returns)
        spearman_r, spearman_p = stats.spearmanr(scores, returns)
        correlation_stats = {
            "n": len(valid),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_rho": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
        }
    else:
        correlation_stats = {"n": len(valid), "note": "insufficient data for correlation"}

    if records:
        with open("correlation_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return {"records": records, "correlation": correlation_stats}
