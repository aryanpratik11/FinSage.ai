"""Simple CLI for running the FinSage planner."""
from finsage.planner import Planner
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker", nargs="?", help="Ticker symbol to analyze (e.g., TSLA)")
    p.add_argument("--peers", nargs="*", help="Peer tickers to compare against")
    args = p.parse_args()

    if not args.ticker:
        print("Please provide a ticker to analyze, e.g. 'python cli.py TSLA'")
        return

    planner = Planner()
    resp = planner.run(f"Analyze {args.ticker}", tickers=[args.ticker], peers=args.peers)
    print("\n=== Final Response ===")
    print(f"Query: {resp.get('query')}")
    print(f"Ticker: {resp.get('ticker')}")
    print(f"Thesis: {resp.get('thesis')}")
    print(f"Fundamentals: {resp.get('fundamentals')}")
    print(f"Metrics: {resp.get('metrics')}")
    print(f"Prediction: {resp.get('prediction')}")
    print(f"News sentiment: {resp.get('news', {}).get('sentiment')}")
    print(f"Risk score: {resp.get('risk')}")


if __name__ == "__main__":
    main()
