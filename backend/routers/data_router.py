from fastapi import APIRouter
from backend.services import data_fetcher

router = APIRouter()

@router.get("/stock/{symbol}")
async def get_stock_info(symbol: str):
    """
    Fetch information about a specific stock.
    Example: /data/stock/RELIANCE
    """
    data = data_fetcher.get_stock_data(symbol)
    return {"symbol": symbol, "data": data}


@router.get("/mutualfund/{name}")
async def get_mutualfund_info(name: str):
    """
    Fetch information about a specific mutual fund.
    Example: /data/mutualfund/HDFC Equity Fund
    """
    data = data_fetcher.get_mutualfund_data(name)
    return {"mutual_fund": name, "data": data}
