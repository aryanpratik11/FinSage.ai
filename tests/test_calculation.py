from finsage.agents import CalculationAgent


def test_pe_calculation():
    agent = CalculationAgent()
    context = {"data": {"price": 220.0}, "eps": 4.0}
    out = agent.run("calc", context)
    assert out["pe_calculated"] == 220.0 / 4.0
