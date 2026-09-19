"""Verify the public-shaped local API directly, without a web server."""

from jev_coreml.api import LocalJev
from jev_coreml.semantic import load


def main() -> None:
    agent = LocalJev(load(compute_units="cpu_and_ne"))
    result = agent.system_one(
        {"ticket": "I was charged twice and need the duplicate refunded today."},
        {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this ticket.",
                "criteria": {
                    "billing": "Charges, refunds, invoices, subscription changes.",
                    "technical": "Bugs, failed integrations, anything not working.",
                    "sales": "Pricing, plan comparisons, plans they do not have.",
                    "other": "None of the above applies.",
                },
            },
            "threatens_churn": {
                "type": "noul",
                "instructions": "The customer threatens to cancel, refund, charge back, or leave.",
                "criteria": {
                    "true": "The customer threatens to cancel, request a refund, charge back, or leave.",
                    "false": "The customer does not threaten to cancel, request a refund, charge back, or leave.",
                },
            },
            "business_impact": {
                "type": "score",
                "instructions": "How much the customer's own business is being harmed right now.",
                "criteria": [
                    "No harm; a question or a preference.",
                    "Inconvenience; a workaround exists.",
                    "Active revenue or operational loss while this continues.",
                ],
            },
        },
    )
    assert result["answers"]["department"]["choice"] == "billing", result
    print(result)


if __name__ == "__main__":
    main()
