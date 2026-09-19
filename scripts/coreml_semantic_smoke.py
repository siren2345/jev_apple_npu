"""Run the Jev-shaped Core ML API on Japanese support-routing examples."""

from jev_coreml.semantic import Question, load


def main() -> None:
    agent = load(compute_units="cpu_and_ne")
    result = agent.predict(
        "クレジットカードで二重に請求されました。返金してほしいです。",
        [
            Question("department", "choice", "この問い合わせをどの担当へ送りますか？", ("請求・返金", "技術サポート", "営業")),
            Question("refund_requested", "noul", "利用者は返金を希望していますか？"),
        ],
    )
    print(result)


if __name__ == "__main__":
    main()
