"""Run the selected multilingual decision checkpoint with Laya's UX contract.

This validates the data contract and checkpoint before its Core ML export. The
Core ML runtime adapter will preserve this request/response shape.
"""

from __future__ import annotations

from pathlib import Path

import laya


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    agent = laya.load(str(ROOT / "models" / "laya-multilingual"), device="cpu")
    result = agent.predict(
        {"subject": "請求が二重になっています", "body": "3月の利用料を二回請求されました。重複分を返金してください。"},
        {
            "department": {
                "type": "choice",
                "instructions": "この問い合わせを担当すべき部門はどれですか？",
                "criteria": {
                    "billing": "請求、支払い、返金に関する問い合わせ",
                    "technical": "障害、バグ、技術的な不具合",
                    "sales": "新規契約、料金プラン、見積もり",
                },
            },
            "refund_requested": {
                "type": "noul",
                "instructions": "利用者は返金を求めていますか？",
            },
        },
    )
    print(result)


if __name__ == "__main__":
    main()
