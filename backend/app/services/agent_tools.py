"""Tools available to the fraud investigation agent.

Design constraint: the agent gets a fixed, whitelisted set of read-only
functions returning structured payloads. It has no database handle, no shell, no
filesystem access and no ability to mutate a decision. Everything it can see is
assembled here first, so the evidence pack is auditable and bounded.

Every tool returns a ``ToolResult`` with an explicit ``available`` flag. When a
tool has nothing to report it says so, which is what lets the agent write
"Insufficient evidence" truthfully instead of inventing history.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fraudshield.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Structured output from one tool invocation."""

    tool: str
    available: bool
    data: dict[str, Any] = field(default_factory=dict)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "available": self.available,
            "data": self.data,
            "note": self.note,
        }


class TransactionHistoryStore:
    """In-memory store of prior transactions, keyed by customer.

    A deliberate stand-in for the account-history service a real deployment would
    call. It is injected rather than imported so tests and the API can supply
    their own data, and so the agent can never reach past it.
    """

    def __init__(self, history: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._history = history or {}

    def add(self, customer_id: str, transaction: dict[str, Any]) -> None:
        self._history.setdefault(customer_id, []).append(transaction)

    def get(self, customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return list(self._history.get(customer_id, []))[-limit:]

    def __len__(self) -> int:
        return sum(len(v) for v in self._history.values())


class InvestigationToolkit:
    """The complete, whitelisted tool surface exposed to the agent."""

    def __init__(
        self,
        transaction: dict[str, Any],
        scoring: dict[str, Any],
        risk: dict[str, Any],
        risk_policy: list[dict[str, Any]],
        model_info: dict[str, Any],
        history_store: TransactionHistoryStore | None = None,
    ) -> None:
        self._transaction = transaction
        self._scoring = scoring
        self._risk = risk
        self._risk_policy = risk_policy
        self._model_info = model_info
        self._history = history_store or TransactionHistoryStore()
        self.call_log: list[str] = []

    # ------------------------------------------------------------------ tools
    def get_transaction_details(self) -> ToolResult:
        """Return the transaction under investigation, excluding raw PCA noise."""
        self.call_log.append("get_transaction_details")
        summary = {
            k: v
            for k, v in self._transaction.items()
            if k in ("transaction_id", "customer_id", "Amount", "Time")
        }
        pca_present = sum(1 for k in self._transaction if k.startswith("V"))
        summary["pca_components_provided"] = pca_present
        return ToolResult("get_transaction_details", True, summary)

    def get_customer_transaction_history(self, limit: int = 10) -> ToolResult:
        """Return recent transactions for this customer, if any are known."""
        self.call_log.append("get_customer_transaction_history")
        customer_id = self._transaction.get("customer_id")
        if not customer_id:
            return ToolResult(
                "get_customer_transaction_history",
                False,
                {},
                note="No customer_id supplied; account history cannot be retrieved.",
            )
        rows = self._history.get(str(customer_id), limit=limit)
        if not rows:
            return ToolResult(
                "get_customer_transaction_history",
                False,
                {"customer_id": customer_id},
                note="No prior transactions on record for this customer.",
            )
        amounts = [float(r.get("Amount", 0.0)) for r in rows]
        return ToolResult(
            "get_customer_transaction_history",
            True,
            {
                "customer_id": customer_id,
                "n_transactions": len(rows),
                "mean_amount": round(sum(amounts) / len(amounts), 2),
                "max_amount": round(max(amounts), 2),
                "transactions": rows,
            },
        )

    def get_model_prediction(self) -> ToolResult:
        """Return the ML model's scoring output. This is authoritative."""
        self.call_log.append("get_model_prediction")
        return ToolResult("get_model_prediction", True, dict(self._scoring))

    def get_shap_explanation(self) -> ToolResult:
        """Return computed SHAP attributions for this prediction."""
        self.call_log.append("get_shap_explanation")
        contributions = self._scoring.get("explanation", []) or []
        if not contributions:
            return ToolResult(
                "get_shap_explanation",
                False,
                {},
                note="SHAP attributions were not computed for this prediction.",
            )
        return ToolResult("get_shap_explanation", True, {"top_features": contributions})

    def get_risk_policy(self) -> ToolResult:
        """Return the deterministic risk bands and their mandated actions."""
        self.call_log.append("get_risk_policy")
        return ToolResult(
            "get_risk_policy",
            True,
            {"assessment": dict(self._risk), "policy": list(self._risk_policy)},
        )

    def get_model_info(self) -> ToolResult:
        """Return provenance of the deployed model."""
        self.call_log.append("get_model_info")
        return ToolResult("get_model_info", True, dict(self._model_info))

    # ------------------------------------------------------------- assembly
    @property
    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        """The whitelist. Nothing outside this mapping is reachable by the agent."""
        return {
            "get_transaction_details": self.get_transaction_details,
            "get_customer_transaction_history": self.get_customer_transaction_history,
            "get_model_prediction": self.get_model_prediction,
            "get_shap_explanation": self.get_shap_explanation,
            "get_risk_policy": self.get_risk_policy,
            "get_model_info": self.get_model_info,
        }

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        """Call a whitelisted tool by name, rejecting anything unrecognised."""
        tool = self.registry.get(name)
        if tool is None:
            logger.warning("Blocked call to non-whitelisted tool '%s'.", name)
            return ToolResult(name, False, {}, note="Tool not permitted.")
        return tool(**kwargs)

    def collect_evidence(self) -> dict[str, Any]:
        """Run every read-only tool once and return the assembled evidence pack."""
        results = [self.invoke(name) for name in self.registry]
        return {
            "evidence": [result.to_dict() for result in results],
            "tools_called": list(self.call_log),
            "unavailable": [r.tool for r in results if not r.available],
        }
