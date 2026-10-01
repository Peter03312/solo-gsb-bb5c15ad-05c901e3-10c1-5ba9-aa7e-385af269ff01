"""批次分装：父批次拆分为可独立投料的子批次。

覆盖：主流程（余量扣减/单位继承/不产生生产单/子批次可投料可再分装）、
拒绝规则（父不存在/隔离/已撤销、超量、编号复用、请求内重复、非正数）、
谱系遍历（分装边与投料边一起遍历）、疑似传播与复核重算（沿两类边）、
撤销交互（产出分装后视为曾被使用）、幂等与并发。
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from app import service
from app.db import session_scope
from app.errors import AppError
from app.models import Batch
from app.schemas import CancelIn, ProductionIn, RawBatchIn, SplitIn, SuspectIn


def _raw(client, batch_no, qty, unit="kg", key=None, note=None):
    return client.post(
        "/raw-batches",
        json={"batch_no": batch_no, "unit": unit, "quantity": str(qty), "note": note},
        headers={"Idempotency-Key": key or f"key-{batch_no}"},
    )


def _produce(client, out_no, inputs, out_qty=None, unit="kg", key=None):
    total = sum((Decimal(str(q)) for _, q in inputs), Decimal(0))
    return client.post(
        "/productions",
        json={
            "output_batch_no": out_no,
            "unit": unit,
            "output_quantity": str(out_qty if out_qty is not None else total),
            "inputs": [{"batch_no": b, "quantity": str(q)} for b, q in inputs],
        },
        headers={"Idempotency-Key": key or f"key-{out_no}"},
    )


def _split(client, parent_no, splits, key):
    return client.post(
        f"/batches/{parent_no}/splits",
        json={"splits": [{"batch_no": b, "quantity": str(q)} for b, q in splits]},
        headers={"Idempotency-Key": key},
    )


def _cancel(client, output_no, reason, key):
    return client.post(
        f"/productions/{output_no}/cancellation",
        json={"cancel_reason": reason},
        headers={"Idempotency-Key": key},
    )


def _flag(client, batch_no, key, note=None):
    return client.post(
        "/contamination",
        json={"batch_no": batch_no, "note": note},
        headers={"Idempotency-Key": key},
    )


def _review(client, batch_no, note, key):
    return client.post(
        "/reviews",
        json={"batch_no": batch_no, "review_note": note},
        headers={"Idempotency-Key": key},
    )


# --------------------------------------------------------------- 分装主流程


def test_split_creates_children_and_deducts_parent(client):
    _raw(client, "RM-A", "100")
    r = _split(client, "RM-A", [("RM-A-1", "30"), ("RM-A-2", "20")], key="s1")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["parent_batch_no"] == "RM-A"
    assert body["unit"] == "kg"
    assert body["parent_remaining_qty"] == 50.0
    assert body["total_split_qty"] == 50.0
    assert body["children"] == [
        {"batch_no": "RM-A-1", "unit": "kg", "quantity": 30.0},
        {"batch_no": "RM-A-2", "unit": "kg", "quantity": 20.0},
    ]

    # 父批次新余量与子批次余量均可查询
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 50.0
    c1 = client.get("/batches/RM-A-1").json()
    assert c1["initial_qty"] == 30.0 and c1["remaining_qty"] == 30.0
    assert c1["unit"] == "kg"  # 子批次继承父批次单位
    assert c1["is_quarantined"] is False and c1["is_cancelled"] is False


def test_split_children_are_consumable_and_splittable(client):
    _raw(client, "RM-A", "100")
    _split(client, "RM-A", [("RM-A-1", "40")], key="s1")
    # 子批次作为正常生产来源投料
    r = _produce(client, "FG-1", [("RM-A-1", "15")], out_qty=14, key="p1")
    assert r.status_code == 201, r.text
    assert client.get("/batches/RM-A-1").json()["remaining_qty"] == 25.0
    # 子批次可再次分装
    r = _split(client, "RM-A-1", [("RM-A-1a", "10")], key="s2")
    assert r.status_code == 201, r.text
    assert client.get("/batches/RM-A-1").json()["remaining_qty"] == 15.0
    assert client.get("/batches/RM-A-1a").json()["remaining_qty"] == 10.0


def test_split_output_batch_parent(client):
    # 生产产出同样可以分装
    _raw(client, "RM-A", "100")
    _produce(client, "MIX-1", [("RM-A", "60")], key="p1")
    r = _split(client, "MIX-1", [("MIX-1-A", "25"), ("MIX-1-B", "25")], key="s1")
    assert r.status_code == 201, r.text
    assert client.get("/batches/MIX-1").json()["remaining_qty"] == 10.0


def test_split_creates_no_production_record(client):
    _raw(client, "RM-A", "100")
    _split(client, "RM-A", [("RM-A-1", "30")], key="s1")
    # 分装不产生生产单：子批次没有生产原单据
    r = client.get("/productions/RM-A-1")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PRODUCTION_NOT_FOUND"


# --------------------------------------------------------------- 拒绝规则


def test_split_unknown_parent_not_found(client):
    r = _split(client, "GHOST", [("C-1", "10")], key="s1")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SOURCE_NOT_FOUND"
    assert client.get("/batches/C-1").status_code == 404


def test_split_quarantined_parent_rejected(client):
    _raw(client, "RM-A", "100")
    _flag(client, "RM-A", "g1")
    r = _split(client, "RM-A", [("C-1", "10")], key="s1")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "BATCH_QUARANTINED"
    # 整笔拒绝：余量不变、子批次未创建
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 100.0
    assert client.get("/batches/C-1").status_code == 404


def test_split_cancelled_output_parent_rejected(client):
    _raw(client, "RM-A", "100")
    _produce(client, "MIX-1", [("RM-A", "60")], key="p1")
    assert _cancel(client, "MIX-1", "录入错误", "c1").status_code == 200
    r = _split(client, "MIX-1", [("C-1", "10")], key="s1")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "BATCH_CANCELLED"
    assert client.get("/batches/C-1").status_code == 404


def test_split_total_exceeds_balance_rejected_without_change(client):
    _raw(client, "RM-A", "50")
    # 单笔不超、合计超：整笔拒绝
    r = _split(client, "RM-A", [("C-1", "30"), ("C-2", "30")], key="s1")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "INSUFFICIENT_BALANCE"
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 50.0
    assert client.get("/batches/C-1").status_code == 404
    assert client.get("/batches/C-2").status_code == 404


def test_split_child_no_reused_rejected(client):
    _raw(client, "RM-A", "100")
    _raw(client, "RM-B", "10")
    # 与既有原料批次冲突
    r = _split(client, "RM-A", [("RM-B", "10")], key="s1")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "BATCH_NO_REUSED"
    # 与父批次自身冲突
    r = _split(client, "RM-A", [("RM-A", "10")], key="s2")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "BATCH_NO_REUSED"
    # 整笔拒绝：同请求中其它合法子批次也不得创建
    r = _split(client, "RM-A", [("C-OK", "10"), ("RM-B", "5")], key="s3")
    assert r.status_code == 409
    assert client.get("/batches/C-OK").status_code == 404
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 100.0


def test_split_duplicate_child_no_in_request_rejected(client):
    _raw(client, "RM-A", "100")
    r = _split(client, "RM-A", [("C-1", "10"), ("C-1", "5")], key="s1")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "DUPLICATE_SPLIT_CHILD"
    assert client.get("/batches/C-1").status_code == 404
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 100.0


def test_split_non_positive_and_empty_rejected(client):
    _raw(client, "RM-A", "100")
    r = _split(client, "RM-A", [("C-1", "0")], key="s1")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = _split(client, "RM-A", [("C-1", "-3")], key="s2")
    assert r.status_code == 422
    r = _split(client, "RM-A", [], key="s3")
    assert r.status_code == 422
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 100.0


def test_split_requires_idempotency_key(client):
    _raw(client, "RM-A", "100")
    r = client.post(
        "/batches/RM-A/splits", json={"splits": [{"batch_no": "C-1", "quantity": 10}]}
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "IDEMPOTENCY_KEY_REQUIRED"
    assert client.get("/batches/C-1").status_code == 404


# --------------------------------------------------------------- 谱系遍历


def test_lineage_traverses_split_and_production_edges(client):
    _raw(client, "RM-A", "100")
    _raw(client, "RM-B", "100")
    _split(client, "RM-A", [("PK-1", "40")], key="s1")
    _produce(client, "FG-1", [("PK-1", "20"), ("RM-B", "10")], key="p1")

    # 下游：RM-A -> PK-1（分装）-> FG-1（投料）
    lin = client.get("/batches/RM-A/lineage").json()
    down = {n["batch_no"]: n["depth"] for n in lin["downstream"]}
    assert down == {"PK-1": 1, "FG-1": 2}
    edges = {(e["from"], e["to"]) for e in lin["edges"]}
    assert ("RM-A", "PK-1") in edges  # 分装边
    assert ("PK-1", "FG-1") in edges  # 投料边

    # 上游：FG-1 <- PK-1 <- RM-A，FG-1 <- RM-B
    up = client.get("/batches/FG-1/lineage").json()
    up_depths = {n["batch_no"]: n["depth"] for n in up["upstream"]}
    assert up_depths == {"PK-1": 1, "RM-B": 1, "RM-A": 2}
    up_edges = {(e["from"], e["to"]) for e in up["edges"]}
    assert up_edges == {("RM-A", "PK-1"), ("PK-1", "FG-1"), ("RM-B", "FG-1")}


def test_contamination_propagates_along_split_edges(client):
    _raw(client, "RM-A", "100")
    _split(client, "RM-A", [("PK-1", "40"), ("PK-2", "10")], key="s1")
    _produce(client, "FG-1", [("PK-1", "20")], key="p1")

    r = _flag(client, "RM-A", "g1")
    assert r.status_code == 200
    # 分装子批次及其生产后代都被隔离
    assert set(r.json()["affected_batch_nos"]) == {"PK-1", "PK-2", "FG-1"}
    for no in ("RM-A", "PK-1", "PK-2", "FG-1"):
        assert client.get(f"/batches/{no}").json()["is_quarantined"] is True
    # 隔离的子批次不得投料也不得再分装
    assert _produce(client, "FG-X", [("PK-1", "5")], key="px").status_code == 409
    assert _split(client, "PK-1", [("C-X", "5")], key="sx").status_code == 409


def test_contamination_from_split_child_spreads_to_its_descendants(client):
    _raw(client, "RM-A", "100")
    _split(client, "RM-A", [("PK-1", "40")], key="s1")
    _produce(client, "FG-1", [("PK-1", "20")], key="p1")

    r = _flag(client, "PK-1", "g1")
    assert r.json()["affected_batch_nos"] == ["FG-1"]
    # 父批次 RM-A 是上游，不被隔离
    assert client.get("/batches/RM-A").json()["is_quarantined"] is False
    assert client.get("/batches/FG-1").json()["is_quarantined"] is True


def test_review_recalculates_along_split_edges(client):
    _raw(client, "RM-A", "100")
    _split(client, "RM-A", [("PK-1", "40")], key="s1")
    _produce(client, "FG-1", [("PK-1", "20")], key="p1")
    _flag(client, "RM-A", "g1")
    assert client.get("/batches/FG-1").json()["is_quarantined"] is True

    r = _review(client, "RM-A", "留样复测合格，确认误报", "r1")
    assert r.status_code == 200
    assert set(r.json()["released_batch_nos"]) == {"RM-A", "PK-1", "FG-1"}
    for no in ("RM-A", "PK-1", "FG-1"):
        assert client.get(f"/batches/{no}").json()["is_quarantined"] is False


def test_review_mixed_keeps_quarantine_while_split_source_suspect(client):
    # PK-1（RM-A 的分装子批次）与 RM-B 混料 -> FG-1；
    # RM-A 与 RM-B 均疑似时，撤销 RM-A 不得放行 FG-1（仍疑似来源 RM-B）。
    _raw(client, "RM-A", "100")
    _raw(client, "RM-B", "100")
    _split(client, "RM-A", [("PK-1", "40")], key="s1")
    _produce(client, "FG-1", [("PK-1", "20"), ("RM-B", "10")], key="p1")
    _flag(client, "RM-A", "g1")
    _flag(client, "RM-B", "g2")

    r = _review(client, "RM-A", "RM-A 误报", "r1")
    assert r.status_code == 200
    assert r.json()["released_batch_nos"] == ["PK-1", "RM-A"]
    assert r.json()["still_quarantined"] == ["FG-1"]
    assert client.get("/batches/FG-1").json()["is_quarantined"] is True

    r = _review(client, "RM-B", "RM-B 误报", "r2")
    assert r.status_code == 200
    assert set(r.json()["released_batch_nos"]) == {"RM-B", "FG-1"}
    assert client.get("/batches/FG-1").json()["is_quarantined"] is False


# --------------------------------------------------------------- 与撤销的交互


def test_split_output_parent_counts_as_used_and_cannot_be_cancelled(client):
    _raw(client, "RM-A", "100")
    _produce(client, "MIX-1", [("RM-A", "60")], key="p1")
    _split(client, "MIX-1", [("MIX-1-A", "25")], key="s1")

    # 子批次尚未投料，父产出仍视为“曾被使用”，撤销入口拒绝
    r = _cancel(client, "MIX-1", "录入错误", "c1")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "OUTPUT_ALREADY_CONSUMED"
    assert err["details"]["consumed_by"] == ["MIX-1-A"]
    # 不改库存、未撤销
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 40.0
    assert client.get("/batches/MIX-1").json()["is_cancelled"] is False


def test_split_raw_batch_parent_unaffected_by_cancellation_rules(client):
    # 原料批次分装后依然不是生产产出，撤销入口维持 PRODUCTION_NOT_FOUND
    _raw(client, "RM-A", "100")
    _split(client, "RM-A", [("PK-1", "40")], key="s1")
    r = _cancel(client, "RM-A", "误把原料当产出", "c1")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PRODUCTION_NOT_FOUND"


# --------------------------------------------------------------- 幂等


def test_split_idempotent_replay_returns_first_result(client):
    _raw(client, "RM-A", "100")
    payload = {"splits": [{"batch_no": "C-1", "quantity": "30"}]}
    r1 = client.post(
        "/batches/RM-A/splits", json=payload, headers={"Idempotency-Key": "K1"}
    )
    assert r1.status_code == 201
    r2 = client.post(
        "/batches/RM-A/splits", json=payload, headers={"Idempotency-Key": "K1"}
    )
    assert r2.status_code == 201
    assert r2.json() == r1.json()
    # 重放不重复扣减、不重复建批
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 70.0
    assert client.get("/batches/C-1").json()["remaining_qty"] == 30.0


def test_split_same_key_different_content_conflicts(client):
    _raw(client, "RM-A", "100")
    p1 = {"splits": [{"batch_no": "C-1", "quantity": "10"}]}
    p2 = {"splits": [{"batch_no": "C-2", "quantity": "10"}]}
    assert client.post(
        "/batches/RM-A/splits", json=p1, headers={"Idempotency-Key": "K"}
    ).status_code == 201
    r = client.post("/batches/RM-A/splits", json=p2, headers={"Idempotency-Key": "K"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert client.get("/batches/C-2").status_code == 404
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 90.0


def test_failed_split_not_stored_same_key_retryable(client):
    _raw(client, "RM-A", "30")
    _produce(client, "MIX-1", [("RM-A", "25")], key="p1")  # 余量被占用到 5
    payload = {"splits": [{"batch_no": "C-1", "quantity": "20"}]}
    # 首次因余量不足失败，失败请求不固化幂等记录
    r1 = client.post(
        "/batches/RM-A/splits", json=payload, headers={"Idempotency-Key": "K1"}
    )
    assert r1.status_code == 409
    assert r1.json()["error"]["code"] == "INSUFFICIENT_BALANCE"
    # 撤销占用后余量回到 30，同键同体重试正常执行
    assert _cancel(client, "MIX-1", "录入错误", "c1").status_code == 200
    r2 = client.post(
        "/batches/RM-A/splits", json=payload, headers={"Idempotency-Key": "K1"}
    )
    assert r2.status_code == 201, r2.text
    assert client.get("/batches/RM-A").json()["remaining_qty"] == 10.0
    assert client.get("/batches/C-1").json()["remaining_qty"] == 20.0


# --------------------------------------------------------------- 并发


def test_concurrent_split_and_production_never_overuse(engine):
    # RM-1 余量 100：一线程分装 60，另一线程生产投料 60，合计 120 超额：
    # 恰好一个成功、一个 INSUFFICIENT_BALANCE，余量恒不为负
    with session_scope() as s:
        service.register_raw_batch(s, RawBatchIn(batch_no="RM-1", unit="kg", quantity="100"))

    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def split_worker() -> None:
        barrier.wait()
        try:
            with session_scope() as s:
                service.split_batch(
                    s,
                    "RM-1",
                    SplitIn(
                        splits=[{"batch_no": "C-1", "quantity": Decimal("60")}]  # type: ignore[list-item]
                    ),
                )
            results["split"] = "ok"
        except AppError as exc:
            results["split"] = exc.code
        except Exception as exc:
            results["split"] = f"ERROR:{type(exc).__name__}:{exc}"

    def produce_worker() -> None:
        barrier.wait()
        try:
            with session_scope() as s:
                service.submit_production(
                    s,
                    ProductionIn(
                        output_batch_no="FG-1",
                        unit="kg",
                        output_quantity=Decimal("60"),
                        inputs=[{"batch_no": "RM-1", "quantity": Decimal("60")}],  # type: ignore[list-item]
                    ),
                )
            results["produce"] = "ok"
        except AppError as exc:
            results["produce"] = exc.code
        except Exception as exc:
            results["produce"] = f"ERROR:{type(exc).__name__}:{exc}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda f: f(), [split_worker, produce_worker]))

    assert not any(str(v).startswith("ERROR") for v in results.values()), results
    assert sorted(results.values()) == ["INSUFFICIENT_BALANCE", "ok"], results
    with session_scope() as s:
        assert s.get(Batch, "RM-1").remaining_qty == Decimal("40.0000")


def test_split_concurrent_with_flag_equivalent_to_serial(engine):
    """分装 RM-1 -> C-1 与标记 RM-1 疑似并发：

    结局 A：分装先 -> C-1 创建成功，标记传播时沿分装边把 C-1 一并隔离（不漏隔离）；
    结局 B：标记先 -> RM-1 已隔离，分装被 BATCH_QUARANTINED 拒绝，C-1 不存在。
    """
    with session_scope() as s:
        service.register_raw_batch(s, RawBatchIn(batch_no="RM-1", unit="kg", quantity="100"))

    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def split_worker() -> None:
        barrier.wait()
        try:
            with session_scope() as s:
                service.split_batch(
                    s,
                    "RM-1",
                    SplitIn(
                        splits=[{"batch_no": "C-1", "quantity": Decimal("40")}]  # type: ignore[list-item]
                    ),
                )
            results["split"] = "ok"
        except AppError as exc:
            results["split"] = exc.code
        except Exception as exc:
            results["split"] = f"ERROR:{type(exc).__name__}:{exc}"

    def flag_worker() -> None:
        barrier.wait()
        try:
            with session_scope() as s:
                service.mark_suspect(s, SuspectIn(batch_no="RM-1"))
            results["flag"] = "ok"
        except Exception as exc:
            results["flag"] = f"ERROR:{type(exc).__name__}:{exc}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda f: f(), [split_worker, flag_worker]))

    assert results["flag"] == "ok", results
    assert not any(str(v).startswith("ERROR") for v in results.values()), results
    assert results["split"] in ("ok", "BATCH_QUARANTINED"), results

    with session_scope() as s:
        rm1 = s.get(Batch, "RM-1")
        assert rm1.is_suspect is True and rm1.is_quarantined is True
        c1 = s.get(Batch, "C-1")
        if results["split"] == "ok":
            # 分装先于标记：C-1 存在，且必须被疑似传播隔离
            assert c1 is not None and c1.is_quarantined is True
            assert rm1.remaining_qty == Decimal("60.0000")
        else:
            assert c1 is None
            assert rm1.remaining_qty == Decimal("100.0000")


def test_split_concurrent_with_cancel_equivalent_to_serial(engine):
    """分装产出 MIX-1 与撤销 MIX-1 并发：

    结局 A：分装先 -> MIX-1 已被分装使用，撤销被 OUTPUT_ALREADY_CONSUMED 拒绝；
    结局 B：撤销先 -> MIX-1 作废，分装被 BATCH_CANCELLED 拒绝。
    绝不允许“分装成功且撤销也成功”。
    """
    with session_scope() as s:
        service.register_raw_batch(s, RawBatchIn(batch_no="RM-1", unit="kg", quantity="100"))
        service.submit_production(
            s,
            ProductionIn(
                output_batch_no="MIX-1",
                unit="kg",
                output_quantity=Decimal("100"),
                inputs=[{"batch_no": "RM-1", "quantity": Decimal("100")}],  # type: ignore[list-item]
            ),
        )

    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def split_worker() -> None:
        barrier.wait()
        try:
            with session_scope() as s:
                service.split_batch(
                    s,
                    "MIX-1",
                    SplitIn(
                        splits=[{"batch_no": "C-1", "quantity": Decimal("40")}]  # type: ignore[list-item]
                    ),
                )
            results["split"] = "ok"
        except AppError as exc:
            results["split"] = exc.code
        except Exception as exc:
            results["split"] = f"ERROR:{type(exc).__name__}:{exc}"

    def cancel_worker() -> None:
        barrier.wait()
        try:
            with session_scope() as s:
                service.cancel_production(s, "MIX-1", CancelIn(cancel_reason="录入错误"))
            results["cancel"] = "ok"
        except AppError as exc:
            results["cancel"] = exc.code
        except Exception as exc:
            results["cancel"] = f"ERROR:{type(exc).__name__}:{exc}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda f: f(), [split_worker, cancel_worker]))

    assert not any(str(v).startswith("ERROR") for v in results.values()), results
    with session_scope() as s:
        mix1 = s.get(Batch, "MIX-1")
        c1 = s.get(Batch, "C-1")
        prod = service.get_production_record(s, "MIX-1")
        if results["split"] == "ok":
            # 分装先：撤销被拒
            assert results["cancel"] == "OUTPUT_ALREADY_CONSUMED", results
            assert c1 is not None and prod.is_cancelled is False
            assert mix1.remaining_qty == Decimal("60.0000")
        else:
            # 撤销先：分装被拒
            assert results["split"] == "BATCH_CANCELLED", results
            assert results["cancel"] == "ok", results
            assert c1 is None and prod.is_cancelled is True
            assert mix1.remaining_qty == Decimal("100.0000")
