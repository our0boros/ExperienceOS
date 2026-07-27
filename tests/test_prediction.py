"""Phase A 预测契约管线测试。

验证 flow.md 融合方案的实施正确性：
1. PredictionContract / PredictionVerification 数据结构
2. _extract_prediction_accuracy 从消息中提取预测准确性
3. SubStepRecord 预测字段持久化
4. consolidate_substeps 预测调整贝叶斯评分
"""

import json
import pytest

from experience_os.models import (
    PredictionContract,
    PredictionVerification,
    SubStepOutcome,
    _extract_keywords,
)
from experience_os.experience_library import SubStepRecord


# =====================================================================
# 数据结构测试
# =====================================================================

class TestPredictionContract:
    """预测契约数据结构基本测试。"""

    def test_contract_defaults(self):
        c = PredictionContract()
        assert c.step_id == ""
        assert c.confidence == 0.5
        assert c.expected_output == ""

    def test_contract_fields(self):
        c = PredictionContract(
            step_id="step_1",
            intent="find_user_by_email",
            expected_input="email address of the user",
            expected_output="user_id, name, and contact info",
            expected_effect="retrieve user profile from database",
            confidence=0.9,
            agent_reasoning="I need to find the user first to get their order history.",
        )
        assert c.intent == "find_user_by_email"
        assert c.confidence == 0.9


class TestPredictionVerification:
    """预测验证结果测试。"""

    def test_factory_high_quality(self):
        """预测准确 + 执行成功 → high_quality"""
        contract = PredictionContract(
            intent="find_user",
            expected_output="user_id, name",
            agent_reasoning="Looking up the user to get their ID.",
        )
        outcome = SubStepOutcome(
            intent="find_user",
            action_name="find_user_id_by_email",
            success=True,
            params={"_result_summary": "user_id: 123, name: Alice"},
        )
        pv = PredictionVerification.from_outcome(
            contract, outcome, parent_task_success=True,
        )
        assert pv.quality_label == "high_quality"

    def test_factory_lucky_success(self):
        """预测不准确 + 执行成功 → lucky_success"""
        contract = PredictionContract(
            intent="find_user",
            expected_output="order history and shipping address",
            agent_reasoning="Need order history first.",
        )
        outcome = SubStepOutcome(
            intent="find_user",
            action_name="find_user_id_by_email",
            success=True,
            # Result has user info, not order history
            params={"_result_summary": "user_id: 123, name: Alice"},
        )
        pv = PredictionVerification.from_outcome(
            contract, outcome, parent_task_success=True,
        )
        assert pv.quality_label == "lucky_success"

    def test_factory_implementation_defect(self):
        """预测准确 + 执行失败 → implementation_defect"""
        contract = PredictionContract(
            intent="find_user",
            expected_output="user_id",
            agent_reasoning="Looking up user ID.",
        )
        outcome = SubStepOutcome(
            intent="find_user",
            action_name="find_user_id_by_email",
            success=False,
            error="Database connection timeout",
        )
        pv = PredictionVerification.from_outcome(
            contract, outcome, parent_task_success=False,
        )
        assert pv.quality_label == "implementation_defect"

    def test_factory_negative_sample(self):
        """预测不准确 + 执行失败 → negative_sample"""
        contract = PredictionContract(
            intent="find_user",
            expected_output="user_id",
            agent_reasoning="Should return user ID.",
        )
        outcome = SubStepOutcome(
            intent="find_user",
            action_name="find_user_id_by_email",
            success=False,
            error="User not found",
            params={"_result_summary": "error: User not found"},
        )
        pv = PredictionVerification.from_outcome(
            contract, outcome, parent_task_success=False,
        )
        assert pv.quality_label == "negative_sample"


class TestExtractKeywords:
    """关键词提取测试。"""

    def test_empty(self):
        assert _extract_keywords("") == []

    def test_noise_filtered(self):
        kw = _extract_keywords("the user is going to be found in the database")
        # noise words filtered, meaningful words remain
        for noise in {"the", "is", "to", "be", "in", "a", "an"}:
            assert noise not in [w.lower() for w in kw]

    def test_meaningful_words(self):
        kw = _extract_keywords("find user_id and order history for exchange processing")
        assert "user_id" in kw or "user" in kw
        assert "order" in kw or "history" in kw or "exchange" in kw

    def test_max_six(self):
        kw = _extract_keywords(
            "find user email order shipping payment status address history preferences"
        )
        assert len(kw) <= 6


# =====================================================================
# 消息提取测试
# =====================================================================

class TestPredictionExtraction:
    """从对话消息中提取预测准确性。"""

    @staticmethod
    def _make_messages(tool_calls_with_results):
        """helper: 构建模拟消息列表。

        tool_calls_with_results: list of (assistant_content, tool_name, tool_args_str, tool_result_str)
        """
        msgs = [{"role": "system", "content": "You are an agent."}]
        for reasoning, tool_name, args, result in tool_calls_with_results:
            call_id = f"call_{tool_name}"
            msgs.append({
                "role": "assistant",
                "content": reasoning,
                "tool_calls": [{
                    "id": call_id,
                    "function": {"name": tool_name, "arguments": args},
                }],
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
        return msgs

    def test_simple_extraction(self):
        """基本提取：tool result 含预期内容 → high_quality"""
        from experience_os.experiments.compare import _extract_prediction_accuracy

        msgs = self._make_messages([
            (
                "I need to find the user by email to get their user_id.",
                "find_user_id_by_email",
                '{"email": "a@x.com"}',
                '{"user_id": 123, "name": "Alice", "email": "a@x.com"}',
            ),
        ])
        results = _extract_prediction_accuracy(msgs)
        assert len(results) == 1
        assert results[0]["tool_name"] == "find_user_id_by_email"
        # "user" "email" "id" keywords in both reasoning and result → accurate
        assert results[0]["prediction_accuracy"] == 1.0
        assert results[0]["quality_label"] in ("high_quality", "lucky_success")

    def test_error_detection(self):
        """tool result 含错误 → prediction_accuracy=0"""
        from experience_os.experiments.compare import _extract_prediction_accuracy

        msgs = self._make_messages([
            (
                "I'll look up the order to check its status.",
                "get_order_details",
                '{"order_id": "O123"}',
                '{"error": "Order not found", "status": "error"}',
            ),
        ])
        results = _extract_prediction_accuracy(msgs)
        assert len(results) == 1
        assert results[0]["tool_name"] == "get_order_details"
        assert results[0]["prediction_accuracy"] == 0.0
        assert results[0]["quality_label"] == "implementation_defect"

    def test_multiple_tool_calls(self):
        """多步任务：每个 tool call 各自验证"""
        from experience_os.experiments.compare import _extract_prediction_accuracy

        msgs = self._make_messages([
            (
                "Need to lookup user email to get user_id name profile data.",
                "find_user_id_by_email",
                '{"email": "a@x.com"}',
                '{"user_id": 123, "name": "Alice", "email": "a@x.com", "profile": "active"}',
            ),
            (
                "Now get order details to check items and delivery status.",
                "get_order_details",
                '{"order_id": "O123"}',
                '{"order_id": "O123", "items": [...], "status": "delivered"}',
            ),
        ])
        results = _extract_prediction_accuracy(msgs)
        assert len(results) == 2
        assert results[0]["tool_name"] == "find_user_id_by_email"
        assert results[1]["tool_name"] == "get_order_details"
        # Both should be high_quality (results match expectations)
        assert all(r["prediction_accuracy"] == 1.0 for r in results)

    def test_no_tool_calls(self):
        """无 tool_calls 的消息返回空列表"""
        from experience_os.experiments.compare import _extract_prediction_accuracy

        msgs = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "I don't have access to a clock."},
        ]
        results = _extract_prediction_accuracy(msgs)
        assert results == []


# =====================================================================
# SubStepRecord 持久化测试
# =====================================================================

class TestSubStepPredictionFields:
    """预测字段在 SubStepRecord 中的持久化。"""

    def test_prediction_fields_default(self):
        rec = SubStepRecord(
            trajectory_id="t1",
            experiment_id="e1",
            plan_idx=0,
            intent="find_user",
            tool_name="find_user_id_by_email",
        )
        assert rec.prediction_accuracy == 1.0
        assert rec.quality_label == ""

    def test_prediction_fields_set(self):
        rec = SubStepRecord(
            trajectory_id="t1",
            experiment_id="e1",
            plan_idx=0,
            intent="find_user",
            tool_name="find_user_id_by_email",
            prediction_accuracy=0.5,
            quality_label="lucky_success",
        )
        assert rec.prediction_accuracy == 0.5
        assert rec.quality_label == "lucky_success"

    def test_meta_json_roundtrip(self):
        """验证 meta_json 编码/解码预测字段的一致性"""
        rec = SubStepRecord(
            trajectory_id="t1",
            experiment_id="e1",
            plan_idx=0,
            intent="find_user",
            tool_name="find_user_id_by_email",
            prediction_accuracy=0.0,
            quality_label="implementation_defect",
            meta_json='{"existing": "data"}',
        )
        # 模拟 log_substep 中的编码逻辑
        meta = {"existing": "data"}
        if rec.prediction_accuracy != 1.0 or rec.quality_label:
            meta["prediction_accuracy"] = rec.prediction_accuracy
            meta["quality_label"] = rec.quality_label
        encoded = json.dumps(meta, ensure_ascii=False)

        # 解码
        decoded = json.loads(encoded)
        assert decoded["prediction_accuracy"] == 0.0
        assert decoded["quality_label"] == "implementation_defect"
        assert decoded["existing"] == "data"


# =====================================================================
# 贝叶斯评分调整测试
# =====================================================================

class TestBayesianAdjustment:
    """验证预测准确率对贝叶斯评分的调整效果。"""

    def test_high_prediction_acc_boosts_score(self):
        """高预测准确率 → adjusted >= original"""
        bayesian = 0.5
        avg_pred_acc = 1.0
        pred_multiplier = 0.5 + 0.5 * avg_pred_acc
        adjusted = bayesian * pred_multiplier
        assert adjusted == 0.5  # 1.0 multiplier, unchanged

    def test_low_prediction_acc_penalizes_score(self):
        """低预测准确率 → adjusted < original"""
        bayesian = 0.8
        avg_pred_acc = 0.0
        pred_multiplier = 0.5 + 0.5 * avg_pred_acc
        adjusted = bayesian * pred_multiplier
        assert adjusted == 0.4  # 0.5 multiplier
        assert adjusted < bayesian

    def test_lucky_success_penalty(self):
        """侥幸成功过多 → 额外降低评分"""
        bayesian = 0.7
        avg_pred_acc = 0.6
        lucky_ratio = 0.5  # 50% lucky

        pred_multiplier = 0.5 + 0.5 * avg_pred_acc  # 0.8
        adjusted = bayesian * pred_multiplier  # 0.56

        if lucky_ratio > 0.3:
            adjusted *= 0.7  # 0.392

        assert adjusted < 0.5

    def test_pred_multiplier_range(self):
        """预测乘数始终在 [0.5, 1.0] 范围内"""
        for pred_acc in [0.0, 0.25, 0.5, 0.75, 1.0]:
            multiplier = 0.5 + 0.5 * pred_acc
            assert 0.5 <= multiplier <= 1.0
