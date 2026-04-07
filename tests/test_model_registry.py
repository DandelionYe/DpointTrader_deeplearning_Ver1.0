"""
测试模型注册表
"""
import pytest

from models import is_torch_model_type, TORCH_MODEL_TYPES


class TestModelRegistry:
    """测试模型注册表"""
    
    def test_mlp_is_torch_model_type(self):
        """测试MLP是torch模型"""
        assert is_torch_model_type("mlp") is True
    
    def test_lstm_is_torch_model_type(self):
        """测试LSTM是torch模型"""
        assert is_torch_model_type("lstm") is True
    
    def test_gru_is_torch_model_type(self):
        """测试GRU是torch模型"""
        assert is_torch_model_type("gru") is True
    
    def test_cnn_is_torch_model_type(self):
        """测试CNN是torch模型"""
        assert is_torch_model_type("cnn") is True
    
    def test_transformer_is_torch_model_type(self):
        """测试Transformer是torch模型"""
        assert is_torch_model_type("transformer") is True
    
    def test_xgb_not_torch_model_type(self):
        """测试XGBoost不是torch模型"""
        assert is_torch_model_type("xgb") is False
    
    def test_torch_model_types_contains_all(self):
        """测试TORCH_MODEL_TYPES包含所有模型"""
        expected = {"mlp", "lstm", "gru", "cnn", "transformer"}
        assert TORCH_MODEL_TYPES == expected
