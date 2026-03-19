# test_conda_env.py
"""
Tests for conda environment switching logic.

Verifies that:
1. Default mode does NOT trigger conda relaunch
2. Explicit --use-conda-env triggers relaunch via subprocess.run
3. Already in target env means no relaunch
"""
import os
import subprocess
import sys

import pytest

from main_cli import is_in_conda_env, warn_if_env_mismatch, relaunch_in_conda


class TestIsInCondaEnv:
    """Tests for is_in_conda_env function."""

    def test_in_target_env(self, monkeypatch):
        """Should return True when CONDA_DEFAULT_ENV matches target."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "ashare_dpoint")
        assert is_in_conda_env("ashare_dpoint") is True

    def test_not_in_target_env(self, monkeypatch):
        """Should return False when CONDA_DEFAULT_ENV differs."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        assert is_in_conda_env("ashare_dpoint") is False

    def test_env_not_set(self, monkeypatch):
        """Should return False when CONDA_DEFAULT_ENV is not set."""
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
        assert is_in_conda_env("ashare_dpoint") is False


class TestWarnIfEnvMismatch:
    """Tests for warn_if_env_mismatch function."""

    def test_in_target_env_no_warning(self, monkeypatch, capsys):
        """Should not warn when already in target environment."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "ashare_dpoint")
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_CONDA", raising=False)

        warn_if_env_mismatch("ashare_dpoint")

        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_ci_env_no_warning(self, monkeypatch, capsys):
        """Should not warn in CI environment."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

        warn_if_env_mismatch("ashare_dpoint")

        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_skip_conda_env_no_warning(self, monkeypatch, capsys):
        """Should not warn when SKIP_CONDA=1."""
        monkeypatch.setenv("SKIP_CONDA", "1")
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

        warn_if_env_mismatch("ashare_dpoint")

        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_not_in_target_env_shows_warning(self, monkeypatch, capsys):
        """Should warn when not in target environment and not in CI."""
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_CONDA", raising=False)
        monkeypatch.delenv("_ASHARE_RELAUNCHED", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

        warn_if_env_mismatch("ashare_dpoint")

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "ashare_dpoint" in captured.out
        assert "--use-conda-env" in captured.out


class TestRelaunchInConda:
    """Tests for relaunch_in_conda function."""

    def test_already_relaunched_returns_false(self, monkeypatch):
        """Should return False if _ASHARE_RELAUNCHED=1 (prevent recursion)."""
        monkeypatch.setenv("_ASHARE_RELAUNCHED", "1")
        assert relaunch_in_conda("ashare_dpoint") is False

    def test_in_target_env_returns_false(self, monkeypatch):
        """Should return False if already in target environment."""
        monkeypatch.delenv("_ASHARE_RELAUNCHED", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "ashare_dpoint")
        assert relaunch_in_conda("ashare_dpoint") is False

    def test_conda_not_found_exits(self, monkeypatch, capsys):
        """Should print error and exit if conda not found when explicitly requested."""
        monkeypatch.delenv("_ASHARE_RELAUNCHED", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

        # Mock shutil.which to return None
        import shutil
        original_which = shutil.which

        def mock_which(name):
            if name == "conda":
                return None
            return original_which(name)

        monkeypatch.setattr(shutil, "which", mock_which)

        with pytest.raises(SystemExit) as exc_info:
            relaunch_in_conda("ashare_dpoint")

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "conda not found" in captured.out

    def test_relaunch_calls_subprocess_run(self, monkeypatch, capsys):
        """Should call subprocess.run when relaunch is needed and return True."""
        monkeypatch.delenv("_ASHARE_RELAUNCHED", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_CONDA", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

        called = {"value": False, "args": None, "kwargs": None}

        def mock_run(*args, **kwargs):
            called["value"] = True
            called["args"] = args
            called["kwargs"] = kwargs
            # Don't actually run, just record the call
            return None

        monkeypatch.setattr(subprocess, "run", mock_run)

        # Mock shutil.which to return a fake conda path
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/conda" if name == "conda" else None)

        # Should return True (relaunch attempted)
        result = relaunch_in_conda("ashare_dpoint")

        assert result is True
        assert called["value"] is True
        assert called["args"] is not None
        assert called["kwargs"] is not None
        
        # Verify the command includes "conda run" with "python" (not sys.executable)
        cmd_args = called["args"][0]
        assert cmd_args[0] == "conda"
        assert "run" in cmd_args
        assert "-n" in cmd_args
        assert "ashare_dpoint" in cmd_args
        # Verify it uses "python" not the full sys.executable path
        assert "python" in cmd_args
        
        # Verify _ASHARE_RELAUNCHED=1 is set in child environment (anti-recursion guard)
        assert "env" in called["kwargs"]
        assert called["kwargs"]["env"]["_ASHARE_RELAUNCHED"] == "1"
        
        # Verify info message was printed
        captured = capsys.readouterr()
        assert "Relaunching inside conda env" in captured.out


class TestDefaultModeDoesNotRelaunch:
    """
    Test that default mode (no --use-conda-env) does NOT trigger subprocess.run.
    
    This is the key test for the "no auto-relaunch by default" behavior.
    """

    def test_default_mode_no_subprocess_call(self, monkeypatch):
        """
        Verify warn_if_env_mismatch does NOT call subprocess.run.
        
        This ensures the default behavior is "warn only, no relaunch".
        """
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_CONDA", raising=False)
        monkeypatch.delenv("_ASHARE_RELAUNCHED", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")

        called = {"value": False}

        def fake_run(*args, **kwargs):
            called["value"] = True
            return None

        monkeypatch.setattr(subprocess, "run", fake_run)

        # Call warn_if_env_mismatch (default mode behavior)
        warn_if_env_mismatch("ashare_dpoint")

        # Should NOT have called subprocess.run
        assert called["value"] is False


class TestHandleCondaEnv:
    """
    Tests for _handle_conda_env function.
    
    This tests the integration of the conda handling logic.
    """

    def test_handle_conda_env_with_use_conda_env(self, monkeypatch):
        """
        When --use-conda-env is provided, should call relaunch_in_conda.
        """
        monkeypatch.delenv("_ASHARE_RELAUNCHED", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        
        # Mock relaunch_in_conda to track calls
        called = {"value": False}
        
        def mock_relaunch(env):
            called["value"] = True
            # Simulate successful relaunch
            raise SystemExit(0)
        
        from main_cli import _handle_conda_env
        monkeypatch.setattr("main_cli.relaunch_in_conda", mock_relaunch)
        
        # Create mock args
        class MockArgs:
            use_conda_env = "ashare_dpoint"
            target_conda_env = "ashare_dpoint"
        
        with pytest.raises(SystemExit) as exc_info:
            _handle_conda_env(MockArgs())
        
        assert called["value"] is True
        assert exc_info.value.code == 0

    def test_handle_conda_env_without_use_conda_env(self, monkeypatch, capsys):
        """
        When --use-conda-env is NOT provided, should only warn.
        """
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_CONDA", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        
        from main_cli import _handle_conda_env
        
        # Create mock args (no use_conda_env)
        class MockArgs:
            use_conda_env = None
            target_conda_env = "ashare_dpoint"
        
        # Should not raise, just warn
        _handle_conda_env(MockArgs())
        
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "ashare_dpoint" in captured.out
