"""Launcher tests run with stdlib unittest without importing the application."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("launcher", Path(__file__).resolve().parents[1] / "scripts/start.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(launcher, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_existing_env_is_preserved(self):
        env = self.root / ".env"
        env.write_text("OPENAI_API_KEY=existing-local-value\n")
        with patch.object(launcher.subprocess, "call", return_value=0), patch.object(launcher, "run") as run:
            launcher.configure("python")
        self.assertEqual(env.read_text(), "OPENAI_API_KEY=existing-local-value\n")
        run.assert_not_called()

    def test_new_env_opens_notepad_and_revalidates(self):
        (self.root / ".env.example").write_text("OPENAI_API_KEY=\n")
        with patch.object(launcher, "WINDOWS", True), patch.object(launcher.subprocess, "call", side_effect=[1, 0]), patch.object(launcher, "run") as run, patch("builtins.input", return_value=""):
            launcher.configure("python")
        self.assertEqual((self.root / ".env").read_text(), "OPENAI_API_KEY=\n")
        run.assert_called_once_with(["notepad.exe", self.root / ".env"])

    def test_empty_key_stops_startup(self):
        (self.root / ".env").write_text("OPENAI_API_KEY=\n")
        with patch.object(launcher.subprocess, "call", return_value=1), patch.object(launcher, "run"), patch("builtins.input", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                launcher.configure("python")

    def prepare_manifests(self):
        (self.root / ".venv").mkdir()
        (self.root / "frontend/node_modules/next").mkdir(parents=True)
        for path in ["requirements.txt", "frontend/package.json", "frontend/package-lock.json"]:
            (self.root / path).write_text("test")

    def test_install_reuses_dependencies_and_detects_changes(self):
        self.prepare_manifests()
        with patch.object(launcher, "run") as run:
            launcher.install("python", "npm")
            self.assertEqual(run.call_count, 2)
            launcher.install("python", "npm")
            self.assertEqual(run.call_count, 2)
            (self.root / "requirements.txt").write_text("changed")
            launcher.install("python", "npm")
            self.assertEqual(run.call_count, 3)

    def test_failed_install_is_retried(self):
        self.prepare_manifests()
        with patch.object(launcher, "run", side_effect=subprocess.CalledProcessError(1, "pip")):
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.install("python", "npm")
        self.assertFalse((self.root / ".venv/setup/python").exists())

    def test_readiness_fails_if_child_exits(self):
        process = Mock()
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited"):
            launcher.ready("http://localhost:8000", [process])

    def test_readiness_timeout(self):
        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            launcher.ready("http://localhost:8000", [], timeout=0)

    def test_windows_cleanup_terminates_process_tree(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch.object(launcher, "WINDOWS", True), patch.object(launcher.subprocess, "run") as run:
            launcher.stop(process)
        self.assertEqual(run.call_args.args[0], ["taskkill", "/PID", "123", "/T", "/F"])
        process.wait.assert_called_once_with(timeout=10)

    def test_startup_failure_cleans_up_backend_and_preserves_corpus(self):
        python = self.root / ".venv" / ("Scripts/python.exe" if launcher.WINDOWS else "bin/python")
        python.parent.mkdir(parents=True)
        python.touch()
        process = Mock()
        with patch.object(launcher.sys, "argv", ["start.py", "--no-browser"]), patch.object(launcher.shutil, "which", return_value="tool"), patch.object(launcher, "check_ports"), patch.object(launcher, "install"), patch.object(launcher, "configure"), patch.object(launcher.subprocess, "check_output", return_value="v24.0.0"), patch.object(launcher.subprocess, "call", return_value=0), patch.object(launcher, "spawn", return_value=process), patch.object(launcher, "ready", side_effect=RuntimeError("failed")), patch.object(launcher, "stop") as stop, patch.object(launcher, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                launcher.main()
        stop.assert_called_once_with(process)
        run.assert_not_called()  # Existing corpus is not seeded again.


if __name__ == "__main__":
    unittest.main()
