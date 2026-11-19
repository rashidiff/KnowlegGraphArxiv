"""One-command local development launcher (Python 3.11, Node 20.9+)."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"


def run(command, **kwargs):
    subprocess.run([str(part) for part in command], check=True, cwd=ROOT, **kwargs)


def install(python, npm):
    state = ROOT / ".venv" / "setup"
    state.mkdir(exist_ok=True)
    for name, files, command, present in [
        ("python", ["requirements.txt"],
         [python, "-m", "pip", "install", "-r", "requirements.txt"], True),
        ("node", ["frontend/package.json", "frontend/package-lock.json"],
         [npm, "ci", "--prefix", str(ROOT / "frontend")],
         (ROOT / "frontend/node_modules/next").exists()),
    ]:
        digest = hashlib.sha256(b"".join((ROOT / f).read_bytes() for f in files)).hexdigest()
        stamp = state / name
        if not present or not stamp.exists() or stamp.read_text() != digest:
            run(command)
            stamp.write_text(digest)


def configure(python, edit=False):
    env = ROOT / ".env"
    if not env.exists():
        # Exclusive creation protects existing credentials.
        with env.open("x", encoding="utf-8") as target:
            target.write((ROOT / ".env.example").read_text(encoding="utf-8"))
    check = [python, "-c", "from dotenv import dotenv_values; import os,sys; "
             "v=os.getenv('OPENAI_API_KEY') or dotenv_values('.env').get('OPENAI_API_KEY'); "
             "sys.exit(0 if v and v.strip() else 1)"]
    if edit or subprocess.call(check, cwd=ROOT) != 0:
        print("Enter your API key in .env, save it, then close the editor.", flush=True)
        if WINDOWS:
            run(["notepad.exe", env])
        else:
            import shlex
            run([*shlex.split(os.getenv("EDITOR", "nano")), env])
        input("After saving .env, press Enter to continue: ")
    if subprocess.call(check, cwd=ROOT) != 0:
        raise RuntimeError("OPENAI_API_KEY is empty. Run again and configure .env.")


def check_ports():
    for port in (8000, 3000):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is busy. Stop the existing server and retry.") from exc


def spawn(command, **kwargs):
    return subprocess.Popen([str(p) for p in command], cwd=ROOT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0,
                            start_new_session=not WINDOWS, **kwargs)


def stop(process):
    if WINDOWS:
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if not WINDOWS:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def ready(url, processes, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("A server exited during startup. See its output above.")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}. See server output above.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-env", action="store_true", help="Reopen API configuration")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--seed-count", type=int, default=100, help="Initial corpus size (default: 100)")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Install Python 3.11 and run start.cmd (or python3.11 scripts/start.py).")
    if args.seed_count < 1:
        parser.error("--seed-count must be positive")
    npm = shutil.which("npm.cmd" if WINDOWS else "npm")
    node = shutil.which("node")
    if not npm or not node:
        raise RuntimeError("Install Node.js 20.9+ with npm, then reopen the terminal.")
    version = subprocess.check_output([node, "--version"], text=True).strip().lstrip("v")
    if tuple(map(int, version.split(".")[:2])) < (20, 9):
        raise RuntimeError("Node.js 20.9+ is required.")
    check_ports()
    python = ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
    if not python.exists():
        run([sys.executable, "-m", "venv", ROOT / ".venv"])
    install(python, npm)
    configure(python, args.edit_env)
    # Ask the selected database rather than relying on a local seed marker.
    probe = [python, "-c", "from backend.database.manager import get_db; import sys; "
             "sys.exit(0 if get_db().get_graph_data(max_nodes=1)['nodes'] else 3)"]
    status = subprocess.call(probe, cwd=ROOT)
    if status == 3:
        print("Preparing initial papers and downloading the embedding model; this may take several minutes.", flush=True)
        run([python, "-m", "backend.scripts.seed_data", str(args.seed_count)])
    elif status:
        raise RuntimeError("Could not inspect the corpus. See database output above.")
    processes = []
    try:
        processes.append(spawn([python, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]))
        ready("http://127.0.0.1:8000/", processes)
        # Next loads root .env explicitly, including NEXT_PUBLIC_API_BASE_URL.
        run_env = os.environ.copy()
        values = subprocess.check_output([python, "-c", "import json; from dotenv import dotenv_values; "
                                          "print(json.dumps(dotenv_values('.env')))"], cwd=ROOT, text=True)
        import json
        run_env.update({k: v for k, v in json.loads(values).items()
                        if k.startswith("NEXT_PUBLIC_") and v is not None and k not in run_env})
        processes.append(spawn([node, ROOT / "frontend/node_modules/next/dist/bin/next", "dev",
                                str(ROOT / "frontend"), "--hostname", "127.0.0.1", "--port", "3000"], env=run_env))
        ready("http://127.0.0.1:3000/", processes)
        print("Ready: http://localhost:3000 — press Ctrl+C to stop both servers.", flush=True)
        if not args.no_browser:
            webbrowser.open("http://localhost:3000")
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A server stopped. Shutting down both services.")
            time.sleep(0.5)
    finally:
        for process in reversed(processes):
            stop(process)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError, OSError, subprocess.SubprocessError, EOFError) as error:
        print(f"Setup/start failed: {error}", file=sys.stderr)
        sys.exit(1)
