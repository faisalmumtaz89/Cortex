#!/usr/bin/env python3
"""
Cortex Installer - Clean, single-script installation for macOS Apple Silicon.
Usage: python3 install.py
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

# ANSI color codes for beautiful output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def print_banner():
    """Print installation banner."""
    print(f"{Colors.CYAN}")
    print("╭─────────────────────────────────────────────────────╮")
    print("│               CORTEX INSTALLER                      │")
    print("│    GPU-Accelerated LLM for Apple Silicon           │")
    print("╰─────────────────────────────────────────────────────╯")
    print(f"{Colors.RESET}\n")

def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")

def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {message}")

def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")

def print_info(message):
    """Print info message."""
    print(f"{Colors.BLUE}→{Colors.RESET} {message}")

def check_system():
    """Check system requirements."""
    print(f"{Colors.BOLD}System Check{Colors.RESET}\n")
    
    # Check OS
    if platform.system() != "Darwin":
        print_error("Cortex requires macOS")
        return False
    print_success("macOS detected")
    
    # Check architecture
    arch = platform.machine()
    if arch != "arm64":
        print_warning(f"Cortex is optimized for Apple Silicon (detected: {arch})")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    else:
        print_success(f"Apple Silicon detected ({arch})")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 11):
        print_error(f"Python 3.11+ required (you have {python_version.major}.{python_version.minor})")
        print_info("Install with: brew install python@3.11")
        return False
    print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return True

def clean_existing_installation():
    """Clean up any existing installation artifacts."""
    print(f"\n{Colors.BOLD}Cleaning Existing Installation{Colors.RESET}\n")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print_info("Removing existing virtual environment...")
        shutil.rmtree(venv_path)
        print_success("Cleaned existing venv")
    
    # Remove old build artifacts
    for pattern in ["*.egg-info", "build", "dist", "__pycache__"]:
        for path in Path(".").glob(f"**/{pattern}"):
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    
    print_success("Clean environment ready")

def setup_venv():
    """Setup fresh virtual environment."""
    print(f"\n{Colors.BOLD}Setting Up Environment{Colors.RESET}\n")
    
    print_info("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print_success("Virtual environment created")
    
    # Get paths
    if sys.platform == "win32":
        pip_path = Path("venv/Scripts/pip")
        python_path = Path("venv/Scripts/python")
    else:
        pip_path = Path("venv/bin/pip")
        python_path = Path("venv/bin/python")
    
    return str(pip_path), str(python_path)

def install_dependencies(pip_path):
    """Install core dependencies."""
    print(f"\n{Colors.BOLD}Installing Dependencies{Colors.RESET}\n")
    
    # Upgrade pip first
    print_info("Upgrading pip...")
    result = subprocess.run(
        [pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print_success("Pip upgraded")
    
    # Install Cortex with all dependencies from pyproject.toml
    print_info("Installing Cortex with dependencies...")
    print(f"{Colors.DIM}This may take a few minutes...{Colors.RESET}")
    
    result = subprocess.run(
        [pip_path, "install", "-e", "."],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        if result.stderr:
            print(f"{Colors.DIM}{result.stderr}{Colors.RESET}")
        print_warning("Some dependencies had issues, trying with --no-deps and manual installation...")
        
        # First install Cortex without dependencies
        subprocess.run([pip_path, "install", "--no-deps", "-e", "."], check=True)
        
        # Then try to install essential dependencies
        requirements_path = Path("requirements.txt")
        if requirements_path.exists():
            result = subprocess.run(
                [pip_path, "install", "-r", str(requirements_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print_success("Installed dependencies from requirements.txt")
                return True
            if result.stderr:
                print(f"{Colors.DIM}{result.stderr}{Colors.RESET}")

        essential_deps = [
            "torch>=2.1.0",
            "mlx>=0.30.4",
            "mlx-lm>=0.30.5",
            "transformers>=4.36.0",
            "sentencepiece>=0.1.99",
            "huggingface-hub>=0.19.0",
            "pyyaml>=6.0",
            "click>=8.1.0",
            "llama-cpp-python>=0.2.0"
        ]
        
        failed = []
        for dep in essential_deps:
            result = subprocess.run(
                [pip_path, "install", dep],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"  {Colors.DIM}✓ {dep}{Colors.RESET}")
            else:
                failed.append(dep)
                print(f"  {Colors.YELLOW}○ {dep} (failed){Colors.RESET}")
        
        if failed:
            critical = ["torch", "transformers", "pyyaml", "click"]
            critical_failed = [d for d in failed if any(c in d for c in critical)]
            if critical_failed:
                print_error(f"Failed to install critical dependencies: {', '.join(critical_failed)}")
                return False
            else:
                print_warning("Some optional dependencies failed, but core is installed")
    else:
        print_success("Cortex installed")
    
    return True

def create_launcher():
    """Create the cortex launcher script."""
    print(f"\n{Colors.BOLD}Creating Launcher{Colors.RESET}\n")
    
    # Try multiple locations for the launcher
    # First try /usr/local/bin (works immediately, may need sudo)
    # Then try ~/.local/bin (needs PATH update)
    
    launcher_locations = [
        Path("/usr/local/bin/cortex"),
        Path.home() / ".local" / "bin" / "cortex"
    ]
    
    cortex_home = Path.cwd()
    
    launcher_content = f'''#!/usr/bin/env python3
"""Cortex launcher - automatically manages environment."""

import os
import sys
import subprocess
from pathlib import Path

CORTEX_HOME = Path("{cortex_home}")
VENV_PATH = CORTEX_HOME / "venv"

def main():
    # Check if venv exists
    if not VENV_PATH.exists():
        print("Error: Cortex not installed. Please run install.py first.")
        print(f"  cd {{CORTEX_HOME}}")
        print("  python3 install.py")
        sys.exit(1)
    
    # Setup environment
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV_PATH)
    
    # Platform-specific paths
    if sys.platform == "win32":
        python_path = VENV_PATH / "Scripts" / "python"
        env["PATH"] = f"{{VENV_PATH / 'Scripts'}};{{env.get('PATH', '')}}"
    else:
        python_path = VENV_PATH / "bin" / "python"
        env["PATH"] = f"{{VENV_PATH / 'bin'}}:{{env.get('PATH', '')}}"
    
    env.pop("PYTHONHOME", None)
    
    # Run cortex
    args = [str(python_path), "-m", "cortex"] + sys.argv[1:]
    
    try:
        result = subprocess.run(args, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    # Try to install in system location first (works immediately)
    for location in launcher_locations:
        try:
            # Create parent directory if needed
            location.parent.mkdir(parents=True, exist_ok=True)
            
            # Write launcher script
            with open(location, 'w') as f:
                f.write(launcher_content)
            
            # Make executable
            location.chmod(0o755)
            launcher_path = location
            print_success(f"Launcher created at {launcher_path}")
            
            # If we used /usr/local/bin, it works immediately
            if "/usr/local/bin" in str(launcher_path):
                print_info("Command 'cortex' is ready to use immediately!")
            
            break
            
        except PermissionError:
            if location == launcher_locations[0]:
                print_warning(f"Cannot write to {location} (needs sudo)")
                # Try to create symlink with sudo
                print_info("Attempting to create system-wide command...")
                user_launcher = Path.home() / ".local" / "bin" / "cortex"
                
                # First create the user launcher
                user_launcher.parent.mkdir(parents=True, exist_ok=True)
                with open(user_launcher, 'w') as f:
                    f.write(launcher_content)
                user_launcher.chmod(0o755)
                
                # Now try to symlink it with sudo
                response = input("Install system-wide 'cortex' command with sudo? (y/N): ").strip().lower()
                if response == "y":
                    print_info("Creating system-wide 'cortex' command...")
                    print_info("You may be prompted for your password")
                    
                    # Use subprocess without capture to allow interactive sudo
                    result = subprocess.run(
                        ["sudo", "ln", "-sf", str(user_launcher), "/usr/local/bin/cortex"]
                    )
                    
                    if result.returncode == 0:
                        launcher_path = Path("/usr/local/bin/cortex")
                        print_success("System-wide 'cortex' command installed!")
                        print_info("Command 'cortex' is ready to use immediately!")
                        break
                    else:
                        print_warning("Could not create system-wide command")
                        print_info("You can manually run:")
                        print(f"   {Colors.CYAN}sudo ln -sf {user_launcher} /usr/local/bin/cortex{Colors.RESET}")
                        launcher_path = user_launcher
                        print_success(f"Launcher created at {launcher_path}")
                        break
                else:
                    launcher_path = user_launcher
                    print_success(f"Launcher created at {launcher_path}")
                    break
            else:
                print_error(f"Cannot create launcher at {location}")
                return None
        except Exception as e:
            print_warning(f"Failed to create at {location}: {e}")
            continue
    
    return launcher_path

def update_shell_config():
    """Update shell configuration to add cortex to PATH."""
    print(f"\n{Colors.BOLD}Updating Shell Configuration{Colors.RESET}\n")
    
    # Find shell config file
    shell_configs = [
        Path.home() / ".zshrc",
        Path.home() / ".bashrc",
        Path.home() / ".bash_profile"
    ]
    
    config_file = None
    for config in shell_configs:
        if config.exists():
            config_file = config
            break
    
    if not config_file:
        # Create .zshrc if none exist (macOS default)
        config_file = Path.home() / ".zshrc"
        config_file.touch()
    
    response = input("Add ~/.local/bin to your PATH in shell config? (y/N): ").strip().lower()
    if response != "y":
        print_info("Skipping shell configuration update")
        return None

    # Check if PATH update is needed
    path_line = 'export PATH="$HOME/.local/bin:$PATH"'
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    if path_line not in content:
        # Add PATH update
        with open(config_file, 'a') as f:
            f.write(f"\n# Cortex\n{path_line}\n")
        print_success(f"Updated {config_file.name}")
        return config_file
    else:
        print_info(f"PATH already configured in {config_file.name}")
        return None

def create_directories():
    """Create necessary directories."""
    print(f"\n{Colors.BOLD}Creating Directories{Colors.RESET}\n")
    
    # Models directory
    models_dir = Path.home() / "models"
    models_dir.mkdir(exist_ok=True)
    print_success(f"Models directory: ~/models")
    
    # Config directory
    config_dir = Path.home() / ".cortex"
    config_dir.mkdir(exist_ok=True)
    print_success(f"Config directory: ~/.cortex")

def verify_installation(python_path):
    """Verify the installation works."""
    print(f"\n{Colors.BOLD}Verifying Installation{Colors.RESET}\n")
    
    # Test import
    result = subprocess.run(
        [python_path, "-c", "import cortex; print('OK')"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and "OK" in result.stdout:
        print_success("Cortex imports correctly")
        return True
    else:
        print_error("Failed to import Cortex")
        if result.stderr:
            print(f"{Colors.DIM}{result.stderr}{Colors.RESET}")
        return False

def print_next_steps(shell_updated, launcher_path):
    """Print completion message and next steps."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}")
    print("╭─────────────────────────────────────────────────────╮")
    print("│                INSTALLATION COMPLETE                │")
    print("╰─────────────────────────────────────────────────────╯")
    print(f"{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.RESET}\n")
    
    # Check if cortex is in a system path
    if launcher_path and "/usr/local/bin" in str(launcher_path):
        print(f"1. Start Cortex immediately:")
        print(f"   {Colors.CYAN}cortex{Colors.RESET}\n")
    elif shell_updated:
        print(f"1. Reload your shell to enable 'cortex' command:")
        print(f"   {Colors.CYAN}source {shell_updated}{Colors.RESET}\n")
        print(f"2. Start Cortex:")
        print(f"   {Colors.CYAN}cortex{Colors.RESET}\n")
    else:
        print(f"1. Add to PATH manually:")
        print(f"   {Colors.CYAN}export PATH=\"$HOME/.local/bin:$PATH\"{Colors.RESET}\n")
        print(f"2. Start Cortex:")
        print(f"   {Colors.CYAN}cortex{Colors.RESET}\n")
    
    print(f"3. Download a model (inside Cortex):")
    print(f"   {Colors.CYAN}/download{Colors.RESET}\n")
    
    print(f"{Colors.DIM}Models directory: ~/models")
    print(f"Config directory: ~/.cortex{Colors.RESET}\n")
    
    # Always show direct run option
    print(f"{Colors.BOLD}Alternative (works immediately):{Colors.RESET}")
    print(f"   {Colors.CYAN}./venv/bin/python -m cortex{Colors.RESET}\n")

def main():
    """Main installation process."""
    try:
        print_banner()
        
        # System check
        if not check_system():
            print_error("\nInstallation aborted")
            sys.exit(1)
        
        # Clean existing installation
        clean_existing_installation()
        
        # Setup virtual environment
        pip_path, python_path = setup_venv()
        
        # Install dependencies
        if not install_dependencies(pip_path):
            print_error("\nInstallation failed")
            sys.exit(1)
        
        # Create launcher
        launcher_path = create_launcher()
        
        # Update shell config
        shell_updated = update_shell_config()
        
        # Create directories
        create_directories()
        
        # Verify installation
        if verify_installation(python_path):
            print_next_steps(shell_updated, launcher_path)
        else:
            print_warning("\nInstallation completed with warnings")
            print_info("Try running: ./venv/bin/python -m cortex")
    
    except KeyboardInterrupt:
        print_error("\n\nInstallation cancelled")
        sys.exit(130)
    except Exception as e:
        print_error(f"\nInstallation failed: {e}")
        import traceback
        print(f"{Colors.DIM}{traceback.format_exc()}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
