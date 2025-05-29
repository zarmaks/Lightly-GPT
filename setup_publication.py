#!/usr/bin/env python3
"""
Setup script for LightlyGPT publication
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ FAILED")
        if result.stderr:
            print(result.stderr)
        return False
    return True

def main():
    """Main setup function"""
    print("🔆 LightlyGPT Publication Setup")
    print("This script will prepare your project for publication")
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    for directory in ["build", "dist", "*.egg-info"]:
        if os.path.exists(directory):
            run_command(f"rmdir /s /q {directory}" if os.name == 'nt' else f"rm -rf {directory}", 
                       f"Removing {directory}")
    
    # Install build tools
    print("\n📦 Installing build tools...")
    if not run_command("pip install --upgrade build twine", "Installing build tools"):
        return False
    
    # Build the package
    print("\n🔨 Building package...")
    if not run_command("python -m build", "Building source distribution and wheel"):
        return False
    
    # Check the package
    print("\n🔍 Checking package...")
    if not run_command("twine check dist/*", "Checking package integrity"):
        return False
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Your package is ready for publication!")
    print("="*60)
    
    print("\n📋 Next steps:")
    print("1. Update pyproject.toml with your actual name and email")
    print("2. Create a GitHub repository and push your code")
    print("3. Test your package locally: pip install -e .")
    print("4. For PyPI TestPyPI: twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
    print("5. For PyPI production: twine upload dist/*")
    print("\n💡 Don't forget to:")
    print("- Add your OpenAI API key to GitHub Secrets for CI/CD")
    print("- Update the README.md with your GitHub username")
    print("- Consider adding screenshots to your README")
    
    print(f"\n📁 Built files are in: {os.path.abspath('dist')}")
    dist_files = os.listdir('dist') if os.path.exists('dist') else []
    for file in dist_files:
        print(f"  - {file}")

if __name__ == "__main__":
    main()
