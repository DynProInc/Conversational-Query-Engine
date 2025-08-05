#!/usr/bin/env python3
"""
Deployment check script for Conversational Query Engine
This script helps verify the environment and dependencies before deployment.
"""

import sys
import os
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 11:
        print("✅ Python 3.11.x is compatible")
        return True
    elif version.major == 3 and version.minor >= 13:
        print("⚠️  Python 3.13+ detected - may have SQLAlchemy compatibility issues")
        return False
    else:
        print("❌ Python version may not be optimal")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'snowflake-connector-python',
        'openai',
        'anthropic',
        'google.generativeai',
        'pandas',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_environment_variables():
    """Check if required environment variables are set"""
    required_vars = [
        'CLIENT_MTS_SNOWFLAKE_USER',
        'CLIENT_MTS_SNOWFLAKE_PASSWORD',
        'CLIENT_MTS_SNOWFLAKE_ACCOUNT',
        'CLIENT_MTS_OPENAI_API_KEY',
        'CLIENT_MTS_ANTHROPIC_API_KEY'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set")
            missing_vars.append(var)
    
    return len(missing_vars) == 0

def main():
    """Main deployment check"""
    print("🔍 Conversational Query Engine - Deployment Check")
    print("=" * 50)
    
    # Check Python version
    print("\n1. Checking Python version...")
    python_ok = check_python_version()
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Check environment variables
    print("\n3. Checking environment variables...")
    env_ok = check_environment_variables()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Deployment Check Summary:")
    print(f"Python version: {'✅' if python_ok else '❌'}")
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"Environment variables: {'✅' if env_ok else '⚠️'}")
    
    if python_ok and deps_ok:
        print("\n🎉 Deployment check passed! Your environment looks good.")
        return 0
    else:
        print("\n❌ Deployment check failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 