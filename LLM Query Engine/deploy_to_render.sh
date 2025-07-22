#!/bin/bash

# Conversational Query Engine - Render Deployment Helper Script

echo "🚀 Conversational Query Engine - Render Deployment Helper"
echo "========================================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository. Please initialize git and add your files:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    echo "   git remote add origin <your-repository-url>"
    echo "   git push -u origin main"
    exit 1
fi

# Check for required files
echo "📋 Checking required files..."

required_files=("api_server.py" "requirements.txt" "render.yaml" "runtime.txt" "Procfile")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo "✅ All required files found!"

# Check for environment files
echo "🔐 Checking environment configuration..."

if [ -d "config/clients/env" ]; then
    echo "⚠️  Found environment files in config/clients/env/"
    echo "   These contain sensitive information and should NOT be committed to git."
    echo "   Make sure they are in your .gitignore file."
    
    # Check if .gitignore contains the env files
    if grep -q "config/clients/env/\*\.env" .gitignore 2>/dev/null; then
        echo "✅ Environment files are properly ignored in .gitignore"
    else
        echo "❌ Environment files are NOT ignored in .gitignore"
        echo "   Please add 'config/clients/env/*.env' to your .gitignore file"
    fi
fi

# Check git status
echo "📊 Checking git status..."

if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes:"
    git status --short
    echo ""
    echo "Please commit your changes before deploying:"
    echo "   git add ."
    echo "   git commit -m 'Prepare for Render deployment'"
    echo "   git push"
else
    echo "✅ All changes are committed"
fi

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo ""
echo "1. Push your code to GitHub/GitLab:"
echo "   git push origin main"
echo ""
echo "2. Go to Render Dashboard: https://dashboard.render.com/"
echo ""
echo "3. Create a new Web Service and connect your repository"
echo ""
echo "4. Configure environment variables in Render dashboard:"
echo "   - CLIENT_MTS_SNOWFLAKE_USER"
echo "   - CLIENT_MTS_SNOWFLAKE_PASSWORD"
echo "   - CLIENT_MTS_SNOWFLAKE_ACCOUNT"
echo "   - CLIENT_MTS_SNOWFLAKE_WAREHOUSE"
echo "   - CLIENT_MTS_SNOWFLAKE_ROLE"
echo "   - CLIENT_MTS_SNOWFLAKE_DATABASE"
echo "   - CLIENT_MTS_SNOWFLAKE_SCHEMA"
echo "   - CLIENT_MTS_OPENAI_API_KEY"
echo "   - CLIENT_MTS_ANTHROPIC_API_KEY"
echo "   - CLIENT_MTS_GEMINI_API_KEY"
echo ""
echo "5. Deploy and test your API!"
echo ""
echo "📖 For detailed instructions, see README_RENDER.md"
echo ""
echo "🔗 Your API will be available at: https://your-service-name.onrender.com" 