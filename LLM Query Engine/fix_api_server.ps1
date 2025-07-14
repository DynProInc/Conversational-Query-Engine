# PowerShell script to fix variable name inconsistencies in api_server.py
$content = Get-Content -Path "api_server.py" -Raw

# Fix OpenAI endpoint
$content = $content -replace 'chart_recommendations=gemini_result\.get\("chart_recommendations"\)', 'chart_recommendations=result.get("chart_recommendations", [])'
$content = $content -replace 'chart_error=gemini_result\.get\("chart_error"\)', 'chart_error=result.get("chart_error")'

# Save the changes
$content | Set-Content -Path "api_server.py"

Write-Host "Fixed variable name inconsistencies in api_server.py"
