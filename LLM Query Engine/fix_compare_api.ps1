# PowerShell script to fix variable name inconsistencies in compare API for Gemini
$content = Get-Content -Path "api_server.py" -Raw

# Fix the compare API for Gemini
$content = $content -replace 'chart_recommendations=gemini_result\.get\("chart_recommendations", None\)', 'chart_recommendations=result.get("chart_recommendations", None)'
$content = $content -replace 'chart_error=gemini_result\.get\("chart_error", None\)', 'chart_error=result.get("chart_error", None)'

# Save the changes
$content | Set-Content -Path "api_server.py"

Write-Host "Fixed variable name inconsistencies in compare API for Gemini"
