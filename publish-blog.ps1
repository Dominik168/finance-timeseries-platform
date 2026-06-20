# publish-blog.ps1
# Renders the Quarto blog and publishes it to GitHub Pages (docs/ folder)
# Usage: run from anywhere, or just: .\publish-blog.ps1

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Project\finance-timeseries-platform"
$blogDir = Join-Path $repoRoot "blog"
$blogDocsDir = Join-Path $blogDir "docs"
$rootDocsDir = Join-Path $repoRoot "docs"

Write-Host "==> Rendering Quarto blog..." -ForegroundColor Cyan
Set-Location $blogDir
quarto render

if ($LASTEXITCODE -ne 0) {
    Write-Host "Quarto render failed. Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "==> Copying rendered output to repo root /docs..." -ForegroundColor Cyan
Set-Location $repoRoot

if (Test-Path $rootDocsDir) {
    Remove-Item $rootDocsDir -Recurse -Force
}
Copy-Item $blogDocsDir $rootDocsDir -Recurse

# Ensure GitHub Pages doesn't try to run Jekyll on the output
$noJekyllFile = Join-Path $rootDocsDir ".nojekyll"
if (-not (Test-Path $noJekyllFile)) {
    New-Item $noJekyllFile -ItemType File | Out-Null
}

Write-Host "==> Committing and pushing to GitHub..." -ForegroundColor Cyan
git add docs
git add blog
$commitMessage = "Update blog - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git commit -m $commitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "Nothing to commit (no changes in docs/), or commit failed." -ForegroundColor Yellow
} else {
    git push
    Write-Host "==> Done! Blog will be live in a few minutes at:" -ForegroundColor Green
    Write-Host "    https://dominik168.github.io/finance-timeseries-platform/" -ForegroundColor Green
}
