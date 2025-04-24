# Stop on errors
$ErrorActionPreference = "Stop"

# Current package names
$CURRENT_PROJECT_NAME = "python-uv-template"
$CURRENT_PACKAGE_NAME = "python_uv_template"

# Get new package name (from argument or current directory name)
if ($args.Count -ge 1) {
    $NEW_PROJECT_NAME = $args[0]
} else {
    $NEW_PROJECT_NAME = (Get-Item -Path ".").Name
}

# Convert project name to package name (replace hyphens with underscores)
$NEW_PACKAGE_NAME = $NEW_PROJECT_NAME -replace "-", "_"

Write-Host "Starting package rename:"
Write-Host "Project name: $CURRENT_PROJECT_NAME → $NEW_PROJECT_NAME"
Write-Host "Package name: $CURRENT_PACKAGE_NAME → $NEW_PACKAGE_NAME"

# Update pyproject.toml
if (Test-Path "pyproject.toml") {
    Write-Host "Updating pyproject.toml..."
    (Get-Content "pyproject.toml") -replace "name = ""$CURRENT_PROJECT_NAME""", "name = ""$NEW_PROJECT_NAME""" | Set-Content "pyproject.toml"
}

# Rename package directory
if (Test-Path "src\$CURRENT_PACKAGE_NAME") {
    Write-Host "Renaming package directory..."

    # Create new directory if it doesn't exist
    if (-not (Test-Path "src\$NEW_PACKAGE_NAME")) {
        New-Item -Path "src\$NEW_PACKAGE_NAME" -ItemType Directory -Force | Out-Null
    }

    # Copy all files (including hidden files)
    Get-ChildItem -Path "src\$CURRENT_PACKAGE_NAME" -Force | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination "src\$NEW_PACKAGE_NAME\" -Recurse -Force
    }

    # Remove old directory
    Remove-Item -Path "src\$CURRENT_PACKAGE_NAME" -Recurse -Force
}

# Update imports in Python files
Write-Host "Updating Python imports..."
Get-ChildItem -Path . -Filter "*.py" -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match $CURRENT_PACKAGE_NAME) {
        $content = $content -replace "import $CURRENT_PACKAGE_NAME", "import $NEW_PACKAGE_NAME"
        $content = $content -replace "from $CURRENT_PACKAGE_NAME", "from $NEW_PACKAGE_NAME"
        Set-Content -Path $_.FullName -Value $content -NoNewline
    }
}

# Update references in documentation
Write-Host "Updating references in documentation..."
Get-ChildItem -Path . -Include "*.md" -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match $CURRENT_PROJECT_NAME -or $content -match $CURRENT_PACKAGE_NAME) {
        $content = $content -replace $CURRENT_PROJECT_NAME, $NEW_PROJECT_NAME
        $content = $content -replace $CURRENT_PACKAGE_NAME, $NEW_PACKAGE_NAME
        Set-Content -Path $_.FullName -Value $content -NoNewline
    }
}

# Update devcontainer settings
if (Test-Path ".devcontainer\devcontainer.json") {
    Write-Host "Updating devcontainer settings..."
    $content = Get-Content ".devcontainer\devcontainer.json" -Raw
    $content = $content -replace $CURRENT_PROJECT_NAME, $NEW_PROJECT_NAME
    Set-Content -Path ".devcontainer\devcontainer.json" -Value $content -NoNewline
}

# Update VSCode settings
if (Test-Path ".vscode\settings.json") {
    Write-Host "Updating VSCode settings..."
    $content = Get-Content ".vscode\settings.json" -Raw
    $content = $content -replace $CURRENT_PROJECT_NAME, $NEW_PROJECT_NAME
    $content = $content -replace $CURRENT_PACKAGE_NAME, $NEW_PACKAGE_NAME
    Set-Content -Path ".vscode\settings.json" -Value $content -NoNewline
}

Write-Host "Package rename completed successfully!"
Write-Host "New project name: $NEW_PROJECT_NAME"
Write-Host "New package name: $NEW_PACKAGE_NAME"
