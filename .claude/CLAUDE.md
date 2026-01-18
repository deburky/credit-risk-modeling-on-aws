# Claude AI Coding Guidelines

## Code Style Rules

### DO NOT USE

1. **Decorative separator lines**
   - `print("=" * 80)`
   - `print("=" * 60)`
   - `print("-" * 50)`
   - Any decorative print statements using repeated characters

2. **Empty print statements for spacing**
   - `print()`
   - Used only for adding blank lines in output

3. **Bullet point summary statements**
   - `print("  • AppConfig configuration was updated without redeploying endpoint")`
   - `print("  - Key finding: ...")`
   - `print("  * Summary: ...")`
   - Any print statements with bullet points or indented summary text

4. **Emojis in output** (Python scripts only)
   - `print("✅ Success")`
   - `print("❌ Error")`
   - `print("📊 Step 1: ...")`
   - Any emoji characters in print statements
   - **Exception**: Emojis and colors are OK in Makefiles

5. **Extra spacing in Makefiles**
   - Empty `@echo ""` statements for spacing
   - Multiple blank lines between targets
   - Extra whitespace in echo statements

6. **Leading spaces in print statements**
   - `print("   Text with leading spaces")`
   - `print(f"    Testing etc")`
   - Any print statements with leading whitespace for indentation
   - Use plain text without leading spaces

### DO USE

- Direct, informative print statements without decorative elements
- Concise output that focuses on actionable information
- No extra spacing or formatting beyond what's necessary
- Plain text status indicators (e.g., "Success:", "Error:", "Step 1:")
- **Makefiles**: Emojis and colors are acceptable, but avoid extra spacing

## Rationale

- Cleaner, more professional output
- Easier to parse programmatically
- Less visual clutter
- Focus on content, not decoration
