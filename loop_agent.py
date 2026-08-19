"""
Loop Agent: Analyzes and fixes C# code iteratively using Claude.
Each iteration: analyze → propose fix → apply fix → re-analyze → repeat until clean.
"""

import os
import re
import json
import anthropic

client = anthropic.Anthropic()
MAX_ITERATIONS = 8

SYSTEM_PROMPT = """You are an expert C# code reviewer and fixer.

Each turn you will receive C# source code. Your job is to:
1. Identify ONE bug or issue at a time (the most critical one remaining)
2. Explain the bug clearly
3. Return the COMPLETE fixed C# file (not just a snippet)

Use the report_analysis tool to return your findings."""

TOOL_SCHEMA = {
    "name": "report_analysis",
    "description": "Report the result of analyzing and fixing C# code.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bug_found": {
                "type": "boolean",
                "description": "Whether a bug was found"
            },
            "bug_description": {
                "type": "string",
                "description": "Clear description of the bug found"
            },
            "bug_location": {
                "type": "string",
                "description": "ClassName.MethodName or line area"
            },
            "fix_description": {
                "type": "string",
                "description": "What was changed and why"
            },
            "fixed_code": {
                "type": "string",
                "description": "The complete fixed C# file content. Omit if no bug found."
            }
        },
        "required": ["bug_found"]
    }
}


def analyze_and_fix(code: str) -> dict:
    """Send code to Claude, get back analysis + fixed code via tool use."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "auto"},
        messages=[
            {
                "role": "user",
                "content": f"Analyze and fix this C# code:\n\n```csharp\n{code}\n```"
            }
        ]
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "report_analysis":
            return block.input
    return {"bug_found": False}


def run_loop_agent(csharp_file: str):
    print("=" * 60)
    print("  C# LOOP AGENT - Bug Analyzer & Fixer")
    print("=" * 60)

    with open(csharp_file, "r") as f:
        current_code = f.read()

    print(f"\n📂 Loaded: {csharp_file}")
    print(f"   Lines: {len(current_code.splitlines())}\n")

    bugs_fixed = []
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"{'─'*60}")
        print(f"🔍 Iteration {iteration}: Analyzing code...")

        result = analyze_and_fix(current_code)

        if not result.get("bug_found"):
            print("\n✅ No more bugs found! Code is clean.")
            break

        bug_desc  = result.get("bug_description", "Unknown bug")
        bug_loc   = result.get("bug_location", "Unknown location")
        fix_desc  = result.get("fix_description", "")
        fixed_code = result.get("fixed_code", "")

        print(f"\n🐛 Bug found at: {bug_loc}")
        print(f"   Issue : {bug_desc}")
        print(f"   Fix   : {fix_desc}")

        bugs_fixed.append({
            "iteration": iteration,
            "location": bug_loc,
            "bug": bug_desc,
            "fix": fix_desc
        })

        if fixed_code:
            current_code = fixed_code
            print(f"   ✏️  Code updated.")
        else:
            print("   ⚠️  No fixed_code returned, stopping.")
            break

    # Save the final fixed file
    output_path = csharp_file.replace(".cs", "_fixed.cs")
    with open(output_path, "w") as f:
        f.write(current_code)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {len(bugs_fixed)} bug(s) fixed in {iteration} iteration(s)")
    print(f"{'='*60}")
    for b in bugs_fixed:
        print(f"\n  [{b['iteration']}] {b['location']}")
        print(f"      Bug: {b['bug']}")
        print(f"      Fix: {b['fix']}")

    print(f"\n💾 Fixed file saved to: {output_path}")
    return output_path, bugs_fixed


if __name__ == "__main__":
    import sys
    file = sys.argv[1] if len(sys.argv) > 1 else "BuggyApp.cs"
    run_loop_agent(file)

