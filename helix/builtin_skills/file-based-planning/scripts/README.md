# Multi-Script Phase Map

Document phase-to-script mapping here.
Example:
- phase: gather-context -> script: scripts/gather_context.py
- phase: execute-plan -> script: scripts/execute_plan.py
- phase: verify-output -> script: scripts/verify_output.py

Core agent should reason between phase executions using runtime stdout/stderr evidence.
