# Fix indentation in navigation_task_gate.py

# Read the file
with open('aerial_gym/task/navigation_task_gate/navigation_task_gate.py', 'r') as f:
    content = f.read()

# Split into lines
lines = content.split('\n')

# Find and fix the problematic line
for i, line in enumerate(lines):
    if 'self.sim_env.delete_env()' in line and line.strip() == 'self.sim_env.delete_env()':
        # Check if it needs proper indentation
        if not line.startswith('                '):  # Should be 16 spaces for proper indentation
            lines[i] = '                self.sim_env.delete_env()'
            print(f'Fixed indentation at line {i+1}')
            break

# Write back to file
with open('aerial_gym/task/navigation_task_gate/navigation_task_gate.py', 'w') as f:
    f.write('\n'.join(lines))

print("Indentation fixed!") 