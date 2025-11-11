import re
import matplotlib.pyplot as plt

log_file = "output_mars_fixed_5k_50ep_gem/log.txt"

iterations = []
mars_values = []

print("Reading log file...")

with open(log_file, 'r') as f:
    for line in f:
        # Look for iteration number
        iter_match = re.search(r'iter:\s+(\d+)', line)
        
        if iter_match:
            # Try to find loss_mars (in the iteration line)
            mars_match = re.search(r'loss_mars:\s+([\d.e\-+]+)', line)
            
            if mars_match:
                iterations.append(int(iter_match.group(1)))
                mars_values.append(float(mars_match.group(1)))

print(f"\nSearching for 'loss_mars' in iteration lines...")
print(f"Found {len(mars_values)} values")

if len(mars_values) == 0:
    print("\n❌ No loss_mars found either!")
    print("\nLet's check what's actually in the log:")
    print("\n--- First 20 lines of log ---")
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 20:
                print(line.rstrip())
            else:
                break
    
    print("\n--- Lines containing 'iter:' ---")
    with open(log_file, 'r') as f:
        count = 0
        for line in f:
            if 'iter:' in line:
                print(line.rstrip())
                count += 1
                if count >= 3:
                    break
    
    print("\n--- Lines containing 'loss' ---")
    with open(log_file, 'r') as f:
        count = 0
        for line in f:
            if 'loss' in line.lower() and 'iter' in line:
                print(line.rstrip())
                count += 1
                if count >= 3:
                    break
else:
    print(f"\n✅ Found loss_mars values!")
    print(f"Iteration range: {min(iterations)} - {max(iterations)}")
    print(f"\nMars loss statistics:")
    print(f"  Min:  {min(mars_values):.6f}")
    print(f"  Max:  {max(mars_values):.6f}")
    print(f"  Mean: {sum(mars_values)/len(mars_values):.6f}")
    print(f"  Range: {max(mars_values) - min(mars_values):.6f}")
    
    unique_values = len(set([round(v, 4) for v in mars_values]))
    print(f"  Unique values (rounded to 4 decimals): {unique_values}")
    
    if unique_values < 5:
        print("\n⚠️  WARNING: Loss appears constant!")
    else:
        print("\n✅ Loss is varying (this is good)")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, mars_values, alpha=0.6, linewidth=0.5, marker='o', markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mars Loss')
    plt.title('Mars Loss Over Training')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mars_loss_trend.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved plot to: mars_loss_trend.png")
    
    # Show last 20 values
    print("\nLast 20 values:")
    for it, val in zip(iterations[-20:], mars_values[-20:]):
        print(f"  iter {it:6d}: {val:.6f}")