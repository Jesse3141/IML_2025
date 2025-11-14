# PAC Learning Analysis
# This code can be copied directly into a Jupyter notebook cell

import numpy as np
import matplotlib.pyplot as plt
#great
# Define the PAC learning function
def min_samples_for_pac(
    h: int = 100,
    eta: float = 0.05,
    delta: float = 0.01
):
    """
    Compute minimal number of samples for PAC learnability.

    Parameters:
    -----------
    h : int
        Size of hypothesis class |H|
    eta : float
        Desired accuracy (epsilon)
    delta : float
        Confidence parameter (probability of failure)

    Returns:
    --------
    m : float
        Minimal number of samples required
    """
    m = 2 * np.log(2 * h / delta) / eta
    return m

print("="*70)
print("PAC LEARNING ANALYSIS")
print("="*70)

# ============================================================================
# Question 1: Compute minimal samples for base parameters
# ============================================================================
print("\n" + "="*70)
print("QUESTION 1: Compute minimal number of samples")
print("="*70)

h1 = 100
eta1 = 0.05
delta1 = 0.01

m1 = min_samples_for_pac(h=h1, eta=eta1, delta=delta1)

print(f"\nInput Parameters:")
print(f"  |H| (hypothesis class size) = {h1}")
print(f"  ε (desired accuracy)        = {eta1}")
print(f"  δ (confidence level)        = {delta1}")
print(f"\nOutput:")
print(f"  m (minimal samples required) = {m1:.2f}")
print(f"  m (rounded up)               = {int(np.ceil(m1))}")

# ============================================================================
# Question 2: Analyze change when hypotheses doubled
# ============================================================================
print("\n" + "="*70)
print("QUESTION 2: Analyze change when |H| is doubled")
print("="*70)

h2 = 200
m2 = min_samples_for_pac(h=h2, eta=eta1, delta=delta1)

print(f"\nInput Parameters:")
print(f"  |H| (hypothesis class size) = {h2} (doubled)")
print(f"  ε (desired accuracy)        = {eta1}")
print(f"  δ (confidence level)        = {delta1}")
print(f"\nOutput:")
print(f"  m (minimal samples required) = {m2:.2f}")
print(f"  m (rounded up)               = {int(np.ceil(m2))}")

print(f"\nComparison:")
print(f"  Original m (|H|=100):  {m1:.2f}")
print(f"  New m (|H|=200):       {m2:.2f}")
print(f"  Absolute change:       {m2 - m1:.2f}")
print(f"  Relative change:       {(m2 - m1) / m1 * 100:.2f}%")

print(f"\nInterpretation:")
print(f"  When |H| doubles, m increases by only {(m2 - m1) / m1 * 100:.2f}%")
print(f"  This is because m grows logarithmically with |H|, not linearly.")
print(f"  Doubling hypotheses adds only log(2) ≈ 0.693 to the numerator.")

# ============================================================================
# Question 3: Plot m as a function of ε
# ============================================================================
print("\n" + "="*70)
print("QUESTION 3: Plot m as a function of ε ∈ [0.01, 0.2]")
print("="*70)

# Generate epsilon values
eta_values = np.linspace(0.01, 0.2, 100)
m_vs_eta = [min_samples_for_pac(h=100, eta=eta, delta=0.01) for eta in eta_values]

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(eta_values, m_vs_eta, 'b-', linewidth=2)
ax.grid(True, alpha=0.3)
ax.set_xlabel('ε (Accuracy Parameter)', fontsize=12, fontweight='bold')
ax.set_ylabel('m (Minimal Sample Size)', fontsize=12, fontweight='bold')
ax.set_title('PAC Learning: Sample Complexity vs. Accuracy\n|H| = 100, δ = 0.01',
             fontsize=14, fontweight='bold')

# Add annotations for key points
ax.axvline(0.05, color='r', linestyle='--', alpha=0.5, label='ε = 0.05')
m_at_005 = min_samples_for_pac(h=100, eta=0.05, delta=0.01)
ax.plot(0.05, m_at_005, 'ro', markersize=8)
ax.annotate(f'm ≈ {m_at_005:.0f}', xy=(0.05, m_at_005),
            xytext=(0.08, m_at_005 + 200),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, fontweight='bold')

ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('pac_m_vs_epsilon.png', dpi=300, bbox_inches='tight')
print("\n  Plot saved as 'pac_m_vs_epsilon.png'")
plt.show()

print(f"\nKey Values:")
print(f"  m(ε=0.01) = {min_samples_for_pac(h=100, eta=0.01, delta=0.01):.2f}")
print(f"  m(ε=0.05) = {min_samples_for_pac(h=100, eta=0.05, delta=0.01):.2f}")
print(f"  m(ε=0.10) = {min_samples_for_pac(h=100, eta=0.10, delta=0.01):.2f}")
print(f"  m(ε=0.20) = {min_samples_for_pac(h=100, eta=0.20, delta=0.01):.2f}")

print(f"\nInterpretation:")
print(f"  • m decreases hyperbolically as ε increases (m ∝ 1/ε)")
print(f"  • Higher accuracy requirements (smaller ε) need exponentially more samples")
print(f"  • To halve the error (ε: 0.1 → 0.05), we need to double the samples")
print(f"  • The relationship is inversely proportional: reducing ε by factor k")
print(f"    increases m by factor k")

# ============================================================================
# Question 4: Plot m as a function of δ (log scale)
# ============================================================================
print("\n" + "="*70)
print("QUESTION 4: Plot m as a function of δ ∈ [10⁻⁴, 0.1] (log scale)")
print("="*70)

# Generate delta values on log scale
delta_values = np.logspace(-4, -1, 100)  # 10^-4 to 10^-1
m_vs_delta = [min_samples_for_pac(h=100, eta=0.05, delta=delta) for delta in delta_values]

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(delta_values, m_vs_delta, 'g-', linewidth=2)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlabel('δ (Failure Probability)', fontsize=12, fontweight='bold')
ax.set_ylabel('m (Minimal Sample Size)', fontsize=12, fontweight='bold')
ax.set_title('PAC Learning: Sample Complexity vs. Confidence\n|H| = 100, ε = 0.05',
             fontsize=14, fontweight='bold')

# Add annotations for key points
ax.axvline(0.01, color='r', linestyle='--', alpha=0.5, label='δ = 0.01')
m_at_001 = min_samples_for_pac(h=100, eta=0.05, delta=0.01)
ax.plot(0.01, m_at_001, 'ro', markersize=8)
ax.annotate(f'm ≈ {m_at_001:.0f}', xy=(0.01, m_at_001),
            xytext=(0.02, m_at_001 + 100),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, fontweight='bold')

ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('pac_m_vs_delta.png', dpi=300, bbox_inches='tight')
print("\n  Plot saved as 'pac_m_vs_delta.png'")
plt.show()

print(f"\nKey Values:")
print(f"  m(δ=0.0001) = {min_samples_for_pac(h=100, eta=0.05, delta=0.0001):.2f}")
print(f"  m(δ=0.001)  = {min_samples_for_pac(h=100, eta=0.05, delta=0.001):.2f}")
print(f"  m(δ=0.01)   = {min_samples_for_pac(h=100, eta=0.05, delta=0.01):.2f}")
print(f"  m(δ=0.1)    = {min_samples_for_pac(h=100, eta=0.05, delta=0.1):.2f}")

print(f"\nInterpretation:")
print(f"  • m increases logarithmically as δ decreases (m ∝ log(1/δ))")
print(f"  • Higher confidence (smaller δ) requires more samples, but the increase")
print(f"    is logarithmic, not linear")
print(f"  • Reducing δ by a factor of 10 adds only log(10) ≈ 2.3 to the numerator")
print(f"  • Going from δ=0.1 to δ=0.0001 (1000x more confident) increases m by")
print(f"    only ~{(min_samples_for_pac(h=100, eta=0.05, delta=0.0001) / min_samples_for_pac(h=100, eta=0.05, delta=0.1)):.1f}x")
print(f"  • This makes achieving high confidence relatively cheap in terms of samples")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)