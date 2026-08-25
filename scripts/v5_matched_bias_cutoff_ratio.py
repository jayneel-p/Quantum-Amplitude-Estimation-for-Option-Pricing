# Does the QCV cutoff advantage survive under matched BIAS budgets (the
# audit's convention) instead of matched exceedance fractions (the paper's)?
# For each bias budget b: find B_A with E[(payA-B_A)+]=b and B_R with
# E[(R-B_R)+]=b; report B_A/B_R.  2M paths, empirical curves.
import math, numpy as np
S0,K,R_,SIG,T,N = 100.0,100.0,0.05,0.2,1.0,252
rng = np.random.default_rng(2026_0713)
M, CH = 2_000_000, 250_000
payA_l, gap_l = [], []
dt=T/N; dr=(R_-0.5*SIG**2)*dt; v=SIG*math.sqrt(dt)
done=0
while done<M:
    m=min(CH,M-done)
    z=rng.standard_normal((m,N))
    logs=math.log(S0)+np.cumsum(dr+v*z,axis=1)
    A=np.exp(logs).mean(axis=1); G=np.exp(logs.mean(axis=1))
    payA_l.append(np.maximum(A-K,0).astype(np.float64))
    gap_l.append((np.maximum(A-K,0)-np.maximum(G-K,0)).astype(np.float64))
    done+=m
payA=np.concatenate(payA_l); resid=np.concatenate(gap_l)
def bias(x,B):
    e=x[x>B]; return float((e-B).sum())/M
def find_B(x,target,lo,hi):
    for _ in range(80):
        mid=0.5*(lo+hi)
        if bias(x,mid)>target: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)
print(f"{'bias budget':>12} {'B_A':>8} {'B_R':>7} {'ratio':>7}  {'paths>B_A':>9} {'paths>B_R':>9}")
for b in [3e-3,1e-3,3e-4,1e-4,3e-5,1e-5]:
    BA=find_B(payA,b,1.0,400.0); BR=find_B(resid,b,0.1,60.0)
    nA=int((payA>BA).sum()); nR=int((resid>BR).sum())
    print(f"{b:>12.0e} {BA:>8.2f} {BR:>7.2f} {BA/BR:>7.1f}  {nA:>9} {nR:>9}")
print("\n(paper's matched-exceedance convention gave 16.0 at the base case;")
print(" entries with <~50 tail paths are extrapolation territory)")
