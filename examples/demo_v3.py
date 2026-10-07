"""Nexora v3 demo: new helpers in one pass (stdlib only, instant)."""
from nexora import Nexora

nx = Nexora()

# I1: any iterable in, I5: describe it
print(nx.describe(tuple("ABCABCABC"))["reason"])
print(nx.describe(range(5))["n"], "ranged observations")

# I9/I35: solve + narrate
print(nx.solve([2, 4, 6, 8])["explanation"])
nx.discover([float(i % 12) for i in range(120)])
pid = nx.top_patterns(1)["patterns"][0]["id"]
print(nx.explain_pattern(pid)["narrative"])

# I15/I17: seasonality + regimes agree on period-12 two-level data
print("seasonality:", nx.seasonality([float(i % 12) for i in range(120)])["period"])
print("regimes:", nx.regimes([5.0] * 20 + [50.0] * 20)["count"])

# I19/I49: frequencies -> discretize -> transitions round-trip
d = nx.discretize([1.0, 2.0, 3.0, 4.0] * 5, bins=2)
print("bins:", sorted(set(d["labels"])))
print("transitions:", nx.transitions(d["labels"])["count"])

# I22/I36: compare + signature
other = Nexora()
other.discover(list("XYZXYZXYZ"))
print(nx.compare(other)["reason"])
print("signature:", nx.state_signature()["signature"])

# I3/I7: global stream positions
nx2 = Nexora()
nx2.update([0.0] * 150)
print("changes:", nx2.update([5.0] * 150)["changes"][0]["stream_pos"] >= 150)
print("OK")
