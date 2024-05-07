```python
from kepy import Orbit, MU_EARTH

orbit = Orbit([2e6], mu=MU_EARTH)
print(orbit)
```

    Orbit([a=2e+6, e=0, i=0, Ω=0, ω=0, θ=0], μ=3.99e+14, type='circular')
