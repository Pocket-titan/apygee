# Apygee

<p align="start">
  <a href="https://github.com/Pocket-titan/apygee">
    <img src="logo.jpg" alt="Logo" width="200">
  </a>
</p>

Apygee (apogee + `py`) is a lightweight Python package for creating, manipulating and visualizing Kepler orbits.

## Installation

```bash
pip install apygee
```

## Usage

The main export of Apygee is the `Orbit` class, which stores the keplerian elements and provides easy access to the astrodynamical properties of the orbit. It also contains methods for visualizing the orbit, and for performing mauevers in order to transfer to other orbits.

```python
from apygee import Orbit, MU_EARTH

orbit = Orbit([2e6], mu=MU_EARTH)
print(orbit)
```

    Orbit([a=2e+6, e=0, i=0, Ω=0, ω=0, θ=0], μ=3.99e+14, type='circular')

## Examples

## Documentation

## Contributing

Contributions are welcome! For bug reports or feature requests, please submit an issue on the GitHub repository.

## Sources

## License

This project is licensed under the terms of the MIT license.
