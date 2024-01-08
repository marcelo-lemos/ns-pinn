# Navier-Stokes Physics Informed Neural Networks (NS-PINN)

[![Maintainability](https://api.codeclimate.com/v1/badges/0430c048641812716970/maintainability)](https://codeclimate.com/github/marcelo-lemos/ns-pinn/maintainability)

Pytorch Lightning implementation of a Physics Informed Neural Network for the Navier-Stokes equations.

## Prerequisites

- Docker

## Getting Started

1. Clone the repository:

```
git clone https://github.com/marcelo-lemos/ns-pinn
```

2. Build the Docker image:

```
docker build -t ns-pinn .
```

3. Run the Docker container:

```
docker run --gpus all -v ./data:app/data ns-pinn
```

## License

MIT License: [LICENSE](LICENSE)
