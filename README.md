# MPAMC
Multi-Perspective Adaptive Manifold Compression (Physics Is For Cheaters)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MPAMC is a novel memory model inspired by theoretical physics and cognitive science. It integrates concepts from the holographic principle, hyperbolic geometry, wormholes, fractal memory, and quantum superposition to create a highly adaptive and efficient system for storing and retrieving information. The model aims to mimic some aspects of human memory, such as its ability to form associations, generalize concepts, and retrieve information from multiple perspectives. It's designed to be both a theoretical exploration and a practical tool for advanced information management.

Key Features:

*   **Holographic Encoding:** Compresses information to a "boundary" representation, inspired by the black hole information paradox, allowing for efficient reconstruction.
*   **Hyperbolic Space:** Represents concepts in a curved space-time, capturing hierarchical and relational structures naturally.
*   **Wormhole Connections:** Creates probabilistic shortcuts between semantically related concepts, enabling faster retrieval.
*   **Multi-Perspective Encoding:** Stores information from multiple viewpoints, enhancing robustness and allowing for nuanced understanding.
*   **Fractal Memory:** Organizes concepts in a scale-free hierarchical structure, providing adaptive levels of detail.
*   **Quantum Superposition Recall:** Retrieves multiple related memory states simultaneously, inspired by quantum superposition.
*   **Self-Organizing Data Flow:** Dynamically rewires memory connections based on usage patterns, optimizing for frequently accessed information.
*    **Wormhole Oracle:** A Reinforcement Learning agent that optimizes wormhole connections dynamically, improving retrieval efficiency.
* **Modular Design:** The system is broken up into classes allowing for the components to be modified, extended, or replaced.

## Architecture

The MPAMC model consists of the following interconnected components:

1.  **Holographic Encoder:** An autoencoder (using PyTorch) that compresses input data to a lower-dimensional boundary representation and reconstructs the original data from this boundary.
2.  **Hyperbolic Space:** A module that implements operations in hyperbolic space, including Mobius addition, exponential and logarithmic maps, and hyperbolic distance calculations.
3.  **Wormhole Network:** A graph-based structure that creates and manages probabilistic connections (wormholes) between concepts, facilitating rapid traversal of the memory space.
4.  **Multi-Perspective Encoder:** Encodes information from multiple perspectives using separate Holographic Encoders and integrates them into a unified latent representation.
5.  **Fractal Memory:** A hierarchical memory structure that stores concepts at different levels of detail, allowing for efficient retrieval at varying granularities.
6.  **Quantum Superposition Memory:** Stores multiple memory states with associated probabilities, enabling retrieval of a superposition of relevant concepts.
7.  **Self-Organizing Memory:** A key-value store that dynamically adjusts the importance of memories based on access frequency and decay, and maintains connections between concepts.
8.  **Wormhole Oracle:** A Reinforcement Learning (RL) agent that uses a neural network to dynamically predict and optimize the strength of wormhole connections, improving long-term retrieval.


## Usage

The `mpamc.py` file contains all the necessary classes and functions. Here's a basic example of how to use the model:

```python
import torch
from mpamc import MPAMC  # Assuming the code is in mpamc.py

# Create the model with specified dimensions
model = MPAMC(input_dim=768, n_perspectives=3)

# Create some sample data (replace with your actual data)
test_embedding = torch.randn(768)
test_perspectives = [torch.randn(768) for _ in range(3)]

# Store a concept
model.store_concept(
    concept_name="dog",
    embedding=test_embedding,
    perspective_data=test_perspectives,
    related_concepts=["animal", "pet"],
    importance=0.8
)

# Store related concepts
model.store_concept(concept_name="animal", embedding=torch.randn(768), importance=0.9)
model.store_concept(concept_name="pet", embedding=torch.randn(768), importance=0.7)
model.store_concept(concept_name="cat", embedding=torch.randn(768), related_concepts=["animal","pet"], importance=0.75)


# Retrieve the concept (with some noise added to the query)
query = test_embedding + 0.1 * torch.randn(768)
results = model.retrieve_concept(query, query_name="dog")

# Print the results
print("Direct lookup:", "Found" if 'direct_lookup' in results else "Not found")
print("Wormhole connections:", [r[0] for r in results.get('wormhole_results', [])])
print("Reconstruction similarity:", torch.nn.functional.cosine_similarity(
    results['holographic_reconstruction'].unsqueeze(0),
    test_embedding.unsqueeze(0)
).item())
print("Fractal Results:", results.get('fractal_results'))
print("Quantum Result:", results.get('quantum_result'))

# Run the wormhole evolution experiment
# from mpamc import run_wormhole_experiment # Assuming the code is in mpamc.py
# wormhole_results = run_wormhole_experiment()
# print("\nWormhole Evolution Experiment.  Sample Connections:")
# for concept, connections in list(wormhole_results.items())[:5]:
#     print(f"Concept {concept}: {connections}")

```
### Wormhole Oracle Example

The `WormholeOracle` is a reinforcement learning agent.  Here's how it's used within the provided code (and how you might use it for more extensive training):

```python
# Example usage within the run_wormhole_experiment function.
# This is already part of the provided code.
from mpamc import WormholeOracle, run_wormhole_experiment
results = run_wormhole_experiment(n_concepts=100, embedding_dim=64, n_iterations=1000) # Reduced dimensions for speed
print("Wormhole Evolution Experiment Complete. Sample Connections:")
for concept, connections in list(results.items())[:5]:
    print(f"Concept {concept}: {connections}")

```

Key aspects of the `WormholeOracle` usage:

*   **Initialization:**  A `WormholeOracle` is created with the number of concepts and embedding dimension.  It initializes its own neural network and a small-world graph.
*   **`predict_connection_strength`:** This method takes two concept embeddings as input and uses the neural network to predict the optimal connection strength between them.
*   **`update_wormhole`:** This is the core RL method.  It takes two concepts, their embeddings, and a reward as input. It calculates the loss between the predicted strength and the reward, and then updates the network's weights using backpropagation.
*   **`decay_wormholes`:**  This method simulates the weakening of unused connections over time.
*    **`get_best_connections`:** Retrieves the strongest wormhole connections for a concept.
*   **`run_wormhole_experiment`:** This function demonstrates how to train the `WormholeOracle` over multiple iterations.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports, feature requests, or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   This project was inspired by research on the holographic principle, hyperbolic geometry, wormholes, and quantum information theory.
* Thanks to the developers of PyTorch, NetworkX, SciPy, and scikit-learn.


