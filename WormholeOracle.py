import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import random

class WormholeOracle:
    """
    Reinforcement Learning agent that dynamically optimizes wormhole connections.
    Learns which connections to strengthen, weaken, or create based on long-term retrieval efficiency.
    """
    def __init__(self, n_concepts: int, embedding_dim: int, lr: float = 0.01):
        self.n_concepts = n_concepts
        self.embedding_dim = embedding_dim

        # Neural network policy model (simple MLP)
        self.model = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output: connection strength score
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Wormhole graph with dynamic weights
        self.graph = nx.Graph()
        self.wormhole_weights = {}
        self.init_graph()

    def init_graph(self):
        """Initialize a small-world graph structure"""
        self.graph = nx.watts_strogatz_graph(n=self.n_concepts, k=max(1, int(self.n_concepts * 0.1)), p=0.2)
        for edge in self.graph.edges():
            self.wormhole_weights[edge] = 0.01  # Start with weak connections

    def predict_connection_strength(self, concept_a: torch.Tensor, concept_b: torch.Tensor) -> float:
        """Predicts the optimal connection strength between two concepts"""
        input_tensor = torch.cat((concept_a, concept_b)).unsqueeze(0)  # Concatenate embeddings
        return self.model(input_tensor).item()

    def update_wormhole(self, concept_i: int, concept_j: int, embedding_i: torch.Tensor, embedding_j: torch.Tensor, reward: float):
        """Updates wormhole connection strength based on reinforcement learning feedback"""
        predicted_strength = self.predict_connection_strength(embedding_i, embedding_j)

        # Ensure loss is a tensor
        loss = F.mse_loss(torch.tensor([predicted_strength], requires_grad=True), torch.tensor([reward], dtype=torch.float32))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update connection
        edge = tuple(sorted([concept_i, concept_j]))  # Ensure consistent edge representation
        if self.graph.has_edge(*edge):
            self.wormhole_weights[edge] = predicted_strength
        elif predicted_strength > 0.5:  # Create new wormhole if strong enough
            self.graph.add_edge(*edge)
            self.wormhole_weights[edge] = predicted_strength

    def decay_wormholes(self, decay_factor: float = 0.99):
        """Gradually weakens unused wormholes over time"""
        edges_to_remove = []  # Track edges to remove separately
        
        for edge in list(self.graph.edges()):  # Convert to list to avoid modification during iteration
            edge = tuple(sorted(edge))  # Ensure consistent edge representation
            if edge in self.wormhole_weights:
                self.wormhole_weights[edge] *= decay_factor
                if self.wormhole_weights[edge] < 0.01:
                    edges_to_remove.append(edge)  # Mark for removal
        
        # Remove edges after iteration
        for edge in edges_to_remove:
            self.graph.remove_edge(*edge)
            del self.wormhole_weights[edge]
    
    def get_best_connections(self, concept_id: int, top_n: int = 5):
        """Returns the strongest wormhole connections for a given concept"""
        connections = []
        
        for neighbor in self.graph.neighbors(concept_id):
            edge = tuple(sorted([concept_id, neighbor]))
            if edge in self.wormhole_weights:
                connections.append((neighbor, self.wormhole_weights[edge]))
        
        # Sort by connection strength (highest first) and take top_n
        connections.sort(key=lambda x: x[1], reverse=True)
        return connections[:top_n]


# --- Experiment: Simulating Evolution of Wormhole Connections ---

def run_wormhole_experiment(n_concepts=100, embedding_dim=128, n_iterations=500):
    oracle = WormholeOracle(n_concepts, embedding_dim)
    concept_embeddings = {i: torch.randn(embedding_dim) for i in range(n_concepts)}

    for step in range(n_iterations):
        concept_a, concept_b = random.sample(range(n_concepts), 2)
        embedding_a, embedding_b = concept_embeddings[concept_a], concept_embeddings[concept_b]

        # Simulate reward: If embeddings are similar, wormhole should strengthen
        reward = torch.cosine_similarity(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0)).item()

        oracle.update_wormhole(concept_a, concept_b, embedding_a, embedding_b, reward)
        oracle.decay_wormholes()

    # Get top connections for analysis
    top_connections = {i: oracle.get_best_connections(i) for i in range(n_concepts)}
    return top_connections

if __name__ == "__main__":
    results = run_wormhole_experiment()
    print("Wormhole Evolution Experiment Complete. Sample Connections:")
    for concept, connections in list(results.items())[:5]:
        print(f"Concept {concept}: {connections}")
