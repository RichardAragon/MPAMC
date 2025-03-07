import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import networkx as nx
import math
from typing import Dict, List, Tuple, Optional, Set, Union

class HolographicEncoder(nn.Module):
    """
    Implements holographic information encoding inspired by the black hole information paradox.
    Instead of storing all internal information, it stores boundary information sufficient to reconstruct the interior.
    """
    def __init__(self, input_dim: int, boundary_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.boundary_dim = boundary_dim  # The "surface" information
        self.latent_dim = latent_dim  # Internal representation
        
        # Encoder compresses to boundary representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, boundary_dim),
            nn.Tanh()
        )
        
        # Decoder reconstructs from boundary to full representation
        self.decoder = nn.Sequential(
            nn.Linear(boundary_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Latent projector for internal representation
        self.latent_projector = nn.Linear(boundary_dim, latent_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to boundary representation"""
        return self.encoder(x)
    
    def decode(self, boundary: torch.Tensor) -> torch.Tensor:
        """Reconstruct full information from boundary"""
        return self.decoder(boundary)
    
    def to_latent(self, boundary: torch.Tensor) -> torch.Tensor:
        """Project boundary to latent space"""
        return self.latent_projector(boundary)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full encode-decode process with latent representation"""
        boundary = self.encode(x)
        latent = self.to_latent(boundary)
        reconstruction = self.decode(boundary)
        return reconstruction, latent


class HyperbolicSpace:
    """
    Implements operations in hyperbolic space for curved space-time memory representation.
    """
    def __init__(self, curvature: float = -1.0):
        self.c = curvature  # Curvature parameter
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from tangent space to hyperbolic space
        Maps vectors in tangent space to points in hyperbolic space
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        # Avoid division by zero
        v_norm = torch.clamp(v_norm, min=1e-10)
        
        # Formula for exponential map in hyperbolic space
        coef = torch.tanh(torch.sqrt(-self.c) * v_norm / 2) / v_norm
        return self.mobius_addition(x, v * coef)
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map from hyperbolic space to tangent space
        Maps points in hyperbolic space to vectors in tangent space
        """
        addition = self.mobius_addition(-x, y)
        addition_norm = torch.norm(addition, dim=-1, keepdim=True)
        # Avoid division by zero
        addition_norm = torch.clamp(addition_norm, min=1e-10)
        
        # Formula for logarithmic map in hyperbolic space
        return 2 / torch.sqrt(-self.c) * torch.atanh(torch.sqrt(-self.c) * addition_norm) / addition_norm * addition
    
    def mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mobius addition in hyperbolic space
        Analogous to vector addition in Euclidean space
        """
        x_dot_y = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        
        numerator = (1 + 2 * self.c * x_dot_y + self.c * y_norm_sq) * x + (1 - self.c * x_norm_sq) * y
        denominator = 1 + 2 * self.c * x_dot_y + self.c * self.c * x_norm_sq * y_norm_sq
        
        return numerator / denominator
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance between points
        """
        # Formula for hyperbolic distance
        addition = self.mobius_addition(-x, y)
        addition_norm = torch.norm(addition, dim=-1)
        return 2 / torch.sqrt(-self.c) * torch.atanh(torch.sqrt(-self.c) * addition_norm)


class WormholeNetwork:
    """
    Implements probabilistic wormholes as shortcuts in concept space.
    Creates direct links between semantically related but distant concepts.
    """
    def __init__(self, n_concepts: int, embedding_dim: int, sparsity: float = 0.95):
        self.n_concepts = n_concepts
        self.embedding_dim = embedding_dim
        self.sparsity = sparsity
        
        # Generate sparse graph with potential wormhole connections
        self.graph = nx.watts_strogatz_graph(n=n_concepts, k=int(n_concepts*0.1), p=0.1)
        self.wormhole_weights = {}
        self.init_wormhole_weights()
    
    def init_wormhole_weights(self):
        """Initialize weights for potential wormhole connections"""
        for edge in self.graph.edges():
            self.wormhole_weights[edge] = 0.01  # Initial low probability
        
    def update_wormhole(self, concept_i: int, concept_j: int, mutual_info: float):
        """
        Update the strength of a wormhole connection based on mutual information
        """
        if self.graph.has_edge(concept_i, concept_j):
            self.wormhole_weights[(concept_i, concept_j)] = mutual_info
        elif mutual_info > 0.5:  # Create new wormhole if mutual info is high enough
            self.graph.add_edge(concept_i, concept_j)
            self.wormhole_weights[(concept_i, concept_j)] = mutual_info
    
    def get_wormholes(self, concept_i: int, threshold: float = 0.3) -> List[Tuple[int, float]]:
        """
        Get all wormhole connections from a given concept above threshold strength
        """
        connections = []
        for neighbor in self.graph.neighbors(concept_i):
            weight = self.wormhole_weights.get((concept_i, neighbor), 0.0)
            if weight >= threshold:
                connections.append((neighbor, weight))
        return connections
    
    def shortest_path(self, start: int, end: int) -> List[int]:
        """Find shortest path considering wormhole connections"""
        # Use NetworkX's shortest path with weights
        weights = {edge: 1.0 / (self.wormhole_weights.get(edge, 0.01) + 0.01) for edge in self.graph.edges()}
        return nx.shortest_path(self.graph, source=start, target=end, weight=weights)


class MultiPerspectiveEncoder:
    """
    Implements multi-perspective entanglement by encoding information
    from multiple different viewpoints simultaneously.
    """
    def __init__(self, n_perspectives: int, dims: List[int]):
        self.n_perspectives = n_perspectives
        self.dims = dims
        
        # Create different encoders for different perspectives
        self.encoders = []
        for i, dim in enumerate(dims):
            self.encoders.append(HolographicEncoder(dim, dim // 4, dim // 8))
    
    def encode_multi(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode data from multiple perspectives"""
        assert len(inputs) == self.n_perspectives, "Number of inputs must match number of perspectives"
        
        encoded = []
        for i, x in enumerate(inputs):
            boundary = self.encoders[i].encode(x)
            encoded.append(boundary)
        
        return encoded
    
    def decode_perspective(self, encoded: List[torch.Tensor], perspective_idx: int) -> torch.Tensor:
        """Decode a specific perspective from the encoded representations"""
        return self.encoders[perspective_idx].decode(encoded[perspective_idx])
    
    def integrate_perspectives(self, encoded: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        """
        Integrate multiple perspectives with weights to create a unified latent representation
        """
        assert len(encoded) == len(weights) == self.n_perspectives, "Dimensions must match"
        
        # Project each perspective to its latent space
        latents = [self.encoders[i].to_latent(encoded[i]) for i in range(self.n_perspectives)]
        
        # Normalize all latents to same dimension for weighted combination
        normalized_dim = min(latent.shape[-1] for latent in latents)
        normalized_latents = [latent[:, :normalized_dim] for latent in latents]
        
        # Weighted combination
        integrated = torch.zeros_like(normalized_latents[0])
        for i, latent in enumerate(normalized_latents):
            integrated += weights[i] * latent
            
        return integrated / sum(weights)


class FractalMemory:
    """
    Implements adaptive fractal memory with scale-free hierarchical storage.
    Concepts are stored in a hierarchical structure where larger concepts
    contain compressed versions of smaller concepts.
    """
    def __init__(self, max_levels: int = 5, branching_factor: int = 4):
        self.max_levels = max_levels
        self.branching_factor = branching_factor
        
        # Hierarchical structure: level -> node_id -> (data, children)
        self.hierarchy = {level: {} for level in range(max_levels)}
        
        # Counter for generating unique IDs
        self.next_id = 0
    
    def get_node_id(self) -> int:
        """Generate a unique node ID"""
        node_id = self.next_id
        self.next_id += 1
        return node_id
    
    def add_concept(self, data: torch.Tensor, level: int = 0, parent_id: Optional[int] = None) -> int:
        """
        Add a concept to the fractal memory structure
        Returns the node ID of the inserted concept
        """
        # Create node
        node_id = self.get_node_id()
        self.hierarchy[level][node_id] = {"data": data, "children": [], "parent": parent_id}
        
        # Link to parent if exists
        if parent_id is not None and level > 0:
            self.hierarchy[level-1][parent_id]["children"].append(node_id)
            
        return node_id
    
    def compress_subtree(self, node_id: int, level: int) -> torch.Tensor:
        """
        Create a compressed representation of a node and all its children
        """
        if level >= self.max_levels - 1 or node_id not in self.hierarchy[level]:
            return self.hierarchy[level][node_id]["data"]
            
        # Get node data
        node_data = self.hierarchy[level][node_id]["data"]
        
        # Get children data
        children = self.hierarchy[level][node_id]["children"]
        if not children:
            return node_data
            
        # Compress children recursively
        children_data = []
        for child_id in children:
            child_data = self.compress_subtree(child_id, level + 1)
            children_data.append(child_data)
            
        # If we have children data, combine with current node
        if children_data:
            # Simple mean pooling as compression strategy
            children_tensor = torch.stack(children_data)
            compressed_children = torch.mean(children_tensor, dim=0)
            
            # Combine node with compressed children (simple weighted average)
            alpha = 0.7  # Weight for current node vs children
            combined = alpha * node_data + (1 - alpha) * compressed_children
            return combined
        
        return node_data
    
    def find_by_similarity(self, query: torch.Tensor, level: int = 0, k: int = 3) -> List[Tuple[int, float]]:
        """
        Find k most similar concepts at a given level using cosine similarity
        """
        if not self.hierarchy[level]:
            return []
            
        similarities = []
        for node_id, node_info in self.hierarchy[level].items():
            node_data = node_info["data"]
            similarity = F.cosine_similarity(query.unsqueeze(0), node_data.unsqueeze(0)).item()
            similarities.append((node_id, similarity))
            
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def drill_down(self, query: torch.Tensor, start_level: int = 0, target_level: int = None) -> List[Tuple[int, float]]:
        """
        Traverse hierarchy to find most relevant concepts at the target level
        """
        if target_level is None:
            target_level = self.max_levels - 1
            
        # Find top matches at starting level
        current_matches = self.find_by_similarity(query, start_level)
        
        # If we're already at the target level or have no matches, return
        if start_level == target_level or not current_matches:
            return current_matches
            
        # Otherwise, get children of top match and search those
        next_level_candidates = []
        for node_id, sim in current_matches:
            children = self.hierarchy[start_level][node_id]["children"]
            next_level_candidates.extend(children)
            
        # If no children, return current matches
        if not next_level_candidates:
            return current_matches
            
        # Recursively search at next level
        return self.drill_down(query, start_level + 1, target_level)


class QuantumSuperpositionMemory:
    """
    Implements quantum-inspired superposition recall to retrieve multiple
    memory states simultaneously.
    """
    def __init__(self, embedding_dim: int, n_states: int = 10):
        self.embedding_dim = embedding_dim
        self.n_states = n_states
        
        # Store multiple states with associated amplitudes (probabilities)
        self.states = []  # List of (state_vector, amplitude) tuples
    
    def add_state(self, state_vector: torch.Tensor, amplitude: float = 1.0):
        """Add a quantum-like state to the memory"""
        # Normalize state vector
        state_vector = state_vector / torch.norm(state_vector)
        
        # Check if similar state exists and update amplitude if so
        for i, (existing_state, existing_amplitude) in enumerate(self.states):
            similarity = F.cosine_similarity(state_vector.unsqueeze(0), existing_state.unsqueeze(0)).item()
            if similarity > 0.9:  # Very similar states
                # Combine states and amplitudes
                combined_amplitude = existing_amplitude + amplitude
                combined_state = (existing_state * existing_amplitude + state_vector * amplitude) / combined_amplitude
                combined_state = combined_state / torch.norm(combined_state)
                
                self.states[i] = (combined_state, combined_amplitude)
                return
                
        # If no similar state found, add as new state
        self.states.append((state_vector, amplitude))
        
        # If too many states, prune the one with lowest amplitude
        if len(self.states) > self.n_states:
            self.states.sort(key=lambda x: x[1], reverse=True)
            self.states = self.states[:self.n_states]
    
    def retrieve(self, query: torch.Tensor, collapse: bool = True) -> Union[torch.Tensor, List[Tuple[torch.Tensor, float]]]:
        """
        Retrieve states based on query
        If collapse=True, returns single most relevant state
        If collapse=False, returns all states with their probabilities
        """
        if not self.states:
            return torch.zeros(self.embedding_dim) if collapse else []
            
        # Normalize query
        query = query / torch.norm(query)
        
        # Calculate probabilities based on overlap with query
        retrieval_probs = []
        for state, amplitude in self.states:
            # Quantum-inspired probability calculation (squared inner product)
            overlap = F.cosine_similarity(query.unsqueeze(0), state.unsqueeze(0)).item()
            probability = amplitude * (overlap ** 2)
            retrieval_probs.append((state, probability))
            
        # Sort by probability
        retrieval_probs.sort(key=lambda x: x[1], reverse=True)
        
        if collapse:
            # Return only the highest probability state
            return retrieval_probs[0][0]
        else:
            # Return all states with probabilities
            return retrieval_probs


class SelfOrganizingMemory:
    """
    Implements self-organizing data flow with dynamic memory rewiring
    based on usage patterns.
    """
    def __init__(self, capacity: int = 1000, reinforcement_rate: float = 0.1, decay_rate: float = 0.01):
        self.capacity = capacity
        self.reinforcement_rate = reinforcement_rate
        self.decay_rate = decay_rate
        
        # Memory structure: key -> (value, importance, last_access_time)
        self.memory = {}
        
        # Access counter
        self.access_count = 0
        
        # Graph for connections between concepts
        self.connections = nx.DiGraph()
    
    def store(self, key: str, value: torch.Tensor, importance: float = 1.0):
        """Store a value with given importance"""
        self.memory[key] = (value, importance, self.access_count)
        
        # Add node to connection graph
        if key not in self.connections:
            self.connections.add_node(key)
            
        # Prune if over capacity
        if len(self.memory) > self.capacity:
            self._prune()
    
    def retrieve(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a value and update its importance"""
        if key not in self.memory:
            return None
            
        value, importance, last_access = self.memory[key]
        
        # Reinforce importance due to access
        new_importance = importance + self.reinforcement_rate
        self.memory[key] = (value, new_importance, self.access_count)
        
        # Update access counter
        self.access_count += 1
        
        return value
    
    def add_connection(self, key1: str, key2: str, strength: float = 1.0):
        """Add or strengthen connection between two memory items"""
        # Ensure both keys exist
        if key1 not in self.memory or key2 not in self.memory:
            return
            
        # Add connection
        if self.connections.has_edge(key1, key2):
            # Strengthen existing connection
            current_weight = self.connections[key1][key2]['weight']
            self.connections[key1][key2]['weight'] = current_weight + strength
        else:
            # Create new connection
            self.connections.add_edge(key1, key2, weight=strength)
    
    def _prune(self):
        """Remove least important items from memory"""
        # Calculate effective importance (base importance decayed by time)
        items_importance = []
        for key, (_, importance, last_access) in self.memory.items():
            # Decay importance based on time since last access
            time_factor = math.exp(-self.decay_rate * (self.access_count - last_access))
            effective_importance = importance * time_factor
            
            # Add connection importance
            if key in self.connections:
                connection_importance = sum(data['weight'] for _, _, data in self.connections.edges(key, data=True))
                effective_importance += connection_importance
                
            items_importance.append((key, effective_importance))
            
        # Sort by importance (lowest first)
        items_importance.sort(key=lambda x: x[1])
        
        # Remove least important items until under capacity
        items_to_remove = len(self.memory) - self.capacity
        for i in range(items_to_remove):
            key_to_remove = items_importance[i][0]
            del self.memory[key_to_remove]
            if key_to_remove in self.connections:
                self.connections.remove_node(key_to_remove)


class MPAMC(nn.Module):
    """
    Multi-Perspective Adaptive Manifold Compression Model
    Integrates all the above components to create a complete model
    """
    def __init__(self, 
                 input_dim: int = 768,
                 n_perspectives: int = 3, 
                 perspective_dims: List[int] = None,
                 boundary_compression_ratio: float = 4.0,
                 hyperbolic_curvature: float = -1.0,
                 memory_capacity: int = 1000):
        super().__init__()
        
        # Set default perspective dimensions if not provided
        if perspective_dims is None:
            perspective_dims = [input_dim] * n_perspectives
            
        # Initialize components
        self.input_dim = input_dim
        
        # Holographic encoder
        self.holo_encoder = HolographicEncoder(
            input_dim=input_dim,
            boundary_dim=int(input_dim / boundary_compression_ratio),
            latent_dim=int(input_dim / (boundary_compression_ratio * 2))
        )
        
        # Hyperbolic space for curved space-time memory
        self.hyperbolic_space = HyperbolicSpace(curvature=hyperbolic_curvature)
        
        # Wormhole network for concept shortcuts
        self.wormhole_network = WormholeNetwork(
            n_concepts=1000,  # Start with space for 1000 concepts
            embedding_dim=int(input_dim / boundary_compression_ratio)
        )
        
        # Multi-perspective encoder
        self.multi_perspective = MultiPerspectiveEncoder(
            n_perspectives=n_perspectives,
            dims=perspective_dims
        )
        
        # Fractal memory for hierarchical storage
        self.fractal_memory = FractalMemory(
            max_levels=4,
            branching_factor=4
        )
        
        # Quantum superposition memory
        self.quantum_memory = QuantumSuperpositionMemory(
            embedding_dim=int(input_dim / boundary_compression_ratio),
            n_states=20
        )
        
        # Self-organizing memory
        self.self_organizing_memory = SelfOrganizingMemory(
            capacity=memory_capacity
        )
        
        # Concept mapping
        self.concept_ids = {}  # String -> numeric ID mapping
        self.next_concept_id = 0
    
    def get_concept_id(self, concept_name: str) -> int:
        """Get or create ID for a concept"""
        if concept_name not in self.concept_ids:
            self.concept_ids[concept_name] = self.next_concept_id
            self.next_concept_id += 1
        return self.concept_ids[concept_name]
    
    def store_concept(self, 
                     concept_name: str, 
                     embedding: torch.Tensor, 
                     perspective_data: List[torch.Tensor] = None,
                     related_concepts: List[str] = None,
                     importance: float = 1.0):
        """
        Store a concept in all memory systems
        """
        # Get concept ID
        concept_id = self.get_concept_id(concept_name)
        
        # 1. Store in holographic encoder
        boundary = self.holo_encoder.encode(embedding)
        
        # 2. Store in hyperbolic space
        # (Just use boundary as hyperbolic coordinate)
        
        # 3. Update wormhole connections
        if related_concepts:
            for related in related_concepts:
                related_id = self.get_concept_id(related)
                self.wormhole_network.update_wormhole(concept_id, related_id, 0.5)  # Default connection strength
        
        # 4. Store in multi-perspective encoder if perspective data provided
        multi_encoded = None
        if perspective_data:
            multi_encoded = self.multi_perspective.encode_multi(perspective_data)
        
        # 5. Store in fractal memory
        level = min(int(importance * 3), 3)  # Map importance to level (0-3)
        fractal_id = self.fractal_memory.add_concept(embedding, level=level)
        
        # 6. Store in quantum memory
        self.quantum_memory.add_state(boundary, amplitude=importance)
        
        # 7. Store in self-organizing memory
        self.self_organizing_memory.store(concept_name, embedding, importance=importance)
        
        # Store connections if related concepts provided
        if related_concepts:
            for related in related_concepts:
                self.self_organizing_memory.add_connection(concept_name, related, strength=0.5)
    
    def retrieve_concept(self, 
                         query_embedding: torch.Tensor, 
                         query_name: Optional[str] = None,
                         perspective_idx: Optional[int] = None,
                         use_wormholes: bool = True,
                         collapse_quantum: bool = True) -> Dict:
        """
        Retrieve concept information using all memory systems
        Returns dict with different retrieval methods
        """
        results = {}
        
        # 1. Encode query to boundary representation
        boundary = self.holo_encoder.encode(query_embedding)
        
        # 2. Try exact name lookup if provided
        if query_name and query_name in self.self_organizing_memory.memory:
            direct_result = self.self_organizing_memory.retrieve(query_name)
            if direct_result is not None:
                results['direct_lookup'] = direct_result
        
        # 3. Use wormhole connections if available
        if use_wormholes and query_name and query_name in self.concept_ids:
            query_id = self.concept_ids[query_name]
            wormhole_connections = self.wormhole_network.get_wormholes(query_id)
            
            if wormhole_connections:
                wormhole_results = []
                for connected_id, strength in wormhole_connections:
                    # Get concept name from ID (reverse lookup)
                    for name, cid in self.concept_ids.items():
                        if cid == connected_id:
                            connected_concept = name
                            if connected_concept in self.self_organizing_memory.memory:
                                concept_emb = self.self_organizing_memory.retrieve(connected_concept)
                                wormhole_results.append((connected_concept, concept_emb, strength))
                            break
                
                results['wormhole_results'] = wormhole_results
        
        # 4. Retrieve from fractal memory
        fractal_results = self.fractal_memory.find_by_similarity(query_embedding)
        if fractal_results:
            results['fractal_results'] = fractal_results
        
        # 5. Retrieve from quantum memory
        quantum_result = self.quantum_memory.retrieve(boundary, collapse=collapse_quantum)
        results['quantum_result'] = quantum_result
        
        # 6. Reconstruct from holographic boundary
        reconstruction = self.holo_encoder.decode(boundary)
        results['holographic_reconstruction'] = reconstruction
        
        return results
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - encode and reconstruct
        """
        # Simple encode-decode for basic usage
        boundary = self.holo_encoder.encode(x)
        reconstruction = self.holo_encoder.decode(boundary)
        return reconstruction


# Demo using the MPAMC model
def demo_mpamc():
    # Create model
    model = MPAMC(input_dim=768, n_perspectives=3)
    
    # Create some test data
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
    model.store_concept(
        concept_name="animal",
        embedding=torch.randn(768),
        importance=0.9
    )
    
    model.store_concept(
        concept_name="pet",
        embedding=torch.randn(768),
        importance=0.7
    )
    
    # Retrieve the concept
    query = test_embedding + 0.1 * torch.randn(768)  # Add some noise
    results = model.retrieve_concept(query, query_name="dog")
    
    print("Direct lookup:", "Found" if 'direct_lookup' in results else "Not found")
    print("Wormhole connections:", [r[0] for r in results.get('wormhole_results', [])])
    print("Reconstruction similarity:", F.cosine_similarity(
        results['holographic_reconstruction'].unsqueeze(0), 
        test_embedding.unsqueeze(0)
    ).item())
    
    return model, results


if __name__ == "__main__":
    # Run demo
    model, results = demo_mpamc()
    print("MPAMC model demonstration complete")
