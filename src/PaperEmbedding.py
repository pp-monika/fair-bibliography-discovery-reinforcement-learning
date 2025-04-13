import torch
import torch.nn as nn

class PaperEmbeddingModule(nn.Module):
    def __init__(self, num_topics, embedding_dim, topic_output_dim, pad_idx, numeric_feature_dim, final_dim):
        """
        Args:
            num_topics (int): Total number of unique topics.
            embedding_dim (int): Dimensionality of each learnable topic embedding.
            topic_output_dim (int): Intermediate dimension for the aggregated topic representation.
            pad_idx (int): The index for the padding token "<PAD>".
            numeric_feature_dim (int): Number of numeric features (e.g., 2 for publication_year and cited_by_count).
            final_dim (int): Desired dimensionality of the final fused paper embedding.
        """
        super(PaperEmbeddingModule, self).__init__()
        
        # Topic embedding branch.
        self.topic_embedding = nn.Embedding(num_topics, embedding_dim, padding_idx=pad_idx)
        # Transform the aggregated topic vector to an intermediate topic representation.
        self.topic_mlp = nn.Sequential(
            nn.Linear(embedding_dim, topic_output_dim),
            nn.ReLU()
        )
        
        # Numeric branch: project publication_year and cited_by_count into the same space.
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_feature_dim, topic_output_dim),
            nn.ReLU()
        )
        
        # Fusion layer: Combine topic and numeric representations.
        self.fusion_layer = nn.Sequential(
            nn.Linear(topic_output_dim * 2, final_dim),
            nn.ReLU()
        )
        
    def forward(self, topic_indices, topic_scores, numeric_features):
        """
        Args:
            topic_indices (LongTensor): Shape [batch_size, num_topics_per_paper] with padded topic indices.
            topic_scores (Tensor): Shape [batch_size, num_topics_per_paper] with topic relevance scores.
            numeric_features (Tensor): Shape [batch_size, numeric_feature_dim] with numeric features 
                                       (e.g., [publication_year, cited_by_count]).
        Returns:
            output (Tensor): Final paper embedding of shape [batch_size, final_dim].
        """
        # --- Topic Branch ---
        # Look up topic embeddings: shape [batch_size, num_topics_per_paper, embedding_dim]
        embeddings = self.topic_embedding(topic_indices)
        # Multiply each topic's embedding by its score.
        topic_scores_expanded = topic_scores.unsqueeze(-1)
        weighted_embeddings = embeddings * topic_scores_expanded
        # Sum across the topic dimension.
        weighted_sum = weighted_embeddings.sum(dim=1)
        # Normalize by sum of scores (avoid division by zero with small constant).
        sum_scores = topic_scores.sum(dim=1, keepdim=True) + 1e-8
        aggregated_topic = weighted_sum / sum_scores  # shape: [batch_size, embedding_dim]
        # Project to topic_output_dim.
        topic_repr = self.topic_mlp(aggregated_topic)   # shape: [batch_size, topic_output_dim]
        
        # --- Numeric Branch ---
        # numeric_features should be preprocessed externally. Here we assume a shape [batch_size, numeric_feature_dim].
        numeric_repr = self.numeric_projection(numeric_features)  # shape: [batch_size, topic_output_dim]
        
        # --- Fusion ---
        # Concatenate the representations along feature dimension.
        combined = torch.cat([topic_repr, numeric_repr], dim=1)     # shape: [batch_size, topic_output_dim * 2]
        final_embedding = self.fusion_layer(combined)                # shape: [batch_size, final_dim]
        return final_embedding