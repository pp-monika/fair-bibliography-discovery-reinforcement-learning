import gym
from gym import spaces
import numpy as np
import random
import UserModel

class PaperDiscoveryEnv(gym.Env):
    def __init__(self, df, num_recommendations=10):
        super(PaperDiscoveryEnv, self).__init__()
        
        # Data
        self.df = df
        topic_to_index = {topic: idx for idx, topic in enumerate(sorted(set(self.df['topics.display_name'].explode().unique())))}
        self.df['topics.indices'] = self.df['topics.display_name'].apply(lambda topics: clean_data.generate_topic_indices(topics, topic_to_index))
        self.df

        num_topics = len(topic_to_index)
        embedding_dim = 64                 # Each topic is represented by a 64-dimensional vector.
        topic_output_dim = 128             # Intermediate representation dimension for the topic branch.
        final_dim = 256                    # Final paper embedding dimension.
        pad_idx = topic_to_index["<PAD>"]
        numeric_feature_dim = 2            # We use two numeric features: publication_year and cited_by_count

        topic_tensor = torch.tensor(df['topics.indices'].tolist(), dtype=torch.long)
        scores_tensor = torch.tensor(df['topics.score'].tolist(), dtype=torch.float)
        publication_year = torch.tensor(np.array(df['publication_year'].tolist(), dtype=np.float32)).unsqueeze(1)
        cited_by_count = torch.tensor(np.array(df['cited_by_count'].tolist(), dtype=np.float32)).unsqueeze(1)
        numeric_features = torch.cat([publication_year, cited_by_count], dim=1)

        embedding = paper_embedding.PaperEmbeddingModule(num_topics, embedding_dim, topic_output_dim, pad_idx,
                             numeric_feature_dim, final_dim)
        self.paper_pool = embedding(topic_tensor, scores_tensor, numeric_features)
        self.num_recommendations = num_recommendations

        # User
        self.user = UserModel.UserModel()

        # State Space (e.g., paper features: relevance, reputability, novelty)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Action Space (10 papers at a time)
        self.action_space = spaces.MultiDiscrete([len(self.df)] * 10)

        # Track seen papers to avoid recommending duplicates
        self.recommended_papers = set()
        self.clicked_papers = set()
        self.clicked_scores = set()

    def reset(self):
        """Reset environment at the start of a new episode."""
        self.recommended_papers = set()
        self.clicked_papers = set()
        self.clicked_scores = set()
        return self.get_state()

    def get_state(self):
        clicked_embeddings = self.paper_pool[list(self.clicked_papers)]
        clicked_scores_tensor = torch.tensor(list(self.clicked_scores), dtype=torch.float)
        weighted_average_clicked_score = (clicked_embeddings * clicked_scores_tensor.unsqueeze(1)).sum(dim=0) / (clicked_scores_tensor.sum() + 1e-8)
        return weighted_average_clicked_score

    def step(self, action):
        """
        Simulate user engagement and return reward.
        
        action: A list of recommended paper's indexes.
        """

        chosen_idx, chosen_scores  = self.user.choose(self.df.iloc[action])

        reward = len(chosen_idx) - 1 * (self.num_recommendations - len(chosen_idx))

        # Track papers that have already been shown to the user
        self.recommended_papers.update(action)
        self.clicked_papers.update(chosen_idx)
        self.clicked_scores.update(chosen_scores)

        # Generate next state: Random unseen paper
        next_state = self.get_state()

        # Done condition: End episode after 10 recommendations
        done = len(self.recommended_papers) >= self.num_recommendations

        return next_state, reward, done, {}