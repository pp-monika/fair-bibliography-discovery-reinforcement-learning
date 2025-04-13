import gym
from gym import spaces
import numpy as np
import random

class UserModel:
    def __init__(self, interests=("scientometrics and bibliometrics research")):
        super(UserModel, self).__init__()

        # Should be in the form of 
        self.interests = interests
        self.current_year = 2025

    def choose(self, displayed_papers, threshold=0.6):
        """
        ** User interests currently doesn't change

        Simulates user engagement by choosing any number of paper among the displayed ones.
        The decision is based on:
          - Relevance: Higher means more attractive.
          - Reputation: Papers with higher normalized citations are favored.
          - Age: Older (not newer) papers are rewarded, simulating a bias against new papers.
          
        The final score is computed as:
        
            score = 0.5 * relevance + 0.3 * normalized_citations + 0.2 * age
            
        where age is computed as (current_year - publication_year).
        
        Returns:
            chosen_index (int): The index of the selected paper from displayed_papers.
            scores (list): The list of computed scores for each paper.
        """

        chosen_dict = {}

        for idx, paper in displayed_papers.iterrows():
            paper_topics = paper['topics.display_name']
            paper_scores = paper['topics.score']
            valid_topics = [topic for topic in paper_topics if topic != "<PAD>"]
            valid_scores = [score for topic, score in zip(paper_topics, paper_scores) if topic != "<PAD>"]

            matched_score_sum = sum(score for topic, score in zip(valid_topics, valid_scores)
                                    if topic in self.interests)
            reputability = paper['cited_by_count_norm']
            publication_year = paper['publication_year']
            age = self.current_year - publication_year

            # Asymmetric score: penalize too-new heavily, too-old mildly
            if age < 3:
                age_score = 0.1  # heavily penalized
            elif 3 <= age <= 15:
                age_score = 1.0  # ideal window
            elif 15 < age <= 25:
                age_score = 0.7  # slightly decayed
            else:
                age_score = 0.4


            combined_score = 0.5 * matched_score_sum + 0.4 * reputability + 0.1 * age_score
            # print(f"idx = {idx}; relevance = {matched_score_sum}, reputability = {reputability}, age_score = {age_score}, total score = {combined_score}")

            if combined_score >= threshold:
                chosen_dict[idx] = combined_score
            
        # If no paper pass the threshold, don't choose anything      
        return chosen_dict


        

        