"""
Movie Recommender System using Matrix Factorization (SVD)
This module implements a collaborative filtering recommender using the Surprise library.
"""

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import pickle
import os
from data_loader import DataLoader


class MovieRecommender:
    """
    Collaborative Filtering Movie Recommender using Matrix Factorization (SVD).
    
    This recommender uses Singular Value Decomposition (SVD) to learn latent factors
    for users and items from the rating matrix, enabling personalized recommendations.
    """
    
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Initialize the recommender system.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            lr_all: Learning rate for all parameters
            reg_all: Regularization term for all parameters
        """
        self.algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        
        self.data_loader = DataLoader()
        self.trainset = None
        self.testset = None
        self.is_trained = False
        
    def train(self, test_size=0.2):
        """
        Train the recommender system.
        
        Args:
            test_size: Proportion of data to use for testing
        """
        print("Loading data...")
        self.trainset, self.testset = self.data_loader.get_train_test_split(test_size=test_size)
        
        print("\nTraining SVD model...")
        print(f"Parameters: n_factors={self.algo.n_factors}, n_epochs={self.algo.n_epochs}")
        
        self.algo.fit(self.trainset)
        self.is_trained = True
        
        print("Training complete!")
        
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item (movie) ID
            
        Returns:
            Prediction object with estimated rating
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.algo.predict(user_id, item_id)
    
    def recommend_for_user(self, user_id, n=10, exclude_rated=True):
        """
        Generate top-N movie recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations to generate
            exclude_rated: Whether to exclude movies the user has already rated
            
        Returns:
            DataFrame with recommended movies and predicted ratings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Get all movies
        all_movies = self.data_loader.movies_df['item_id'].unique()
        
        # Get movies already rated by user
        if exclude_rated:
            user_ratings = self.data_loader.ratings_df[
                self.data_loader.ratings_df['user_id'] == user_id
            ]['item_id'].unique()
            movies_to_predict = [m for m in all_movies if m not in user_ratings]
        else:
            movies_to_predict = all_movies
        
        # Predict ratings for all candidate movies
        predictions = []
        for item_id in movies_to_predict:
            pred = self.algo.predict(user_id, item_id)
            predictions.append({
                'item_id': item_id,
                'predicted_rating': pred.est
            })
        
        # Convert to DataFrame and sort by predicted rating
        recommendations_df = pd.DataFrame(predictions)
        recommendations_df = recommendations_df.sort_values(
            'predicted_rating', 
            ascending=False
        ).head(n)
        
        # Add movie titles
        recommendations_df = recommendations_df.merge(
            self.data_loader.movies_df[['item_id', 'title']],
            on='item_id',
            how='left'
        )
        
        return recommendations_df[['item_id', 'title', 'predicted_rating']].reset_index(drop=True)
    
    def recommend_similar_movies(self, item_id, n=10):
        """
        Find similar movies based on latent factor similarity.
        
        Args:
            item_id: Item (movie) ID
            n: Number of similar movies to return
            
        Returns:
            DataFrame with similar movies
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar items")
        
        # Get the item's latent factors
        try:
            inner_id = self.trainset.to_inner_iid(item_id)
            item_factors = self.algo.qi[inner_id]
        except ValueError:
            return pd.DataFrame()  # Item not in training set
        
        # Calculate similarity with all other items
        similarities = []
        for other_item_id in self.data_loader.movies_df['item_id'].unique():
            if other_item_id == item_id:
                continue
            
            try:
                other_inner_id = self.trainset.to_inner_iid(other_item_id)
                other_factors = self.algo.qi[other_inner_id]
                
                # Cosine similarity
                similarity = np.dot(item_factors, other_factors) / (
                    np.linalg.norm(item_factors) * np.linalg.norm(other_factors)
                )
                
                similarities.append({
                    'item_id': other_item_id,
                    'similarity': similarity
                })
            except ValueError:
                continue
        
        # Convert to DataFrame and sort
        similar_df = pd.DataFrame(similarities)
        similar_df = similar_df.sort_values('similarity', ascending=False).head(n)
        
        # Add movie titles
        similar_df = similar_df.merge(
            self.data_loader.movies_df[['item_id', 'title']],
            on='item_id',
            how='left'
        )
        
        return similar_df[['item_id', 'title', 'similarity']].reset_index(drop=True)
    
    def get_top_n_for_all_users(self, n=10):
        """
        Generate top-N recommendations for all users in the training set.
        
        Args:
            n: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to list of (item_id, rating) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get all users in trainset
        all_users = [self.trainset.to_raw_uid(inner_id) 
                     for inner_id in range(self.trainset.n_users)]
        
        top_n = {}
        for user_id in all_users:
            recommendations = self.recommend_for_user(user_id, n=n, exclude_rated=True)
            top_n[user_id] = [
                (row['item_id'], row['predicted_rating'])
                for _, row in recommendations.iterrows()
            ]
        
        return top_n
    
    def save_model(self, filepath='model/movie_recommender.pkl'):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.algo, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model/movie_recommender.pkl'):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            self.algo = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    # Test the recommender
    print("=== Testing Movie Recommender ===\n")
    
    recommender = MovieRecommender()
    recommender.train(test_size=0.2)
    
    # Test user recommendations
    user_id = 196
    print(f"\n=== Recommendations for User {user_id} ===")
    
    # Show what user has liked
    user_liked = recommender.data_loader.get_user_ratings(user_id, min_rating=4.0)
    print(f"\nUser {user_id} has highly rated ({len(user_liked)} movies):")
    print(user_liked[['title', 'rating']].head(5))
    
    # Get recommendations
    recommendations = recommender.recommend_for_user(user_id, n=10)
    print(f"\nTop 10 Recommendations:")
    print(recommendations)
    
    # Test similar movies
    print(f"\n=== Movies Similar to 'Toy Story (1995)' ===")
    toy_story_id = 1
    similar_movies = recommender.recommend_similar_movies(toy_story_id, n=5)
    print(similar_movies)
