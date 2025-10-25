"""
Data Loading and Preprocessing Module
Downloads and prepares the MovieLens 100K dataset for the recommender system.
"""

import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import requests
import zipfile
from io import BytesIO


class DataLoader:
    """Handles data loading and preprocessing for MovieLens dataset."""
    
    def __init__(self, data_dir='data'):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.surprise_data = None
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def download_movielens_100k(self):
        """Download MovieLens 100K dataset if not already present."""
        ratings_path = os.path.join(self.data_dir, 'u.data')
        movies_path = os.path.join(self.data_dir, 'u.item')
        users_path = os.path.join(self.data_dir, 'u.user')
        
        # Check if data already exists
        if os.path.exists(ratings_path) and os.path.exists(movies_path):
            print("Dataset already exists. Loading from local files...")
            return
        
        print("Downloading MovieLens 100K dataset...")
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                # Extract necessary files
                zip_file.extract('ml-100k/u.data', self.data_dir)
                zip_file.extract('ml-100k/u.item', self.data_dir)
                zip_file.extract('ml-100k/u.user', self.data_dir)
                
                # Move files to data directory
                for file in ['u.data', 'u.item', 'u.user']:
                    src = os.path.join(self.data_dir, 'ml-100k', file)
                    dst = os.path.join(self.data_dir, file)
                    os.rename(src, dst)
                
                # Remove empty ml-100k directory
                os.rmdir(os.path.join(self.data_dir, 'ml-100k'))
            
            print("Dataset downloaded successfully!")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from: https://grouplens.org/datasets/movielens/100k/")
            raise
    
    def load_data(self):
        """Load all datasets into pandas DataFrames."""
        # Download data if needed
        self.download_movielens_100k()
        
        # Load ratings data
        ratings_path = os.path.join(self.data_dir, 'u.data')
        self.ratings_df = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Load movies data
        movies_path = os.path.join(self.data_dir, 'u.item')
        self.movies_df = pd.read_csv(
            movies_path,
            sep='|',
            names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                  [f'genre_{i}' for i in range(19)],
            encoding='latin-1',
            usecols=[0, 1]  # Only load item_id and title
        )
        
        # Load users data
        users_path = os.path.join(self.data_dir, 'u.user')
        self.users_df = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
        
        print(f"Loaded {len(self.ratings_df)} ratings")
        print(f"Loaded {len(self.movies_df)} movies")
        print(f"Loaded {len(self.users_df)} users")
        
        return self.ratings_df, self.movies_df, self.users_df
    
    def prepare_surprise_data(self):
        """Prepare data in Surprise library format."""
        if self.ratings_df is None:
            self.load_data()
        
        # Define a Reader with the rating scale
        reader = Reader(rating_scale=(1, 5))
        
        # Load data into Surprise format
        self.surprise_data = Dataset.load_from_df(
            self.ratings_df[['user_id', 'item_id', 'rating']],
            reader
        )
        
        return self.surprise_data
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            trainset, testset: Train and test data in Surprise format
        """
        if self.surprise_data is None:
            self.prepare_surprise_data()
        
        trainset, testset = train_test_split(
            self.surprise_data,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"Training set: {trainset.n_ratings} ratings")
        print(f"Test set: {len(testset)} ratings")
        
        return trainset, testset
    
    def get_movie_title(self, item_id):
        """Get movie title by item ID."""
        if self.movies_df is None:
            self.load_data()
        
        movie = self.movies_df[self.movies_df['item_id'] == item_id]
        if not movie.empty:
            return movie.iloc[0]['title']
        return f"Unknown Movie (ID: {item_id})"
    
    def get_user_info(self, user_id):
        """Get user information by user ID."""
        if self.users_df is None:
            self.load_data()
        
        user = self.users_df[self.users_df['user_id'] == user_id]
        if not user.empty:
            return user.iloc[0].to_dict()
        return None
    
    def get_user_ratings(self, user_id, min_rating=4.0):
        """
        Get highly-rated movies for a user.
        
        Args:
            user_id: User ID
            min_rating: Minimum rating threshold
            
        Returns:
            DataFrame with user's highly-rated movies
        """
        if self.ratings_df is None:
            self.load_data()
        
        user_ratings = self.ratings_df[
            (self.ratings_df['user_id'] == user_id) & 
            (self.ratings_df['rating'] >= min_rating)
        ]
        
        # Merge with movie titles
        user_ratings = user_ratings.merge(
            self.movies_df[['item_id', 'title']],
            on='item_id',
            how='left'
        )
        
        return user_ratings.sort_values('rating', ascending=False)
    
    def get_dataset_statistics(self):
        """Get basic statistics about the dataset."""
        if self.ratings_df is None:
            self.load_data()
        
        stats = {
            'num_users': self.ratings_df['user_id'].nunique(),
            'num_items': self.ratings_df['item_id'].nunique(),
            'num_ratings': len(self.ratings_df),
            'rating_scale': (self.ratings_df['rating'].min(), self.ratings_df['rating'].max()),
            'avg_rating': self.ratings_df['rating'].mean(),
            'sparsity': 1 - (len(self.ratings_df) / (
                self.ratings_df['user_id'].nunique() * 
                self.ratings_df['item_id'].nunique()
            ))
        }
        
        return stats


if __name__ == '__main__':
    # Test the data loader
    loader = DataLoader()
    ratings_df, movies_df, users_df = loader.load_data()
    
    print("\n=== Dataset Statistics ===")
    stats = loader.get_dataset_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Sample Ratings ===")
    print(ratings_df.head(10))
    
    print("\n=== Sample Movies ===")
    print(movies_df.head(10))
