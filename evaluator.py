"""
Evaluation Module for Movie Recommender System
Implements various metrics: RMSE, Precision@K, MAP, nDCG, MRR
"""

import numpy as np
import pandas as pd
from surprise import accuracy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class RecommenderEvaluator:
    """
    Evaluates recommender system performance using multiple metrics.
    """
    
    def __init__(self, recommender):
        """
        Initialize evaluator with a trained recommender.
        
        Args:
            recommender: Trained MovieRecommender instance
        """
        self.recommender = recommender
        self.results = {}
    
    def calculate_rmse(self):
        """
        Calculate Root Mean Square Error on test set.
        
        Returns:
            RMSE value
        """
        if self.recommender.testset is None:
            raise ValueError("No test set available")
        
        predictions = self.recommender.algo.test(self.recommender.testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        
        self.results['rmse'] = rmse
        return rmse
    
    def calculate_mae(self):
        """
        Calculate Mean Absolute Error on test set.
        
        Returns:
            MAE value
        """
        if self.recommender.testset is None:
            raise ValueError("No test set available")
        
        predictions = self.recommender.algo.test(self.recommender.testset)
        mae = accuracy.mae(predictions, verbose=False)
        
        self.results['mae'] = mae
        return mae
    
    def precision_recall_at_k(self, k=10, threshold=4.0):
        """
        Calculate Precision@K and Recall@K.
        
        Args:
            k: Number of top recommendations to consider
            threshold: Rating threshold to consider as relevant
            
        Returns:
            Dictionary with precision and recall values
        """
        # Get predictions for test set
        predictions = self.recommender.algo.test(self.recommender.testset)
        
        # Group predictions by user
        user_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((iid, true_r, est))
        
        precisions = []
        recalls = []
        
        for uid, user_ratings in user_predictions.items():
            # Sort by estimated rating
            user_ratings.sort(key=lambda x: x[2], reverse=True)
            
            # Get top k recommendations
            top_k = user_ratings[:k]
            
            # Count relevant items in top k
            n_rel_and_rec = sum(1 for (_, true_r, _) in top_k if true_r >= threshold)
            
            # Count total relevant items
            n_rel = sum(1 for (_, true_r, _) in user_ratings if true_r >= threshold)
            
            # Calculate precision and recall
            precision = n_rel_and_rec / k if k > 0 else 0
            recall = n_rel_and_rec / n_rel if n_rel > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        self.results[f'precision@{k}'] = avg_precision
        self.results[f'recall@{k}'] = avg_recall
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) 
                       if (avg_precision + avg_recall) > 0 else 0
        }
    
    def mean_average_precision(self, k=10, threshold=4.0):
        """
        Calculate Mean Average Precision (MAP@K).
        
        Args:
            k: Number of top recommendations to consider
            threshold: Rating threshold to consider as relevant
            
        Returns:
            MAP@K value
        """
        predictions = self.recommender.algo.test(self.recommender.testset)
        
        # Group predictions by user
        user_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((iid, true_r, est))
        
        average_precisions = []
        
        for uid, user_ratings in user_predictions.items():
            # Sort by estimated rating (descending)
            user_ratings.sort(key=lambda x: x[2], reverse=True)
            
            # Calculate average precision for this user
            relevant_count = 0
            precision_sum = 0.0
            
            for i, (_, true_r, _) in enumerate(user_ratings[:k], 1):
                if true_r >= threshold:
                    relevant_count += 1
                    precision_at_i = relevant_count / i
                    precision_sum += precision_at_i
            
            # Average precision for this user
            if relevant_count > 0:
                avg_precision = precision_sum / relevant_count
            else:
                avg_precision = 0.0
            
            average_precisions.append(avg_precision)
        
        map_score = np.mean(average_precisions)
        self.results[f'map@{k}'] = map_score
        
        return map_score
    
    def ndcg_at_k(self, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@K).
        
        Args:
            k: Number of top recommendations to consider
            
        Returns:
            nDCG@K value
        """
        predictions = self.recommender.algo.test(self.recommender.testset)
        
        # Group predictions by user
        user_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((iid, true_r, est))
        
        ndcg_scores = []
        
        for uid, user_ratings in user_predictions.items():
            # Sort by estimated rating (descending)
            user_ratings_sorted = sorted(user_ratings, key=lambda x: x[2], reverse=True)
            
            # Get top k recommendations
            top_k = user_ratings_sorted[:k]
            
            # Calculate DCG
            dcg = 0.0
            for i, (_, true_r, _) in enumerate(top_k, 1):
                dcg += (2**true_r - 1) / np.log2(i + 1)
            
            # Calculate IDCG (ideal DCG)
            # Sort by true ratings to get ideal ranking
            ideal_sorted = sorted(user_ratings, key=lambda x: x[1], reverse=True)
            ideal_top_k = ideal_sorted[:k]
            
            idcg = 0.0
            for i, (_, true_r, _) in enumerate(ideal_top_k, 1):
                idcg += (2**true_r - 1) / np.log2(i + 1)
            
            # Calculate nDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcg_scores.append(ndcg)
        
        avg_ndcg = np.mean(ndcg_scores)
        self.results[f'ndcg@{k}'] = avg_ndcg
        
        return avg_ndcg
    
    def mean_reciprocal_rank(self, threshold=4.0):
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            threshold: Rating threshold to consider as relevant
            
        Returns:
            MRR value
        """
        predictions = self.recommender.algo.test(self.recommender.testset)
        
        # Group predictions by user
        user_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((iid, true_r, est))
        
        reciprocal_ranks = []
        
        for uid, user_ratings in user_predictions.items():
            # Sort by estimated rating (descending)
            user_ratings.sort(key=lambda x: x[2], reverse=True)
            
            # Find rank of first relevant item
            for rank, (_, true_r, _) in enumerate(user_ratings, 1):
                if true_r >= threshold:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # No relevant item found
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks)
        self.results['mrr'] = mrr
        
        return mrr
    
    def coverage(self, k=10):
        """
        Calculate catalog coverage - percentage of items recommended.
        
        Args:
            k: Number of top recommendations per user
            
        Returns:
            Coverage percentage
        """
        # Get all users
        all_users = set(self.recommender.trainset.all_users())
        
        # Track all recommended items
        recommended_items = set()
        
        for inner_uid in list(all_users)[:100]:  # Sample 100 users for efficiency
            uid = self.recommender.trainset.to_raw_uid(inner_uid)
            recommendations = self.recommender.recommend_for_user(uid, n=k)
            recommended_items.update(recommendations['item_id'].values)
        
        # Calculate coverage
        total_items = len(self.recommender.data_loader.movies_df)
        coverage = len(recommended_items) / total_items
        
        self.results['coverage'] = coverage
        return coverage
    
    def evaluate_all(self, k=10):
        """
        Run all evaluation metrics.
        
        Args:
            k: Number of recommendations to consider
            
        Returns:
            Dictionary with all metrics
        """
        print("=== Evaluating Recommender System ===\n")
        
        print("Calculating RMSE and MAE...")
        rmse = self.calculate_rmse()
        mae = self.calculate_mae()
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        print(f"\nCalculating Precision@{k} and Recall@{k}...")
        pr_results = self.precision_recall_at_k(k=k)
        print(f"  Precision@{k}: {pr_results['precision']:.4f}")
        print(f"  Recall@{k}: {pr_results['recall']:.4f}")
        print(f"  F1@{k}: {pr_results['f1_score']:.4f}")
        
        print(f"\nCalculating MAP@{k}...")
        map_score = self.mean_average_precision(k=k)
        print(f"  MAP@{k}: {map_score:.4f}")
        
        print(f"\nCalculating nDCG@{k}...")
        ndcg = self.ndcg_at_k(k=k)
        print(f"  nDCG@{k}: {ndcg:.4f}")
        
        print("\nCalculating MRR...")
        mrr = self.mean_reciprocal_rank()
        print(f"  MRR: {mrr:.4f}")
        
        print("\nCalculating Coverage...")
        coverage = self.coverage(k=k)
        print(f"  Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
        
        return self.results
    
    def plot_results(self, save_path='evaluation_results.png'):
        """
        Visualize evaluation results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No results to plot. Run evaluate_all() first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Recommender System Evaluation Results', fontsize=16)
        
        # Plot 1: Accuracy Metrics (RMSE, MAE)
        accuracy_metrics = {k: v for k, v in self.results.items() if k in ['rmse', 'mae']}
        if accuracy_metrics:
            axes[0, 0].bar(accuracy_metrics.keys(), accuracy_metrics.values(), color='skyblue')
            axes[0, 0].set_title('Accuracy Metrics')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].tick_params(axis='x', rotation=0)
        
        # Plot 2: Precision/Recall/F1
        pr_metrics = {k: v for k, v in self.results.items() 
                     if any(x in k for x in ['precision', 'recall'])}
        if pr_metrics:
            axes[0, 1].bar(pr_metrics.keys(), pr_metrics.values(), color='lightcoral')
            axes[0, 1].set_title('Precision, Recall & F1 Metrics')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Ranking Metrics (MAP, nDCG, MRR)
        ranking_metrics = {k: v for k, v in self.results.items() 
                          if any(x in k for x in ['map', 'ndcg', 'mrr'])}
        if ranking_metrics:
            axes[1, 0].bar(ranking_metrics.keys(), ranking_metrics.values(), color='lightgreen')
            axes[1, 0].set_title('Ranking Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Coverage
        if 'coverage' in self.results:
            axes[1, 1].bar(['Coverage'], [self.results['coverage']], color='plum')
            axes[1, 1].set_title('Catalog Coverage')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.close()


if __name__ == '__main__':
    from recommender import MovieRecommender
    
    print("=== Testing Evaluator ===\n")
    
    # Train recommender
    recommender = MovieRecommender()
    recommender.train(test_size=0.2)
    
    # Evaluate
    evaluator = RecommenderEvaluator(recommender)
    results = evaluator.evaluate_all(k=10)
    
    # Plot results
    evaluator.plot_results()
