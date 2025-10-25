"""
Interactive Demo Application for Movie Recommender System
This script demonstrates the recommender system with clear examples and evaluation.
"""

import sys
from recommender import MovieRecommender
from evaluator import RecommenderEvaluator
from data_loader import DataLoader
import pandas as pd


def print_section_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demonstrate_use_case_1(recommender):
    """
    Use Case 1: Personalized Movie Recommendations for New Weekend Plans
    Input: User ID (existing user in database)
    Output: Top 10 personalized movie recommendations
    """
    print_section_header("USE CASE 1: Personalized Movie Recommendations")
    
    user_id = 196
    print(f"Scenario: User {user_id} is looking for movie recommendations for the weekend.")
    print(f"The system will analyze their viewing history and predict what they'll enjoy.\n")
    
    # Show user's previous preferences
    print(f"📊 User {user_id}'s Viewing History (highly rated movies):")
    user_liked = recommender.data_loader.get_user_ratings(user_id, min_rating=4.0)
    print(user_liked[['title', 'rating']].head(5).to_string(index=False))
    
    # Get recommendations
    print(f"\n🎬 Top 10 Recommended Movies for User {user_id}:")
    recommendations = recommender.recommend_for_user(user_id, n=10)
    
    for idx, row in recommendations.iterrows():
        print(f"  {idx+1}. {row['title']:<50} (Predicted Rating: {row['predicted_rating']:.2f})")
    
    print("\n💡 Interpretation: The system predicts the user will rate these movies highly")
    print("   based on patterns learned from similar users and their rating history.")


def demonstrate_use_case_2(recommender):
    """
    Use Case 2: Similar Movie Discovery
    Input: Movie title/ID
    Output: Similar movies based on latent factors
    """
    print_section_header("USE CASE 2: Similar Movie Discovery")
    
    movie_name = "Toy Story (1995)"
    movie_id = 1
    
    print(f"Scenario: A user loved '{movie_name}' and wants to find similar movies.")
    print(f"The system will find movies with similar characteristics based on user patterns.\n")
    
    print(f"🎯 Input Movie: {movie_name} (ID: {movie_id})")
    
    # Get similar movies
    similar_movies = recommender.recommend_similar_movies(movie_id, n=10)
    
    print(f"\n🎬 Movies Similar to '{movie_name}':")
    for idx, row in similar_movies.iterrows():
        print(f"  {idx+1}. {row['title']:<50} (Similarity: {row['similarity']:.4f})")
    
    print("\n💡 Interpretation: These movies have similar latent factors (patterns in how")
    print("   users rate them), suggesting viewers who enjoyed Toy Story will likely")
    print("   enjoy these recommendations as well.")


def demonstrate_use_case_3(recommender):
    """
    Use Case 3: Cold Start - New User Profile
    Input: Different user profiles
    Output: How recommendations vary by user
    """
    print_section_header("USE CASE 3: Recommendation Diversity Across Users")
    
    print("Scenario: Comparing how the system provides personalized recommendations")
    print("for different users with different taste profiles.\n")
    
    # Select diverse users
    user_ids = [1, 100, 200]
    
    for user_id in user_ids:
        print(f"\n👤 User {user_id}:")
        
        # Show user info
        user_info = recommender.data_loader.get_user_info(user_id)
        if user_info:
            print(f"   Age: {user_info['age']}, Gender: {user_info['gender']}, " +
                  f"Occupation: {user_info['occupation']}")
        
        # Show favorite movie
        user_liked = recommender.data_loader.get_user_ratings(user_id, min_rating=5.0)
        if not user_liked.empty:
            print(f"   Favorite Movie: {user_liked.iloc[0]['title']} (rated {user_liked.iloc[0]['rating']})")
        
        # Get top 3 recommendations
        recommendations = recommender.recommend_for_user(user_id, n=3)
        print(f"   Top 3 Recommendations:")
        for idx, row in recommendations.iterrows():
            print(f"     {idx+1}. {row['title']} ({row['predicted_rating']:.2f})")
    
    print("\n💡 Interpretation: The system provides personalized recommendations that")
    print("   differ based on each user's unique viewing history and preferences.")


def run_evaluation(recommender):
    """Run comprehensive evaluation with all metrics."""
    print_section_header("EVALUATION RESULTS")
    
    evaluator = RecommenderEvaluator(recommender)
    
    print("Running comprehensive evaluation with multiple metrics...")
    print("This validates the recommender system's performance.\n")
    
    results = evaluator.evaluate_all(k=10)
    
    print("\n" + "-" * 80)
    print("📈 SUMMARY OF EVALUATION METRICS")
    print("-" * 80)
    print("\n1. ACCURACY METRICS (Lower is better):")
    print(f"   • RMSE: {results['rmse']:.4f} - Root Mean Square Error of rating predictions")
    print(f"   • MAE:  {results['mae']:.4f} - Mean Absolute Error of rating predictions")
    
    print("\n2. PRECISION & RECALL METRICS (Higher is better):")
    print(f"   • Precision@10: {results['precision@10']:.4f} - Fraction of recommended items that are relevant")
    print(f"   • Recall@10:    {results['recall@10']:.4f} - Fraction of relevant items that are recommended")
    
    print("\n3. RANKING METRICS (Higher is better):")
    print(f"   • MAP@10: {results['map@10']:.4f} - Mean Average Precision at top 10")
    print(f"   • nDCG@10: {results['ndcg@10']:.4f} - Normalized Discounted Cumulative Gain")
    print(f"   • MRR: {results['mrr']:.4f} - Mean Reciprocal Rank")
    
    print("\n4. COVERAGE METRIC:")
    print(f"   • Coverage: {results['coverage']:.4f} ({results['coverage']*100:.2f}%) - Percentage of catalog recommended")
    
    print("\n" + "-" * 80)
    print("✅ Evaluation Complete!")
    print("-" * 80)
    
    # Save visualization
    try:
        evaluator.plot_results('evaluation_results.png')
        print("\n📊 Evaluation visualization saved to 'evaluation_results.png'")
    except Exception as e:
        print(f"\n⚠️  Could not save visualization: {e}")
    
    return results


def main():
    """Main demo execution."""
    print("\n" + "=" * 80)
    print("  🎬 MOVIE RECOMMENDER SYSTEM - INTERACTIVE DEMO")
    print("  Assignment 2: Recommender System Application")
    print("=" * 80)
    
    print("\n📦 Initializing Movie Recommender System...")
    print("   AI Method: Matrix Factorization (SVD - Singular Value Decomposition)")
    print("   Dataset: MovieLens 100K (100,000 ratings from 943 users on 1,682 movies)")
    
    # Initialize and train recommender
    recommender = MovieRecommender(n_factors=100, n_epochs=20)
    
    print("\n🔄 Training the recommender system...")
    print("   This involves learning latent factors for users and movies...")
    recommender.train(test_size=0.2)
    
    # Run use cases
    print("\n" + "=" * 80)
    print("  DEMONSTRATION OF USE CASES")
    print("=" * 80)
    
    demonstrate_use_case_1(recommender)
    demonstrate_use_case_2(recommender)
    demonstrate_use_case_3(recommender)
    
    # Run evaluation
    results = run_evaluation(recommender)
    
    # Final summary
    print_section_header("DEMO COMPLETE")
    print("✅ Successfully demonstrated:")
    print("   1. Personalized movie recommendations for users")
    print("   2. Similar movie discovery")
    print("   3. Recommendation diversity across different user profiles")
    print("   4. Comprehensive evaluation with accuracy and ranking metrics")
    
    print("\n📄 For detailed assignment answers, see 'ASSIGNMENT_ANSWERS.md'")
    print("📖 For implementation details, see 'README.md'")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
