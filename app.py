from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movie ratings data
ratings = pd.read_csv('ratings.csv')

# Create a user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Calculate similarity between users
user_similarity = cosine_similarity(user_item_matrix)

def get_recommendations(user_id, num_recs=5):
    # Get similar users
    similar_users = user_similarity[user_id].argsort()[:-num_recs-1:-1]
    
    # Get movies liked by similar users
    liked_movies = user_item_matrix.iloc[similar_users].sum(axis=0)
    
    # Get top recommended movies
    recommended_movies = liked_movies.nlargest(num_recs)
    
    return recommended_movies.index.tolist()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def get_recommendations_api():
    user_id = int(request.args.get('user_id'))
    num_recs = int(request.args.get('num_recs', 5))
    recommended_movies = get_recommendations(user_id, num_recs)
    return jsonify({'recommended_movies': recommended_movies})

if __name__ == '__main__':
    app.run(debug=True)