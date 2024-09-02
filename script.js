const form = document.querySelector('form');
const recommendedMoviesDiv = document.querySelector('#recommended-movies');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const userId = document.querySelector('#user_id').value;
    const numRecs = document.querySelector('#num_recs').value;
    fetch(`/recommendations?user_id=${userId}&num_recs=${numRecs}`)
        .then(response => response.json())
        .then(data => {
            const recommendedMovies = data.recommended_movies;
            recommendedMoviesDiv.innerHTML = '';
            recommendedMovies.forEach((movieId) => {
                const movieDiv = document.createElement('div');
                movieDiv.className = 'movie';
                movieDiv.textContent = `Movie ${movieId}`;
                recommendedMoviesDiv.appendChild(movieDiv);
            });
        });
});