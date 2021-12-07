--  lists all shows contained in hbtn_0d_tvshows that have at least one genre linked
SELECT tv_shows.title, tv_genres.id FROM tv_shows, tv_show_genres, tv_genres
WHERE tv_show_genres.show_id=tv_shows.id and tv_show_genres.genre_id=tv_genres.id
-- SELECT tv_shows.title, tv_genres.id AS genre_id FROM tv_show_genres
-- JOIN tv_shows ON tv_show_genres.show_id=tv_shows.id
-- JOIN tv_genres ON tv_show_genres.genre_id=tv_genres.id
ORDER BY tv_shows.title ASC,  tv_show_genres.genre_id ASC;