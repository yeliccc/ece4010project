DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS movies;

-- 创建movies表
CREATE TABLE movies (
    movieId INT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    genres VARCHAR(255)
);

-- 创建ratings表
CREATE TABLE ratings (
    userId INT,
    movieId INT,
    rating FLOAT,
    timestamp BIGINT,
    PRIMARY KEY (userId, movieId),
    FOREIGN KEY (movieId) REFERENCES movies(movieId)
);