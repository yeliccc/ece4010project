CREATE DATABASE IF NOT EXISTS movielens;

USE movielens;

-- 创建 movies 表
CREATE TABLE IF NOT EXISTS movies (
    movieId INT PRIMARY KEY,
    title VARCHAR(255),
    genres VARCHAR(255)
);

-- 创建 ratings 表
CREATE TABLE IF NOT EXISTS ratings (
    userId INT,
    movieId INT,
    rating FLOAT,
    timestamp BIGINT,
    PRIMARY KEY (userId, movieId)
);

--创建 users 表
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
