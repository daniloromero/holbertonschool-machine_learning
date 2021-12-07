-- creates a stored procedure ComputeAverageScoreForUser
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id INT)
BEGIN
    SET @avg_scr = (SELECT AVG(corrections.score) FROM corrections WHERE corrections.user_id = user_id);
    UPDATE users
        SET average_score = @avg_scr WHERE id = user_id;
END //
DELIMITER  ;
