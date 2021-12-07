-- creates a function SafeDiv that divides (and returns) the first by the second number
DELIMITER //

DROP FUNCTION IF EXISTS SafeDiv;

CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS FLOAT

BEGIN
    IF b = 0 THEN
        RETURN 0;
    END IF;
    RETURN (a / b);
END; //

DELIMITER ;
