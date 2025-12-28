-- psqlで: \set q '猫'
SELECT
  id, title,
  pgroonga_score(tableoid, ctid) AS score
FROM documents
WHERE content &@~ :'q'
ORDER BY score DESC
LIMIT 10;
